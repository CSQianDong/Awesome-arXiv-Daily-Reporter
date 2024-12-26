# Long-Form Speech Generation with Spoken Language Models 

**Title (ZH)**: 使用口语语言模型生成长段语音内容 

**Authors**: Se Jin Park, Julian Salazar, Aren Jansen, Keisuke Kinoshita, Yong Man Ro, RJ Skerry-Ryan  

**Link**: [PDF](https://arxiv.org/pdf/2412.18603)  

**Abstract**: We consider the generative modeling of speech over multiple minutes, a requirement for long-form multimedia generation and audio-native voice assistants. However, current spoken language models struggle to generate plausible speech past tens of seconds, from high temporal resolution of speech tokens causing loss of coherence, to architectural issues with long-sequence training or extrapolation, to memory costs at inference time. With these considerations we propose SpeechSSM, the first speech language model to learn from and sample long-form spoken audio (e.g., 16 minutes of read or extemporaneous speech) in a single decoding session without text intermediates, based on recent advances in linear-time sequence modeling. Furthermore, to address growing challenges in spoken language evaluation, especially in this new long-form setting, we propose: new embedding-based and LLM-judged metrics; quality measurements over length and time; and a new benchmark for long-form speech processing and generation, LibriSpeech-Long. Speech samples and the dataset are released at this https URL 

**Abstract (ZH)**: 我们研究了跨多分钟的语音生成模型，这对长格式多媒体生成和原生音频语音助手至关重要。然而，当前的语言模型在超过几十秒的时间内生成合乎逻辑的语音时存在困难，原因包括语音 token 的高时间分辨率导致连贯性丧失，架构问题如长序列训练或外推，以及推理时的记忆成本。基于这些考虑，我们提出了 SpeechSSM，这是首个能够在一个解码会话中从头学习并生成长期形式的口语音频（例如，16 分钟的朗读或即兴演讲）的语言模型，无需经过文本中介，同时利用了近期在线性时间序列建模方面的进展。此外，为应对日益增长的口语语言评价挑战，尤其是在这一新的长期形式中，我们提出了新的基于嵌入的和大规模语言模型（LLM）评判的度量标准；长时间和宽范围的质量测量；以及一个新的长格式语音处理和生成基准，LibriSpeech-Long。数据集和语音样本在此处发布：[提供链接] 

---
# Exploring Embedding Priors in Prompt-Tuning for Improved Interpretability and Control 

**Title (ZH)**: 探索嵌入先验知识以提高提示调优的可解释性和控制能力 

**Authors**: Sergey Sedov, Sumanth Bharadwaj Hachalli Karanam, Venu Gopal Kadamba  

**Link**: [PDF](https://arxiv.org/pdf/2412.18582)  

**Abstract**: Prompt-Tuning is an efficient method for adapting pre-trained language models to new tasks with minimal computational overhead by modifying prompt embeddings. In this work, we investigate how crucial the phenomenon of embedding collapse, frequently observed in Prompt-Tuning, is for the final performance of the model. To address this question, we designed embedding priors and compared them with posteriors of the converged Soft and Deep Prompt-Tuning methods. Our findings suggest that priors strongly affect the position of the tuned embeddings, and models can effectively work with embeddings from different parts of activation spaces, including completely new regions. As the final Prompt-Tuning capabilities are limited, we hypothesize that controllable Prompt-Tuning posteriors may serve as a good starting point for tasks such as chain-of-thought (COT) distillation. Our experiments also show that generated trajectories are not localized in the activation space of the models. However, there are distinct clusters of activations for distant tasks (e.g., NLP and arithmetic), while activations between NLP tasks (e.g., Question-Answering and MLM) lie in the same cluster. These observations raise questions about the importance of a single activation cluster for the generalization abilities of large language models. 

**Abstract (ZH)**: 提示调优是一种有效的方法，通过修改提示嵌入来以最小的计算开销将预训练语言模型适应到新任务。在这项研究中，我们探讨了提示调优中经常观察到的嵌入塌陷现象对最终模型性能的重要性。为了回答这一问题，我们设计了嵌入先验，并将其与收敛的柔性提示调优和深层提示调优的方法的后验进行了比较。我们的研究结果表明，先验强烈影响调优嵌入的位置，模型可以有效处理来自不同激活空间区域的嵌入，包括全新的区域。鉴于提示调优能力的限制，我们假设可控的提示调优后验可能成为像链式思考（COT）蒸馏等任务的良好起点。我们的实验还表明，生成的轨迹并不限定于模型的激活空间。然而，对于遥远的任务（如NLP和算术），存在不同的激活簇，而NLP任务之间（如问答与掩码语言模型）的激活则位于同一个簇中。这些观察结果提出了关于大型语言模型泛化能力是否依赖于单一激活簇的重要问题。 

---
# Zero-resource Speech Translation and Recognition with LLMs 

**Title (ZH)**: 使用大语言模型的零资源语音翻译与识别 

**Authors**: Karel Mundnich, Xing Niu, Prashant Mathur, Srikanth Ronanki, Brady Houston, Veera Raghavendra Elluru, Nilaksh Das, Zejiang Hou, Goeric Huybrechts, Anshu Bhatia, Daniel Garcia-Romero, Kyu J. Han, Katrin Kirchhoff  

**Link**: [PDF](https://arxiv.org/pdf/2412.18566)  

**Abstract**: Despite recent advancements in speech processing, zero-resource speech translation (ST) and automatic speech recognition (ASR) remain challenging problems. In this work, we propose to leverage a multilingual Large Language Model (LLM) to perform ST and ASR in languages for which the model has never seen paired audio-text data. We achieve this by using a pre-trained multilingual speech encoder, a multilingual LLM, and a lightweight adaptation module that maps the audio representations to the token embedding space of the LLM. We perform several experiments both in ST and ASR to understand how to best train the model and what data has the most impact on performance in previously unseen languages. In ST, our best model is capable to achieve BLEU scores over 23 in CoVoST2 for two previously unseen languages, while in ASR, we achieve WERs of up to 28.2\%. We finally show that the performance of our system is bounded by the ability of the LLM to output text in the desired language. 

**Abstract (ZH)**: 尽管近期在语音处理方面取得了进展，零资源语音翻译（ST）和自动语音识别（ASR）仍然是具有挑战性的问题。在本研究中，我们提出利用多语言大型语言模型（LLM）来处理那些模型从未见过配对音频-文本数据的语言的ST和ASR任务。我们通过使用一个预训练的多语言语音编码器、一个多语言LLM以及一个轻量级的适应模块（该模块将音频表示映射到LLM的标记嵌入空间）来实现这一点。我们分别在ST和ASR中进行了多项实验，以了解如何最好地训练模型，并分析对未见过语言的性能有何影响。在ST任务中，我们的最佳模型能在CoVoST2数据集中实现超过23的BLEU分数，对于两种未见过的语言。而在ASR任务中，我们达到了高达28.2%的WER。最后，我们展示了该系统的性能受限于LLM能够输出目标语言文本的能力。 

---
# Distilling Fine-grained Sentiment Understanding from Large Language Models 

**Title (ZH)**: 从大规模语言模型中提炼细粒度情感理解 

**Authors**: Yice Zhang, Guangyu Xie, Hongling Xu, Kaiheng Hou, Jianzhu Bao, Qianlong Wang, Shiwei Chen, Ruifeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2412.18552)  

**Abstract**: Fine-grained sentiment analysis (FSA) aims to extract and summarize user opinions from vast opinionated text. Recent studies demonstrate that large language models (LLMs) possess exceptional sentiment understanding capabilities. However, directly deploying LLMs for FSA applications incurs high inference costs. Therefore, this paper investigates the distillation of fine-grained sentiment understanding from LLMs into small language models (SLMs). We prompt LLMs to examine and interpret the sentiments of given reviews and then utilize the generated content to pretrain SLMs. Additionally, we develop a comprehensive FSA benchmark to evaluate both SLMs and LLMs. Extensive experiments on this benchmark reveal that: (1) distillation significantly enhances the performance of SLMs in FSA tasks, achieving a 6.00\% improvement in $F_1$-score, and the distilled model can outperform Llama-2-7b with only 220M parameters; (2) distillation equips SLMs with excellent zero-shot sentiment classification capabilities, enabling them to match or even exceed their teacher models. These results suggest that distillation from LLMs is a highly promising direction for FSA. We will release our code, data, and pretrained model weights at \url{this https URL}. 

**Abstract (ZH)**: 精细粒度情感分析（Fine-grained Sentiment Analysis, FSA）旨在从大量的意见性文本中提取和总结用户观点。近期的研究表明，大型语言模型（Large Language Models, LLMs）具有出色的情感理解能力。然而，直接将LLMs部署到FSA应用中会产生较高的推理成本。因此，本文探讨了将LLMs的情感理解能力精简到小型语言模型（Small Language Models, SLMs）中的方法。我们促使LLMs审查和解释给定评论的情感，并利用生成的内容对SLMs进行预训练。此外，我们还开发了一个全面的FSA基准，用于评估SLMs和LLMs。在该基准上的广泛实验表明：(1) 精简显著提高了SLMs在FSA任务中的性能，实现了$F_1$-分数6.00%的提升，并且精简后的模型在仅有220M参数的情况下仍能优于Llama-2-7b；(2) 精简赋予了SLMs出色的零样本情感分类能力，使它们能够与甚至超越其教师模型。这些结果表明，从LLMs中提取情感理解能力是一种非常有前景的方向。我们将在 \url{this https URL} 释放我们的代码、数据和预训练模型权重。 

---
# Libra-Leaderboard: Towards Responsible AI through a Balanced Leaderboard of Safety and Capability 

**Title (ZH)**: Libra-排行榜：通过安全与能力平衡的排行榜迈向负责任的人工智能 

**Authors**: Haonan Li, Xudong Han, Zenan Zhai, Honglin Mu, Hao Wang, Zhenxuan Zhang, Yilin Geng, Shom Lin, Renxi Wang, Artem Shelmanov, Xiangyu Qi, Yuxia Wang, Donghai Hong, Youliang Yuan, Meng Chen, Haoqin Tu, Fajri Koto, Tatsuki Kuribayashi, Cong Zeng, Rishabh Bhardwaj, Bingchen Zhao, Yawen Duan, Yi Liu, Emad A. Alghamdi, Yaodong Yang, Yinpeng Dong, Soujanya Poria, Pengfei Liu, Zhengzhong Liu, Xuguang Ren, Eduard Hovy, Iryna Gurevych, Preslav Nakov, Monojit Choudhury, Timothy Baldwin  

**Link**: [PDF](https://arxiv.org/pdf/2412.18551)  

**Abstract**: To address this gap, we introduce Libra-Leaderboard, a comprehensive framework designed to rank LLMs through a balanced evaluation of performance and safety. Combining a dynamic leaderboard with an interactive LLM arena, Libra-Leaderboard encourages the joint optimization of capability and safety. Unlike traditional approaches that average performance and safety metrics, Libra-Leaderboard uses a distance-to-optimal-score method to calculate the overall rankings. This approach incentivizes models to achieve a balance rather than excelling in one dimension at the expense of some other ones. In the first release, Libra-Leaderboard evaluates 26 mainstream LLMs from 14 leading organizations, identifying critical safety challenges even in state-of-the-art models. 

**Abstract (ZH)**: 为了填补这一空白，我们提出了Libra-Leaderboard，这是一个全面的框架，旨在通过对性能和安全性的平衡评估来对语言模型进行排名。Libra-Leaderboard 结合了一个动态排行榜和一个交互式的语言模型竞技场，鼓励对能力和安全性的联合优化。与传统的平均性能和安全性指标的方法不同，Libra-Leaderboard 使用了一种基于“距离最优得分”的方法来计算总体排名。这种方法激励模型在不同维度上追求平衡，而不是仅仅在某个维度上表现出色而牺牲其他方面。在首次发布中，Libra-Leaderboard 评估了来自14家领先组织的26个主流语言模型，并在最先进的模型中也识别出了关键的安全挑战。 

---
# Token-Budget-Aware LLM Reasoning 

**Title (ZH)**: 面向token预算的大型语言模型推理 

**Authors**: Tingxu Han, Chunrong Fang, Shiyu Zhao, Shiqing Ma, Zhenyu Chen, Zhenting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.18547)  

**Abstract**: Reasoning is critical for large language models (LLMs) to excel in a wide range of tasks. While methods like Chain-of-Thought (CoT) reasoning enhance LLM performance by decomposing problems into intermediate steps, they also incur significant overhead in token usage, leading to increased costs. We find that the reasoning process of current LLMs is unnecessarily lengthy and it can be compressed by including a reasonable token budget in the prompt, but the choice of token budget plays a crucial role in the actual compression effectiveness. We then propose a token-budget-aware LLM reasoning framework, which dynamically estimates token budgets for different problems based on reasoning complexity and uses the estimated token budgets to guide the reasoning process. Experiments show that our method effectively reduces token costs in CoT reasoning with only a slight performance reduction, offering a practical solution to balance efficiency and accuracy in LLM reasoning. Code: this https URL. 

**Abstract (ZH)**: 推理对于大型语言模型（LLMs）在广泛任务中的卓越表现至关重要。虽然像逐步推理（Chain-of-Thought, CoT）这样的方法通过将问题分解为中间步骤来增强LLM的性能，但这也带来了显著的令牌使用量 overhead，从而增加了成本。我们发现当前LLMs的推理过程过长，可以通过在提示中包含合理的令牌预算来压缩推理过程，但令牌预算的选择在实际压缩效果中起着关键作用。随后，我们提出了一种意识令牌预算的LLM推理框架，该框架根据推理复杂度动态估计不同问题的令牌预算，并使用估算的令牌预算来引导推理过程。实验结果显示，我们的方法在CoT推理中有效降低了令牌成本，仅轻微降低了性能，提供了一种平衡效率和准确性的实用解决方案。代码：[这里提供链接] 

---
# Harnessing Large Language Models for Knowledge Graph Question Answering via Adaptive Multi-Aspect Retrieval-Augmentation 

**Title (ZH)**: 利用自适应多方面检索增强技术解锁大规模语言模型在知识图谱问答中的潜力 

**Authors**: Derong Xu Xinhang Li, Ziheng Zhang, Zhenxi Lin, Zhihong Zhu, Zhi Zheng, Xian Wu, Xiangyu Zhao, Tong Xu, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.18537)  

**Abstract**: Large Language Models (LLMs) demonstrate remarkable capabilities, yet struggle with hallucination and outdated knowledge when tasked with complex knowledge reasoning, resulting in factually incorrect outputs. Previous studies have attempted to mitigate it by retrieving factual knowledge from large-scale knowledge graphs (KGs) to assist LLMs in logical reasoning and prediction of answers. However, this kind of approach often introduces noise and irrelevant data, especially in situations with extensive context from multiple knowledge aspects. In this way, LLM attention can be potentially mislead from question and relevant information. In our study, we introduce an Adaptive Multi-Aspect Retrieval-augmented over KGs (Amar) framework. This method retrieves knowledge including entities, relations, and subgraphs, and converts each piece of retrieved text into prompt embeddings. The Amar framework comprises two key sub-components: 1) a self-alignment module that aligns commonalities among entities, relations, and subgraphs to enhance retrieved text, thereby reducing noise interference; 2) a relevance gating module that employs a soft gate to learn the relevance score between question and multi-aspect retrieved data, to determine which information should be used to enhance LLMs' output, or even filtered altogether. Our method has achieved state-of-the-art performance on two common datasets, WebQSP and CWQ, showing a 1.9\% improvement in accuracy over its best competitor and a 6.6\% improvement in logical form generation over a method that directly uses retrieved text as context prompts. These results demonstrate the effectiveness of Amar in improving the reasoning of LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）展示了显著的能力，但在处理复杂知识推理任务时，它们会遇到幻觉和过时知识的问题，导致事实错误的输出。以往的研究尝试通过从大规模知识图谱（KGs）中检索事实知识来辅助LLMs的逻辑推理和答案预测，以减轻这些问题。然而，这种做法往往引入了噪声和无关的数据，尤其是在涉及多方面知识背景的广泛上下文情况下。这样可能会误导LLMs的注意力，使其偏离问题和相关信息。在本研究中，我们提出了一种自适应的多方面检索增强知识图谱框架（Amar）。该方法检索包括实体、关系和子图的知识，并将每一段检索到的文字转换成提示嵌入。Amar框架包含两个关键子模块：1) 自对齐模块，该模块通过对齐实体、关系和子图之间的共同点来增强检索文本，从而减少噪声干扰；2) 相关性闸门模块，该模块使用软闸门来学习问题与多方面检索数据的相关性评分，从而决定哪些信息应该用于增强LLMs的输出，甚至可以过滤掉。我们的方法在两个常用数据集WebQSP和CWQ上达到了最先进的性能，通过与最佳竞争对手相比，在准确率上提高了1.9%，在逻辑形式生成上提高了6.6%。这些结果证明了Amar在提高LLMs推理能力方面的有效性。 

---
# Think or Remember? Detecting and Directing LLMs Towards Memorization or Generalization 

**Title (ZH)**: 思考还是记忆？检测并引导大语言模型向记忆或泛化方向发展 

**Authors**: Yi-Fu Fu, Yu-Chieh Tu, Tzu-Ling Cheng, Cheng-Yu Lin, Yi-Ting Yang, Heng-Yi Liu, Keng-Te Liao, Da-Cheng Juan, Shou-De Lin  

**Link**: [PDF](https://arxiv.org/pdf/2412.18497)  

**Abstract**: In this paper, we explore the foundational mechanisms of memorization and generalization in Large Language Models (LLMs), inspired by the functional specialization observed in the human brain. Our investigation serves as a case study leveraging specially designed datasets and experimental-scale LLMs to lay the groundwork for understanding these behaviors. Specifically, we aim to first enable LLMs to exhibit both memorization and generalization by training with the designed dataset, then (a) examine whether LLMs exhibit neuron-level spatial differentiation for memorization and generalization, (b) predict these behaviors using model internal representations, and (c) steer the behaviors through inference-time interventions. Our findings reveal that neuron-wise differentiation of memorization and generalization is observable in LLMs, and targeted interventions can successfully direct their behavior. 

**Abstract (ZH)**: 在本文中，我们探讨了大型语言模型（LLMs）中记忆和泛化的基本机制，受到了人类大脑功能专业化观察的启发。我们的研究通过利用特别设计的数据集和实验规模的LLM作为案例研究，为理解这些行为奠定了基础。具体而言，我们旨在通过使用设计的数据集对LLM进行训练，首先让LLM展现出记忆和泛化的双重特性，然后（a）考察LLM是否在神经元级别上表现出记忆和泛化的空间分化，（b）通过模型内部表示预测这些行为，并（c）通过推理时的干预引导这些行为。我们的发现表明，神经元级别的记忆和泛化分化在LLM中是可以观察到的，并且有针对性的干预可以成功地指导其行为。 

---
# Generating event descriptions under syntactic and semantic constraints 

**Title (ZH)**: 在语法和语义约束下的事件描述生成 

**Authors**: Angela Cao, Faye Holt, Jonas Chan, Stephanie Richter, Lelia Glass, Aaron Steven White  

**Link**: [PDF](https://arxiv.org/pdf/2412.18496)  

**Abstract**: With the goal of supporting scalable lexical semantic annotation, analysis, and theorizing, we conduct a comprehensive evaluation of different methods for generating event descriptions under both syntactic constraints -- e.g. desired clause structure -- and semantic constraints -- e.g. desired verb sense. We compare three different methods -- (i) manual generation by experts; (ii) sampling from a corpus annotated for syntactic and semantic information; and (iii) sampling from a language model (LM) conditioned on syntactic and semantic information -- along three dimensions of the generated event descriptions: (a) naturalness, (b) typicality, and (c) distinctiveness. We find that all methods reliably produce natural, typical, and distinctive event descriptions, but that manual generation continues to produce event descriptions that are more natural, typical, and distinctive than the automated generation methods. We conclude that the automated methods we consider produce event descriptions of sufficient quality for use in downstream annotation and analysis insofar as the methods used for this annotation and analysis are robust to a small amount of degradation in the resulting event descriptions. 

**Abstract (ZH)**: 为了支持可扩展的词汇语义标注、分析与理论化，我们对在语法约束（例如，所需的从句结构）和语义约束（例如，所需的动词义项）下生成事件描述的不同方法进行了全面评估。我们比较了三种不同的方法：（i）专家的手工生成；（ii）从标注了语法和语义信息的语料库中抽样；以及（iii）从基于语法和语义信息的語言模型（LM）中抽样。我们在生成事件描述的三个维度进行比较：（a）自然度；（b）典型性；（c）独特性。研究发现，所有方法都能够生成自然、典型和独特事件描述，但手工生成的描述在自然度、典型性和独特性方面仍然优于自动化生成的方法。我们得出结论，所考虑的自动化方法生成的事件描述在下游标注和分析过程中品质足够，前提是用于此标注和分析的方法能够容忍生成事件描述的小幅降级。 

---
# How "Real" is Your Real-Time Simultaneous Speech-to-Text Translation System? 

**Title (ZH)**: 您的实时同步语音转文字翻译系统有多“真实”？ 

**Authors**: Sara Papi, Peter Polak, Ondřej Bojar, Dominik Macháček  

**Link**: [PDF](https://arxiv.org/pdf/2412.18495)  

**Abstract**: Simultaneous speech-to-text translation (SimulST) translates source-language speech into target-language text concurrently with the speaker's speech, ensuring low latency for better user comprehension. Despite its intended application to unbounded speech, most research has focused on human pre-segmented speech, simplifying the task and overlooking significant challenges. This narrow focus, coupled with widespread terminological inconsistencies, is limiting the applicability of research outcomes to real-world applications, ultimately hindering progress in the field. Our extensive literature review of 110 papers not only reveals these critical issues in current research but also serves as the foundation for our key contributions. We 1) define the steps and core components of a SimulST system, proposing a standardized terminology and taxonomy; 2) conduct a thorough analysis of community trends, and 3) offer concrete recommendations and future directions to bridge the gaps in existing literature, from evaluation frameworks to system architectures, for advancing the field towards more realistic and effective SimulST solutions. 

**Abstract (ZH)**: 同步语音翻译（SimulST）在演讲者说话的同时将源语言语音实时转换为目标语言文本，确保低延迟，以提高用户的理解能力。尽管SimulST旨在应用于无限制语音，但大多数研究侧重于人工预分段的语音，简化了任务并忽略了重大挑战。这种狭窄的焦点以及术语使用的一致性问题限制了研究成果在实际应用中的适用性，最终阻碍了该领域的进步。我们对110篇论文的广泛文献综述不仅揭示了当前研究中的关键问题，还构成了我们主要贡献的基础。我们1）定义了SimulST系统的步骤和核心组件，提出了一套标准化的术语和分类体系；2）对社区趋势进行了详尽分析，并3）提供了具体的建议和未来方向，以填补现有文献中的空白，包括从评估框架到系统架构，从而推动该领域向更现实有效的SimulST解决方案前进。 

---
# Segment-Based Attention Masking for GPTs 

**Title (ZH)**: 基于段落的注意力掩蔽机制用于GPT模型 

**Authors**: Shahar Katz, Liran Ringel, Yaniv Romano, Lior Wolf  

**Link**: [PDF](https://arxiv.org/pdf/2412.18487)  

**Abstract**: Modern Language Models (LMs) owe much of their success to masked causal attention, the backbone of Generative Pre-Trained Transformer (GPT) models. Although GPTs can process the entire user prompt at once, the causal masking is applied to all input tokens step-by-step, mimicking the generation process. This imposes an unnecessary constraint during the initial "prefill" phase when the model processes the input prompt and generates the internal representations before producing any output tokens. In this work, attention is masked based on the known block structure at the prefill phase, followed by the conventional token-by-token autoregressive process after that. For example, in a typical chat prompt, the system prompt is treated as one block, and the user prompt as the next one. Each of these is treated as a unit for the purpose of masking, such that the first tokens in each block can access the subsequent tokens in a non-causal manner. Then, the model answer is generated in the conventional causal manner. This Segment-by-Segment scheme entails no additional computational overhead. When integrating it into models such as Llama and Qwen, state-of-the-art performance is consistently achieved. 

**Abstract (ZH)**: 现代语言模型（LMs）的成功很大程度上归功于遮蔽因果注意力机制，这是生成预训练变压器（GPT）模型的核心组成部分。尽管GPT可以一次性处理整个用户提示，但在生成过程中，因果遮蔽会逐步应用于所有输入令牌，即在模型处理输入提示并生成内部表示之前，对所有输入令牌进行逐步遮蔽。这在初始的“预填充”阶段施加了不必要的约束，因为在该阶段模型生成任何输出令牌之前需要处理输入提示。在本文中，遮蔽基于预填充阶段已知的块结构进行，并在此之后常规地按令牌顺序进行自回归处理。例如，在典型的聊天提示中，系统提示被视为一个块，用户提示作为下一个块。这些块中的每一个在遮蔽过程中都被视为一个单元，使得每个块中的第一个令牌可以以非因果的方式访问后续的令牌。然后，模型以常规的因果方式生成答案。这种段落级方案不增加额外的计算开销。将其集成到如Llama和Qwen等模型中时，可以实现最先进的性能。 

---
# Is Large Language Model Good at Triple Set Prediction? An Empirical Study 

**Title (ZH)**: 大型语言模型在三元组集预测方面表现如何？一项实证研究 

**Authors**: Yuan Yuan, Yajing Xu, Wen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.18443)  

**Abstract**: The core of the Knowledge Graph Completion (KGC) task is to predict and complete the missing relations or nodes in a KG. Common KGC tasks are mostly about inferring unknown elements with one or two elements being known in a triple. In comparison, the Triple Set Prediction (TSP) task is a more realistic knowledge graph completion task. It aims to predict all elements of unknown triples based on the information from known triples. In recent years, large language models (LLMs) have exhibited significant advancements in language comprehension, demonstrating considerable potential for KGC tasks. However, the potential of LLM on the TSP task has not yet to be investigated. Thus in this paper we proposed a new framework to explore the strengths and limitations of LLM in the TSP task. Specifically, the framework consists of LLM-based rule mining and LLM-based triple set prediction. The relation list of KG embedded within rich semantic information is first leveraged to prompt LLM in the generation of rules. This process is both efficient and independent of statistical information, making it easier to mine effective and realistic rules. For each subgraph, the specified rule is applied in conjunction with the relevant triples within that subgraph to guide the LLM in predicting the missing triples. Subsequently, the predictions from all subgraphs are consolidated to derive the complete set of predicted triples on KG. Finally, the method is evaluated on the relatively complete CFamily dataset. The experimental results indicate that when LLMs are required to adhere to a large amount of factual knowledge to predict missing triples, significant hallucinations occurs, leading to a noticeable decline in performance. To further explore the causes of this phenomenon, this paper presents a comprehensive analysis supported by a detailed case study. 

**Abstract (ZH)**: 知识图谱补全（Knowledge Graph Completion, KGC）任务的核心在于预测和补全知识图谱中缺失的关系或节点。常见的KGC任务通常是在已知一个或两个元素的情况下，推断三元组中的未知元素。相比之下，三元组集预测（Triple Set Prediction, TSP）任务是一种更加现实的KGC任务，它旨在基于已知三元组的信息预测所有未知三元组的元素。近年来，大型语言模型（Large Language Models, LLMs）在语言理解方面取得了显著进步，展现了在KGC任务中的巨大潜力。然而，LLM在TSP任务中的潜力尚未得到充分研究。因此，在本文中，我们提出了一种新的框架，以探索LLM在TSP任务中的优点和局限性。具体而言，该框架由基于LLM的规则挖掘和基于LLM的三元组集预测两部分组成。首先，嵌入丰富语义信息的知识图谱关系列表被利用来提示LLM生成规则，这一过程既高效又独立于统计信息，使得有效且真实的规则易于挖掘。对于每个子图，特定的规则与该子图相关的三元组结合使用，以指导LLM预测缺失的三元组。随后，从所有子图的预测中整合出完整的预测三元组集。最后，该方法在相对完整的CFamily数据集上进行了评估。实验结果表明，当LLM需要依据大量事实知识预测缺失的三元组时，会出现显著的虚构现象，导致性能明显下降。为进一步探讨这一现象的原因，本文通过详细的案例研究进行了一项全面的分析。 

---
# Unlocking the Potential of Multiple BERT Models for Bangla Question Answering in NCTB Textbooks 

**Title (ZH)**: 解锁多BERT模型在NCTB教材中 Bangla 问答任务的潜在能力 

**Authors**: Abdullah Khondoker, Enam Ahmed Taufik, Md Iftekhar Islam Tashik, S M Ishtiak mahmud, Antara Firoz Parsa  

**Link**: [PDF](https://arxiv.org/pdf/2412.18440)  

**Abstract**: Evaluating text comprehension in educational settings is critical for understanding student performance and improving curricular effectiveness. This study investigates the capability of state-of-the-art language models-RoBERTa Base, Bangla-BERT, and BERT Base-in automatically assessing Bangla passage-based question-answering from the National Curriculum and Textbook Board (NCTB) textbooks for classes 6-10. A dataset of approximately 3,000 Bangla passage-based question-answering instances was compiled, and the models were evaluated using F1 Score and Exact Match (EM) metrics across various hyperparameter configurations. Our findings revealed that Bangla-BERT consistently outperformed the other models, achieving the highest F1 (0.75) and EM (0.53) scores, particularly with smaller batch sizes, the inclusion of stop words, and a moderate learning rate. In contrast, RoBERTa Base demonstrated the weakest performance, with the lowest F1 (0.19) and EM (0.27) scores under certain configurations. The results underscore the importance of fine-tuning hyperparameters for optimizing model performance and highlight the potential of machine learning models in evaluating text comprehension in educational contexts. However, limitations such as dataset size, spelling inconsistencies, and computational constraints emphasize the need for further research to enhance the robustness and applicability of these models. This study lays the groundwork for the future development of automated evaluation systems in educational institutions, providing critical insights into model performance in the context of Bangla text comprehension. 

**Abstract (ZH)**: 在教育环境中评估文本理解能力对于理解学生表现并提高课程效果至关重要。本研究探讨了当前最先进的语言模型——RoBERTa Base、Bangla-BERT 和 BERT Base 在自动评估来自全国课程和教材委员会（NCTB）6-10 年级基于段落的问题回答的能力。我们构建了一个包含约 3,000 个基于段落的问题回答实例的数据集，并通过不同超参数配置下的 F1 分数和精确匹配（EM）指标评估了这些模型。研究结果表明，Bangla-BERT 在各个超参数配置下始终表现最好，获得了最高的 F1 得分（0.75）和 EM 得分（0.53），特别是在较小的批次大小、包含停用词和适度的学习率时。相比之下，RoBERTa Base 在某些配置下表现出最弱的性能，其 F1 得分（0.19）和 EM 得分（0.27）最低。这些结果强调了调整超参数以优化模型性能的重要性，并突显了机器学习模型在教育环境中评估文本理解的潜力。然而，数据集规模、拼写不一致以及计算限制等局限性强调了进一步研究的必要性，以增强这些模型的稳健性和适用性。本研究为未来教育机构中自动评估系统的开发奠定了基础，提供了关于这些模型在孟加拉语文本理解背景下表现的宝贵见解。 

---
# GeAR: Graph-enhanced Agent for Retrieval-augmented Generation 

**Title (ZH)**: GeAR：图增强代理用于检索增强生成 

**Authors**: Zhili Shen, Chenxin Diao, Pavlos Vougiouklis, Pascual Merita, Shriram Piramanayagam, Damien Graux, Dandan Tu, Zeren Jiang, Ruofei Lai, Yang Ren, Jeff Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2412.18431)  

**Abstract**: Retrieval-augmented generation systems rely on effective document retrieval capabilities. By design, conventional sparse or dense retrievers face challenges in multi-hop retrieval scenarios. In this paper, we present GeAR, which advances RAG performance through two key innovations: (i) graph expansion, which enhances any conventional base retriever, such as BM25, and (ii) an agent framework that incorporates graph expansion. Our evaluation demonstrates GeAR's superior retrieval performance on three multi-hop question answering datasets. Additionally, our system achieves state-of-the-art results with improvements exceeding 10% on the challenging MuSiQue dataset, while requiring fewer tokens and iterations compared to other multi-step retrieval systems. 

**Abstract (ZH)**: 检索增强生成系统依赖于有效的文档检索能力。传统稀疏检索器或稠密检索器在多跳检索场景中设计上面临挑战。本文提出了一种名为GeAR的新系统，通过两项关键创新来提升RAG（Retrieval-Augmented Generation）的表现：(i) 图扩展，它可以增强任何传统的基础检索器，如BM25；(ii) 代理框架，该框架整合了图扩展。我们的评估结果表明，GeAR在三个多跳问答数据集上显示出优越的检索性能。此外，与其它多步检索系统相比，我们的系统在具有挑战性的MuSiQue数据集上取得了最先进的结果，且所需token数量和迭代次数更少，改善幅度超过10%。 

---
# Multilingual Mathematical Reasoning: Advancing Open-Source LLMs in Hindi and English 

**Title (ZH)**: 多语言数学推理：推进印地语和英语开源大语言模型的发展 

**Authors**: Avinash Anand, Kritarth Prasad, Chhavi Kirtani, Ashwin R Nair, Manvendra Kumar Nema, Raj Jaiswal, Rajiv Ratn Shah  

**Link**: [PDF](https://arxiv.org/pdf/2412.18415)  

**Abstract**: Large Language Models (LLMs) excel in linguistic tasks but struggle with mathematical reasoning, particularly in non English languages like Hindi. This research aims to enhance the mathematical reasoning skills of smaller, resource efficient open-source LLMs in both Hindi and English. We evaluate models like OpenHathi 7B, LLaMA-2 7B, WizardMath 7B, Mistral 7B, LLeMMa 7B, MAmmoTH 7B, Gemini Pro, and GPT-4 using zero-shot, few-shot chain-of-thought (CoT) methods, and supervised fine-tuning. Our approach incorporates curriculum learning, progressively training models on increasingly difficult problems, a novel Decomposition Strategy to simplify complex arithmetic operations, and a Structured Solution Design that divides solutions into phases. Our experiments result in notable performance enhancements. WizardMath 7B exceeds Gemini's accuracy on English datasets by +6% and matches Gemini's performance on Hindi datasets. Adopting a bilingual approach that combines English and Hindi samples achieves results comparable to individual language models, demonstrating the capability to learn mathematical reasoning in both languages. This research highlights the potential for improving mathematical reasoning in open-source LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）在语言任务上表现出色，但在数学推理方面存在困难，尤其是在非英语语言如印地语中。本研究旨在增强较小的、资源高效的开源LLMs在印地语和英语中的数学推理能力。我们使用零样本、少量样本的思考链（CoT）方法和监督微调来评估OpenHathi 7B、LLaMA-2 7B、WizardMath 7B、Mistral 7B、LLeMMa 7B、MAmmoTH 7B、Gemini Pro和GPT-4等模型。我们的方法结合了逐渐学习，通过逐步训练模型解决越来越难的问题，提出了一种新颖的分解策略来简化复杂的算术运算，并采用结构化解决方案设计将解决方案划分为多个阶段。实验结果显示显著的性能提升。WizardMath 7B在英语数据集上的准确率比Gemini高出6%，在印地语数据集上与其性能相当。采用双语方法结合英语和印地语样本，其结果与单一语言模型相当，表明模型可以在两种语言中学习数学推理的能力。本研究突显了提高开源LLMs的数学推理能力的潜力。 

---
# ChaI-TeA: A Benchmark for Evaluating Autocompletion of Interactions with LLM-based Chatbots 

**Title (ZH)**: ChaI-TeA：基于大语言模型的聊天机器人互动自动补全评估基准 

**Authors**: Shani Goren, Oren Kalinsky, Tomer Stav, Yuri Rapoport, Yaron Fairstein, Ram Yazdy, Nachshon Cohen, Alexander Libov, Guy Kushilevitz  

**Link**: [PDF](https://arxiv.org/pdf/2412.18377)  

**Abstract**: The rise of LLMs has deflected a growing portion of human-computer interactions towards LLM-based chatbots. The remarkable abilities of these models allow users to interact using long, diverse natural language text covering a wide range of topics and styles. Phrasing these messages is a time and effort consuming task, calling for an autocomplete solution to assist users. We introduce the task of chatbot interaction autocomplete. We present ChaI-TeA: CHat InTEraction Autocomplete; An autcomplete evaluation framework for LLM-based chatbot interactions. The framework includes a formal definition of the task, coupled with suitable datasets and metrics. We use the framework to evaluate After formally defining the task along with suitable datasets and metrics, we test 9 models on the defined auto completion task, finding that while current off-the-shelf models perform fairly, there is still much room for improvement, mainly in ranking of the generated suggestions. We provide insights for practitioners working on this task and open new research directions for researchers in the field. We release our framework to serve as a foundation for future research. 

**Abstract (ZH)**: 大规模语言模型（LLM）的兴起正逐渐将越来越多的人机交互转向基于LLM的聊天机器人。这些模型的强大能力使用户能够使用长篇且多样化的自然语言文本，涵盖广泛的主题和风格进行交互。撰写这些消息是一个既耗费时间又费力的任务，因此需要一种自动补全解决方案来辅助用户。我们提出了聊天机器人交互自动补全任务。我们提出ChaI-TeA：聊天互动自动补全；一种用于基于LLM的聊天机器人交互的自动补全评估框架。该框架包括对该任务的正式定义，以及配套的数据集和评估指标。我们使用该框架来评估任务，并测试了9个模型在定义的自动补全任务上的表现，发现尽管现有的现成模型表现尚可，但在生成的建议排序上仍有很大的改进空间。我们为从事该任务的实践者提供了见解，并为该领域的研究人员开辟了新的研究方向。我们发布该框架，作为未来研究的基础。 

---
# Bidirectional Topic Matching: Quantifying Thematic Overlap Between Corpora Through Topic Modelling 

**Title (ZH)**: 双向主题匹配：通过主题建模量化语料库间的主题重叠程度 

**Authors**: Raven Adam, Marie Lisa Kogler  

**Link**: [PDF](https://arxiv.org/pdf/2412.18376)  

**Abstract**: This study introduces Bidirectional Topic Matching (BTM), a novel method for cross-corpus topic modeling that quantifies thematic overlap and divergence between corpora. BTM is a flexible framework that can incorporate various topic modeling approaches, including BERTopic, Top2Vec, and Latent Dirichlet Allocation (LDA). BTM employs a dual-model approach, training separate topic models for each corpus and applying them reciprocally to enable comprehensive cross-corpus comparisons. This methodology facilitates the identification of shared themes and unique topics, providing nuanced insights into thematic relationships. Validation against cosine similarity-based methods demonstrates the robustness of BTM, with strong agreement metrics and distinct advantages in handling outlier topics. A case study on climate news articles showcases BTM's utility, revealing significant thematic overlaps and distinctions between corpora focused on climate change and climate action. BTM's flexibility and precision make it a valuable tool for diverse applications, from political discourse analysis to interdisciplinary studies. By integrating shared and unique topic analyses, BTM offers a comprehensive framework for exploring thematic relationships, with potential extensions to multilingual and dynamic datasets. This work highlights BTM's methodological contributions and its capacity to advance discourse analysis across various domains. 

**Abstract (ZH)**: 这项研究引入了一种名为双向主题匹配（BTM）的新方法，这是一种用于跨语料库主题建模的技术，能够量化不同语料库之间的主题重叠和分歧。BTM 是一个灵活的框架，可以结合包括 BERTopic、Top2Vec 和潜在狄利克雷分配（LDA）在内的各种主题建模方法。BTM 采用双模型方法，为每个语料库训练独立的主题模型，并相互应用以实现全面的跨语料库比较。该方法有助于识别共享主题和独特的主题，提供关于主题关系的深入洞见。与基于余弦相似度的方法进行验证，显示了 BTM 的稳健性，具有强烈的共识指标和处理异常主题的独特优势。在气候新闻文章案例研究中，展示了 BTM 的应用价值，揭示了气候变化和气候行动集中讨论的语料库之间的重要主题重叠和区分。BTM 的灵活性和精确度使其成为政治话语分析和跨学科研究等多种应用的重要工具。通过结合共享和独特主题的分析，BTM 为探索主题关系提供了一个全面的框架，具有扩展至多语言和动态数据集的潜力。本研究突显了 BTM 的方法论贡献及其在各领域提升话语分析的能力。 

---
# Towards Global AI Inclusivity: A Large-Scale Multilingual Terminology Dataset 

**Title (ZH)**: 迈向全球人工智能包容性：大规模多语种术语数据集 

**Authors**: Jiarui Liu, Iman Ouzzani, Wenkai Li, Lechen Zhang, Tianyue Ou, Houda Bouamor, Zhijing Jin, Mona Diab  

**Link**: [PDF](https://arxiv.org/pdf/2412.18367)  

**Abstract**: The field of machine translation has achieved significant advancements, yet domain-specific terminology translation, particularly in AI, remains challenging. We introduced GIST, a large-scale multilingual AI terminology dataset containing 5K terms extracted from top AI conference papers spanning 2000 to 2023. The terms were translated into Arabic, Chinese, French, Japanese, and Russian using a hybrid framework that combines LLMs for extraction with human expertise for translation. The dataset's quality was benchmarked against existing resources, demonstrating superior translation accuracy through crowdsourced evaluation. GIST was integrated into translation workflows using post-translation refinement methods that required no retraining, where LLM prompting consistently improved BLEU and COMET scores. A web demonstration on the ACL Anthology platform highlights its practical application, showcasing improved accessibility for non-English speakers. This work aims to address critical gaps in AI terminology resources and fosters global inclusivity and collaboration in AI research. 

**Abstract (ZH)**: 机器翻译领域已经取得了显著的进步，但在特定领域的术语翻译，特别是在人工智能领域，依然存在挑战。我们引入了一个名为GIST的大规模多语言AI术语数据集，包含从2000年至2023年顶级人工智能会议论文中提取的5000个术语，并将其翻译成阿拉伯语、中文、法语、日语和俄语。翻译采用了结合大型语言模型（LLM）进行提取和人工专家进行翻译的混合框架。该数据集的质量通过众包评价与现有资源进行了基准测试，结果显示其翻译准确性更高。GIST数据集被集成到翻译工作流程中，通过不需要重新训练的后翻译润色方法应用，LLM的提示方法持续提高了BLEU和COMET评分。在ACL Anthology平台上的网络演示展示了其实际应用，突显了对非英语使用者的改进易用性。本项工作旨在填补人工智能术语资源中的关键空白，并促进全球在人工智能研究中的包容性和合作。 

---
# Extracting triples from dialogues for conversational social agents 

**Title (ZH)**: 将对话中的三元组提取用于会话社会代理 

**Authors**: Piek Vossen, Selene Báez Santamaría, Lenka Bajčetić, Thomas Belluci  

**Link**: [PDF](https://arxiv.org/pdf/2412.18364)  

**Abstract**: Obtaining an explicit understanding of communication within a Hybrid Intelligence collaboration is essential to create controllable and transparent agents. In this paper, we describe a number of Natural Language Understanding models that extract explicit symbolic triples from social conversation. Triple extraction has mostly been developed and tested for Knowledge Base Completion using Wikipedia text and data for training and testing. However, social conversation is very different as a genre in which interlocutors exchange information in sequences of utterances that involve statements, questions, and answers. Phenomena such as co-reference, ellipsis, coordination, and implicit and explicit negation or confirmation are more prominent in conversation than in Wikipedia text. We therefore describe an attempt to fill this gap by releasing data sets for training and testing triple extraction from social conversation. We also created five triple extraction models and tested them in our evaluation data. The highest precision is 51.14 for complete triples and 69.32 for triple elements when tested on single utterances. However, scores for conversational triples that span multiple turns are much lower, showing that extracting knowledge from true conversational data is much more challenging. 

**Abstract (ZH)**: 了解混合智能协作中通信的显式理解对于创建可控和透明的智能体至关重要。在本文中，我们描述了一些自然语言理解模型，这些模型可以从社会对话中提取显式的符号三元组。三元组提取主要在知识库完成任务中进行开发和测试，通常使用维基百科的文本和数据进行训练和测试。然而，社会对话作为一种体裁存在巨大差异，在这种对话中，对话参与者通过一系列包括陈述、提问和回答在内的话语交换信息。在对话中，共指、省略、并列、以及隐式和显式的否定或确认现象更为明显。因此，我们尝试填补这一空白，通过发布社会对话数据集来训练和测试三元组提取。我们还创建了五个三元组提取模型，并在我们的评估数据集上进行了测试。在单独话语上测试时，完整三元组的最高精度为51.14%，三元组元素的最高精度为69.32%。然而，跨多个回合的对话三元组的得分明显较低，这表明从真实对话数据中提取知识更具挑战性。 

---
# Multi-Agents Based on Large Language Models for Knowledge-based Visual Question Answering 

**Title (ZH)**: 基于大型语言模型的多agents知识驱动视觉问答系统 

**Authors**: Zhongjian Hu, Peng Yang, Bing Li, Zhenqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.18351)  

**Abstract**: Large Language Models (LLMs) have achieved impressive results in knowledge-based Visual Question Answering (VQA). However existing methods still have challenges: the inability to use external tools autonomously, and the inability to work in teams. Humans tend to know whether they need to use external tools when they encounter a new question, e.g., they tend to be able to give a direct answer to a familiar question, whereas they tend to use tools such as search engines when they encounter an unfamiliar question. In addition, humans also tend to collaborate and discuss with others to get better answers. Inspired by this, we propose the multi-agent voting framework. We design three LLM-based agents that simulate different levels of staff in a team, and assign the available tools according to the levels. Each agent provides the corresponding answer, and finally all the answers provided by the agents are voted to get the final answer. Experiments on OK-VQA and A-OKVQA show that our approach outperforms other baselines by 2.2 and 1.0, respectively. 

**Abstract (ZH)**: 大型语言模型（LLMs）已经在基于知识的视觉问答（VQA）任务上取得了显著成果。然而，现有的方法仍然存在一些挑战：自主使用外部工具的能力不足，以及无法团队协作工作。人类在遇到新问题时通常能判断是否需要使用外部工具，例如，他们往往能直接回答熟悉的问题，而遇到不熟悉的问题时会使用诸如搜索引擎等工具。此外，人类还倾向于与他人协作讨论以获得更好的答案。受此启发，我们提出了多Agent投票框架。我们设计了三个基于LLM的Agent，模拟团队中不同级别的工作人员，并根据其级别分配可用的工具。每个Agent提供相应的答案，最后通过投票将所有Agent提供的答案结合以获得最终答案。实验结果表明，我们的方法在OK-VQA和A-OKVQA数据集上分别优于其他基线方法2.2和1.0的分数。 

---
# M-Ped: Multi-Prompt Ensemble Decoding for Large Language Models 

**Title (ZH)**: M-Ped：面向大规模语言模型的多提示集成解码 

**Authors**: Jiaxin Guo, Daimeng Wei, Yuanchang Luo, Shimin Tao, Hengchao Shang, Zongyao Li, Shaojun Li, Jinlong Yang, Zhanglin Wu, Zhiqiang Rao, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2412.18299)  

**Abstract**: With the widespread application of Large Language Models (LLMs) in the field of Natural Language Processing (NLP), enhancing their performance has become a research hotspot. This paper presents a novel multi-prompt ensemble decoding approach designed to bolster the generation quality of LLMs by leveraging the aggregation of outcomes from multiple prompts. Given a unique input $X$, we submit $n$ variations of prompts with $X$ to LLMs in batch mode to decode and derive probability distributions. For each token prediction, we calculate the ensemble probability by averaging the $n$ probability distributions within the batch, utilizing this aggregated probability to generate the token. This technique is dubbed Inner-Batch Ensemble. To facilitate efficient batch inference, we implement a Left-Padding strategy to maintain uniform input lengths across the n prompts. Through extensive experimentation on diverse NLP tasks, including machine translation, code generation, and text simplification, we demonstrate the efficacy of our method in enhancing LLM performance. The results show substantial improvements in BLEU scores, pass@$k$ rates, and LENS metrics over conventional methods. 

**Abstract (ZH)**: 随着大规模语言模型（LLMs）在自然语言处理（NLP）领域的广泛应用，提高其性能已成为研究的热点。本文提出了一种新颖的多提示集ensemble解码方法，该方法通过聚合多个提示的输出结果，以增强LLMs的生成质量。对于给定的唯一输入$X$，我们将$X$的$n$个不同版本的提示提交给LLMs进行批量解码，并推导出概率分布。对于每个令牌预测，我们通过计算批量内的$n$个概率分布的平均值来计算集成概率，并利用该集成概率生成令牌。该方法称为内部批量ensemble。为了促进高效的批量推理，我们采用左填充策略来保持$n$个提示之间输入长度的一致性。通过在机器翻译、代码生成和文本简化等多种NLP任务上的广泛实验，我们展示了该方法在提高LLMs性能方面的有效性。结果表明，该方法在BLEU分数、pass@$k$率和LENS度量方面比传统方法有显著提高。 

---
# GenAI Content Detection Task 2: AI vs. Human -- Academic Essay Authenticity Challenge 

**Title (ZH)**: GenAI内容检测任务2：AI vs.人类——学术论文真实性挑战 

**Authors**: Shammur Absar Chowdhury, Hind Almerekhi, Mucahid Kutlu, Kaan Efe Keles, Fatema Ahmad, Tasnim Mohiuddin, George Mikros, Firoj Alam  

**Link**: [PDF](https://arxiv.org/pdf/2412.18274)  

**Abstract**: This paper presents a comprehensive overview of the first edition of the Academic Essay Authenticity Challenge, organized as part of the GenAI Content Detection shared tasks collocated with COLING 2025. This challenge focuses on detecting machine-generated vs. human-authored essays for academic purposes. The task is defined as follows: "Given an essay, identify whether it is generated by a machine or authored by a human.'' The challenge involves two languages: English and Arabic. During the evaluation phase, 25 teams submitted systems for English and 21 teams for Arabic, reflecting substantial interest in the task. Finally, seven teams submitted system description papers. The majority of submissions utilized fine-tuned transformer-based models, with one team employing Large Language Models (LLMs) such as Llama 2 and Llama 3. This paper outlines the task formulation, details the dataset construction process, and explains the evaluation framework. Additionally, we present a summary of the approaches adopted by participating teams. Nearly all submitted systems outperformed the n-gram-based baseline, with the top-performing systems achieving F1 scores exceeding 0.98 for both languages, indicating significant progress in the detection of machine-generated text. 

**Abstract (ZH)**: 本文概述了在2025年COLING会议期间协办的首个学术论文真实性挑战活动的第一版。该挑战旨在识别用于学术目的的机器生成论文和人类作者论文。任务定义如下：“给定一篇论文，判断它是机器生成的还是人类撰写的。”该挑战涉及两种语言：英语和阿拉伯语。在评估阶段，25支队伍提交了英语系统的模型，21支队伍提交了阿拉伯语系统的模型，表明该任务引发了显著的兴趣。最终，共有七支队伍提交了系统描述论文。提交的大部分系统使用了微调的转换器模型，其中一支队伍使用了大型语言模型（LLMs），如Llama 2和Llama 3。本文概述了任务的制定过程，详细介绍了数据集的构建过程，并解释了评估框架。此外，我们还总结了参赛队伍所采用的方法。近所有提交的系统都优于基于n-克隆的基线系统，最佳系统在两种语言上的F1分数均超过0.98，显示了在识别机器生成文本方面取得了显著进展。 

---
# Investigating Large Language Models for Code Vulnerability Detection: An Experimental Study 

**Title (ZH)**: 探究大型语言模型在代码漏洞检测中的应用：一项实验研究 

**Authors**: Xuefeng Jiang, Lvhua Wu, Sheng Sun, Jia Li, Jingjing Xue, Yuwei Wang, Tingting Wu, Min Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.18260)  

**Abstract**: Code vulnerability detection (CVD) is essential for addressing and preventing system security issues, playing a crucial role in ensuring software security. Previous learning-based vulnerability detection methods rely on either fine-tuning medium-size sequence models or training smaller neural networks from scratch. Recent advancements in large pre-trained language models (LLMs) have showcased remarkable capabilities in various code intelligence tasks including code understanding and generation. However, the effectiveness of LLMs in detecting code vulnerabilities is largely under-explored. This work aims to investigate the gap by fine-tuning LLMs for the CVD task, involving four widely-used open-source LLMs. We also implement other five previous graph-based or medium-size sequence models for comparison. Experiments are conducted on five commonly-used CVD datasets, including both the part of short samples and long samples. In addition, we conduct quantitative experiments to investigate the class imbalance issue and the model's performance on samples of different lengths, which are rarely studied in previous works. To better facilitate communities, we open-source all codes and resources of this study in this https URL and this https URL. 

**Abstract (ZH)**: 代码漏洞检测（CVD）对于解决和预防系统安全问题至关重要，对于确保软件安全起到重要作用。现有的基于学习的漏洞检测方法依赖于微调中等大小的序列模型或从零开始训练较小的神经网络。近年来，大规模预训练语言模型（LLMs）在代码理解和生成等多种代码智能任务中展示了显著的能力。然而，LLMs在检测代码漏洞方面的有效性尚未得到充分探索。本文旨在通过微调LLMs来填补这一空白，涉及四个广泛使用的开源LLMs。我们还实现了五个基于图的方法或中等大小的序列模型进行比较研究。实验在五个常用的CVD数据集上进行，包括短样本和长样本的部分。此外，我们还进行了定量实验，以研究类别不平衡问题以及模型在不同长度样本上的性能，这些问题在之前的研究中很少被探讨。为了更好地促进学术界的发展，我们在以下链接开源了该研究的所有代码和资源：此链接和此链接。 

---
# Robustness-aware Automatic Prompt Optimization 

**Title (ZH)**: Robustness-aware 自动提示优化 

**Authors**: Zeru Shi, Zhenting Wang, Yongye Su, Weidi Luo, Fan Yang, Yongfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.18196)  

**Abstract**: The performance of Large Language Models (LLMs) is based on the quality of the prompts and the semantic and structural integrity information of the input data. However, current prompt generation methods primarily focus on generating prompts for clean input data, often overlooking the impact of perturbed inputs on prompt performance. To address this limitation, we propose BATprompt (By Adversarial Training prompt), a novel method for prompt generation designed to withstand input perturbations (such as typos in the input). Inspired by adversarial training techniques, BATprompt demonstrates strong performance on a variety of perturbed tasks through a two-step process: adversarial perturbation and iterative optimization on unperturbed input via LLM. Unlike conventional adversarial attack methods, BATprompt avoids reliance on real gradients or model parameters. Instead, it leverages the advanced reasoning, language understanding and self reflection capabilities of LLMs to simulate gradients, guiding the generation of adversarial perturbations and optimizing prompt performance. In our experiments, we evaluate BATprompt on multiple datasets across both language understanding and generation tasks. The results indicate that BATprompt outperforms existing prompt generation methods, delivering superior robustness and performance under diverse perturbation scenarios. 

**Abstract (ZH)**: 大型语言模型（LLMs）的性能取决于提示的质量以及输入数据在语义和结构完整性方面的信息。然而，当前的提示生成方法主要集中在生成适用于干净输入数据的提示上，往往忽视了扰动输入对提示性能的影响。为了解决这一限制，我们提出了BATprompt（通过对抗训练生成提示）——一种设计用于抵御输入扰动（例如输入中的拼写错误）的新型提示生成方法。受到对抗训练技术的启发，BATprompt 通过两步过程在扰动任务上表现出强大的性能：对抗扰动和基于LLM对未受扰输入的迭代优化。与传统的对抗攻击方法不同，BATprompt 避免了对真实梯度或模型参数的依赖。相反，它利用了LLM的高级推理、语言理解和自我反思能力来模拟梯度，引导对抗扰动的生成并优化提示性能。在我们的实验中，我们在语言理解和生成任务的多个数据集上评估了BATprompt。结果表明，BATprompt 在各种扰动场景下的鲁棒性和性能均优于现有的提示生成方法。 

---
# An Analysis on Automated Metrics for Evaluating Japanese-English Chat Translation 

**Title (ZH)**: 对评估日英聊天翻译的自动化指标的分析 

**Authors**: Andre Rusli, Makoto Shishido  

**Link**: [PDF](https://arxiv.org/pdf/2412.18190)  

**Abstract**: This paper analyses how traditional baseline metrics, such as BLEU and TER, and neural-based methods, such as BERTScore and COMET, score several NMT models performance on chat translation and how these metrics perform when compared to human-annotated scores. The results show that for ranking NMT models in chat translations, all metrics seem consistent in deciding which model outperforms the others. This implies that traditional baseline metrics, which are faster and simpler to use, can still be helpful. On the other hand, when it comes to better correlation with human judgment, neural-based metrics outperform traditional metrics, with COMET achieving the highest correlation with the human-annotated score on a chat translation. However, we show that even the best metric struggles when scoring English translations from sentences with anaphoric zero-pronoun in Japanese. 

**Abstract (ZH)**: 本文分析了传统基准指标（如BLEU和TER）和基于神经网络的方法（如BERTScore和COMET）在聊天翻译中对多种神经机器翻译（NMT）模型性能的评分情况，并探讨了这些指标与人工标注分数的对比性能。结果表明，在排名聊天翻译的NMT模型时，所有指标在决定哪个模型表现更优方面似乎是一致的。这表明，尽管传统基准指标更快且更简单易用，它们仍然具有一定的帮助。另一方面，在与人工判断的相关性方面，基于神经网络的指标优于传统指标，其中COMET在聊天翻译中与人工标注分数的关联性最高。然而，我们还表明，即使最好的指标，在对包含日语零指代词的英文译文进行评分时也会表现出困难。 

---
# On the Applicability of Zero-Shot Cross-Lingual Transfer Learning for Sentiment Classification in Distant Language Pairs 

**Title (ZH)**: 零样本跨语言迁移学习在远距离语言配对情感分类中的适用性研究 

**Authors**: Andre Rusli, Makoto Shishido  

**Link**: [PDF](https://arxiv.org/pdf/2412.18188)  

**Abstract**: This research explores the applicability of cross-lingual transfer learning from English to Japanese and Indonesian using the XLM-R pre-trained model. The results are compared with several previous works, either by models using a similar zero-shot approach or a fully-supervised approach, to provide an overview of the zero-shot transfer learning approach's capability using XLM-R in comparison with existing models. Our models achieve the best result in one Japanese dataset and comparable results in other datasets in Japanese and Indonesian languages without being trained using the target language. Furthermore, the results suggest that it is possible to train a multi-lingual model, instead of one model for each language, and achieve promising results. 

**Abstract (ZH)**: 本研究探讨了使用XLM-R预训练模型从英语跨语言转移学习到日语和印度尼西亚语的适用性。我们将结果与使用类似零样本方法或完全监督方法的几种先前工作进行了比较，以提供一个关于使用XLM-R在零样本转移学习方面的能力与现有模型相比的概览。我们的模型在日语数据集中取得了最佳结果，在其他日语和印度尼西亚语数据集上也取得了可比拟的结果，而无需使用目标语言进行训练。此外，结果表明，可以训练一个多语言模型，而不是为每种语言分别训练一个模型，并且能够取得令人鼓舞的结果。 

---
# Survey of Pseudonymization, Abstractive Summarization & Spell Checker for Hindi and Marathi 

**Title (ZH)**: pseudonymization、抽象总结及拼写检查在印地语和马拉地语中的综述 

**Authors**: Rasika Ransing, Mohammed Amaan Dhamaskar, Ayush Rajpurohit, Amey Dhoke, Sanket Dalvi  

**Link**: [PDF](https://arxiv.org/pdf/2412.18163)  

**Abstract**: India's vast linguistic diversity presents unique challenges and opportunities for technological advancement, especially in the realm of Natural Language Processing (NLP). While there has been significant progress in NLP applications for widely spoken languages, the regional languages of India, such as Marathi and Hindi, remain underserved. Research in the field of NLP for Indian regional languages is at a formative stage and holds immense significance. The paper aims to build a platform which enables the user to use various features like text anonymization, abstractive text summarization and spell checking in English, Hindi and Marathi language. The aim of these tools is to serve enterprise and consumer clients who predominantly use Indian Regional Languages. 

**Abstract (ZH)**: 印度庞大的语言多样性为技术创新，尤其是自然语言处理（NLP）领域，带来了独特的机会和挑战。尽管在广泛使用的语言上已经取得了显著的NLP应用进展，但印度的区域语言，如马拉地语和 Hindi，仍然得不到充分的服务。印度区域语言的NLP研究还处于初级阶段，但具有巨大的重要意义。本文旨在构建一个平台，使用户能够使用诸如文本匿名化、抽象文本概要和拼写检查等功能，支持英语、Hindi 和马拉地语。这些工具的目的是服务于主要使用印度区域语言的企业和消费者客户。 

---
# CoAM: Corpus of All-Type Multiword Expressions 

**Title (ZH)**: CoAM：各种类型的多词表达式语料库 

**Authors**: Yusuke Ide, Joshua Tanner, Adam Nohejl, Jacob Hoffman, Justin Vasselli, Hidetaka Kamigaito, Taro Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2412.18151)  

**Abstract**: Multiword expressions (MWEs) refer to idiomatic sequences of multiple words. MWE identification, i.e., detecting MWEs in text, can play a key role in downstream tasks such as machine translation. Existing datasets for MWE identification are inconsistently annotated, limited to a single type of MWE, or limited in size. To enable reliable and comprehensive evaluation, we created CoAM: Corpus of All-Type Multiword Expressions, a dataset of 1.3K sentences constructed through a multi-step process to enhance data quality consisting of human annotation, human review, and automated consistency checking. MWEs in CoAM are tagged with MWE types, such as Noun and Verb, to enable fine-grained error analysis. Annotations for CoAM were collected using a new interface created with our interface generator, which allows easy and flexible annotation of MWEs in any form, including discontinuous ones. Through experiments using CoAM, we find that a fine-tuned large language model outperforms the current state-of-the-art approach for MWE identification. Furthermore, analysis using our MWE type tagged data reveals that Verb MWEs are easier than Noun MWEs to identify across approaches. 

**Abstract (ZH)**: 多词表达（Multiword Expressions，MWEs）是指由多个词组成的惯用词组。MWE识别，即在文本中检测MWEs，可以在机器翻译等下游任务中发挥关键作用。现有用于MWE识别的数据集在标注上不一致，仅限于一种类型的MWE，或者规模有限。为进行可靠而全面的评估，我们创建了CoAM：全类型多词表达语料库，这是一个包含1300个句子的数据集，通过多步过程构建而成，旨在提高数据质量，包括人工标注、人工审核和自动一致性检查。CoAM中的MWEs根据类型（如名词和动词）进行标记，以支持精细粒度的错误分析。标注CoAM的数据使用了我们新开发的界面生成器收集，该工具允许用户轻松灵活地标注任何形式，包括不连续的MWEs。通过使用CoAM进行的实验表明， fine-tuned 大型语言模型在MWE识别方面的表现优于当前最先进的方法。此外，使用我们标记了MWE类型的数据显示，动词MWEs比名词MWEs更容易被各种方法识别。 

---
# Ensuring Consistency for In-Image Translation 

**Title (ZH)**: 确保图像内一致性的翻译

如果是指论文标题的翻译，可以更正式地表述为：

确保图像内翻译的一致性 

**Authors**: Chengpeng Fu, Xiaocheng Feng, Yichong Huang, Wenshuai Huo, Baohang Li, Zhirui Zhang, Yunfei Lu, Dandan Tu, Duyu Tang, Hui Wang, Bing Qin, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.18139)  

**Abstract**: The in-image machine translation task involves translating text embedded within images, with the translated results presented in image format. While this task has numerous applications in various scenarios such as film poster translation and everyday scene image translation, existing methods frequently neglect the aspect of consistency throughout this process. We propose the need to uphold two types of consistency in this task: translation consistency and image generation consistency. The former entails incorporating image information during translation, while the latter involves maintaining consistency between the style of the text-image and the original image, ensuring background integrity. To address these consistency requirements, we introduce a novel two-stage framework named HCIIT (High-Consistency In-Image Translation) which involves text-image translation using a multimodal multilingual large language model in the first stage and image backfilling with a diffusion model in the second stage. Chain of thought learning is utilized in the first stage to enhance the model's ability to leverage image information during translation. Subsequently, a diffusion model trained for style-consistent text-image generation ensures uniformity in text style within images and preserves background details. A dataset comprising 400,000 style-consistent pseudo text-image pairs is curated for model training. Results obtained on both curated test sets and authentic image test sets validate the effectiveness of our framework in ensuring consistency and producing high-quality translated images. 

**Abstract (ZH)**: 图像内机器翻译任务涉及将嵌入图像中的文本进行翻译，并以图像格式呈现翻译结果。尽管该任务在电影海报翻译和日常场景图像翻译等各个场景中具有广泛的应用前景，现有的方法往往忽略了这一过程中的连贯性问题。我们提出了在这项任务中保持两种连贯性的必要性：翻译连贯性和图像生成连贯性。前者涉及在翻译过程中融合图像信息，后者则要求保持文本图像样式与原图像的一致性，确保背景完整性。为了满足这些连贯性要求，我们提出了一种新颖的两阶段框架，称为HCIIT（高连贯性图像内翻译），该框架包括第一阶段使用多模态多语言大语言模型进行文本图像翻译，第二阶段使用扩散模型进行图像填充。在第一阶段中，采用链式思维学习方法以增强模型在翻译过程中利用图像信息的能力。随后，一种为样式一致的文本图像生成训练的扩散模型确保了图像中文本样式的统一性，并保留了背景细节。为了训练模型，我们编纂了一个包含400,000个样式一致的伪文本图像对的数据集。在自编数据集和真实图像数据集上的实验结果验证了我们框架在保证连贯性和生成高质量翻译图像方面的有效性。 

---
# LSAQ: Layer-Specific Adaptive Quantization for Large Language Model Deployment 

**Title (ZH)**: LSAQ：大型语言模型部署的分层自适应量化 

**Authors**: Binrui Zeng, Bin Ji, Xiaodong Liu, Jie Yu, Shasha Li, Jun Ma, Xiaopeng Li, Shangwen Wang, Xinran Hong  

**Link**: [PDF](https://arxiv.org/pdf/2412.18135)  

**Abstract**: As large language models (LLMs) demonstrate exceptional performance across various domains, the deployment of these models on edge devices has emerged as a new trend. Quantization techniques, which reduce the size and memory footprint of LLMs, are effective for enabling deployment on resource-constrained edge devices. However, existing one-size-fits-all quantization methods often fail to dynamically adjust the memory consumption of LLMs based on specific hardware characteristics and usage scenarios. To address this limitation, we propose LSAQ (Layer-Specific Adaptive Quantization), a system for adaptive quantization and dynamic deployment of LLMs based on layer importance. LSAQ evaluates layer importance by constructing top-k token sets from the inputs and outputs of each layer and calculating their Jaccard coefficient. Using this evaluation, the system adaptively adjusts quantization strategies in real time according to the resource availability of edge devices, assigning different precision levels to layers of varying importance. This approach significantly reduces the storage requirements of LLMs while maintaining model performance, enabling efficient deployment across diverse hardware platforms and usage scenarios. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在各个领域展现出卓越的性能，这些模型在边缘设备上的部署已成为新的趋势。量化技术通过减少LLMs的大小和内存占用，有效促进了在资源受限的边缘设备上的部署。然而，现有的“一刀切”量化方法往往无法根据特定的硬件特性和使用场景动态调整LLMs的内存消耗。为解决这一局限，我们提出了一种名为LSAQ（层特定自适应量化）的系统，该系统基于层的重要性实现了适应性和动态部署LLMs。LSAQ通过从每个层的输入和输出中构建top-k令牌集并计算其交并比系数来评估层的重要性。利用这种评估，系统会实时根据边缘设备的可用资源来调整量化策略，对不同重要性的层分配不同的精度等级。这种方法在显著减少LLMs的存储需求的同时，仍能保持模型性能，从而能够高效地支持多种硬件平台和使用场景下的部署。 

---
# Do Language Models Understand the Cognitive Tasks Given to Them? Investigations with the N-Back Paradigm 

**Title (ZH)**: 语言模型是否理解分配给它们的认知任务？基于N-Back范式的探究 

**Authors**: Xiaoyang Hu, Richard L. Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2412.18120)  

**Abstract**: Cognitive tasks originally developed for humans are now increasingly used to study language models. While applying these tasks is often straightforward, interpreting their results can be challenging. In particular, when a model underperforms, it's often unclear whether this results from a limitation in the cognitive ability being tested or a failure to understand the task itself. A recent study argued that GPT 3.5's declining performance on 2-back and 3-back tasks reflects a working memory capacity limit similar to humans. By analyzing a range of open-source language models of varying performance levels on these tasks, we show that the poor performance instead reflects a limitation in task comprehension and task set maintenance. In addition, we push the best performing model to higher n values and experiment with alternative prompting strategies, before analyzing model attentions. Our larger aim is to contribute to the ongoing conversation around refining methodologies for the cognitive evaluation of language models. 

**Abstract (ZH)**: 原先设计用于人类的认知任务现在越来越多地被用于研究语言模型。虽然将这些任务应用于语言模型通常是直接的，但解读其结果可能会遇到挑战。特别是当模型表现不佳时，往往不清楚这是由于被测试的认知能力的局限性，还是因为未能理解任务本身。最近的一项研究认为，GPT-3.5在2-back和3-back任务中的表现下降反映了与其类似的人类工作记忆容量限制。通过对一系列不同性能水平的开源语言模型在这两项任务上的表现进行分析，我们发现其表现不佳实际上反映了任务理解能力和任务集维持能力的局限性。此外，我们推动最佳性能的模型达到更高的n值，并尝试不同的提示策略，在此之后分析模型的注意力机制。我们的更大目标是为语言模型的认知评估方法的精化研究贡献力量。 

---
# Molly: Making Large Language Model Agents Solve Python Problem More Logically 

**Title (ZH)**: 莫莉：使大型语言模型代理更逻辑地解决Python问题 

**Authors**: Rui Xiao, Jiong Wang, Lu Han, Na Zong, Han Wu  

**Link**: [PDF](https://arxiv.org/pdf/2412.18093)  

**Abstract**: Applying large language models (LLMs) as teaching assists has attracted much attention as an integral part of intelligent education, particularly in computing courses. To reduce the gap between the LLMs and the computer programming education expert, fine-tuning and retrieval augmented generation (RAG) are the two mainstream methods in existing researches. However, fine-tuning for specific tasks is resource-intensive and may diminish the model`s generalization capabilities. RAG can perform well on reducing the illusion of LLMs, but the generation of irrelevant factual content during reasoning can cause significant confusion for learners. To address these problems, we introduce the Molly agent, focusing on solving the proposed problem encountered by learners when learning Python programming language. Our agent automatically parse the learners' questioning intent through a scenario-based interaction, enabling precise retrieval of relevant documents from the constructed knowledge base. At generation stage, the agent reflect on the generated responses to ensure that they not only align with factual content but also effectively answer the user's queries. Extensive experimentation on a constructed Chinese Python QA dataset shows the effectiveness of the Molly agent, indicating an enhancement in its performance for providing useful responses to Python questions. 

**Abstract (ZH)**: 将以下论文内容或标题翻译成中文，并符合学术规范：

将大型语言模型（LLMs）用作教学助手，作为智能教育，尤其是在计算课程中的一项重要组成部分，已经引起了广泛关注。为减少LLMs与计算机编程教育专家之间的差距，现有的研究主要采用了微调和检索增强生成（RAG）两种方法。然而，针对特定任务的微调耗时且可能削弱模型的一般化能力。RAG 可在减少LLMs的幻觉方面表现出色，但在推理过程中生成的相关事实内容不足可能导致学习者的困惑。为解决这些问题，我们引入了Molly代理，专注于解决学习Python编程语言过程中学习者遇到的问题。我们的代理通过基于场景的交互自动解析学习者的提问意图，从而从构建的知识库中精确检索相关文档。在生成阶段，代理反思生成的回答，以确保它们不仅与事实内容一致，还能有效地回答用户的问题。通过对构建的中文Python问答数据集进行广泛的实验，展示了Molly代理的有效性，表明其性能在提供有用的Python问题回答方面有所提升。 

---
# Improving Factuality with Explicit Working Memory 

**Title (ZH)**: 提高事实准确性与显性工作记忆的关系 

**Authors**: Mingda Chen, Yang Li, Karthik Padthe, Rulin Shao, Alicia Sun, Luke Zettlemoyer, Gargi Gosh, Wen-tau Yih  

**Link**: [PDF](https://arxiv.org/pdf/2412.18069)  

**Abstract**: Large language models can generate factually inaccurate content, a problem known as hallucination. Recent works have built upon retrieved-augmented generation to improve factuality through iterative prompting but these methods are limited by the traditional RAG design. To address these challenges, we introduce EWE (Explicit Working Memory), a novel approach that enhances factuality in long-form text generation by integrating a working memory that receives real-time feedback from external resources. The memory is refreshed based on online fact-checking and retrieval feedback, allowing EWE to rectify false claims during the generation process and ensure more accurate and reliable outputs. Our experiments demonstrate that Ewe outperforms strong baselines on four fact-seeking long-form generation datasets, increasing the factuality metric, VeriScore, by 2 to 10 points absolute without sacrificing the helpfulness of the responses. Further analysis reveals that the design of rules for memory updates, configurations of memory units, and the quality of the retrieval datastore are crucial factors for influencing model performance. 

**Abstract (ZH)**: 大型语言模型可以生成事实不准确的内容，这一问题被称为幻觉。近年来的研究通过迭代提示来增强检索增强生成方法以提高事实准确性，但这些方法受限于传统的RAG设计。为了解决这些问题，我们提出了一种名为EWE（显式工作记忆）的新颖方法，该方法通过集成一个实时从外部资源接收反馈的工作记忆来增强长文本生成的事实准确性。该工作记忆根据在线事实核查和检索反馈进行更新，使EWE在生成过程中能够更正虚假声明，从而确保输出更加准确可靠。我们的实验表明，EWE在四个事实寻求的长文本生成数据集上表现优于强大的基线方法，在不牺牲回复的有用性的情况下，VeriScore准确性指标提高了2到10个百分点。进一步的分析表明，记忆更新规则的设计、记忆单元的配置以及检索数据存储的质量对模型性能有重要影响。 

---
# Neuron Empirical Gradient: Connecting Neurons' Linear Controllability and Representational Capacity 

**Title (ZH)**: 神经元经验梯度：连接神经元的线性可控性和表示能力 

**Authors**: Xin Zhao, Zehui Jiang, Naoki Yoshinaga  

**Link**: [PDF](https://arxiv.org/pdf/2412.18053)  

**Abstract**: Although neurons in the feed-forward layers of pre-trained language models (PLMs) can store factual knowledge, most prior analyses remain qualitative, leaving the quantitative relationship among knowledge representation, neuron activations, and model output poorly understood. In this study, by performing neuron-wise interventions using factual probing datasets, we first reveal the linear relationship between neuron activations and output token probabilities. We refer to the gradient of this linear relationship as ``neuron empirical gradients.'' and propose NeurGrad, an efficient method for their calculation to facilitate quantitative neuron analysis. We next investigate whether neuron empirical gradients in PLMs encode general task knowledge by probing skill neurons. To this end, we introduce MCEval8k, a multi-choice knowledge evaluation benchmark spanning six genres and 22 tasks. Our experiments confirm that neuron empirical gradients effectively capture knowledge, while skill neurons exhibit efficiency, generality, inclusivity, and interdependency. These findings link knowledge to PLM outputs via neuron empirical gradients, shedding light on how PLMs store knowledge. The code and dataset are released. 

**Abstract (ZH)**: 尽管预训练语言模型（PLMs）中的前向层神经元可以存储事实知识，但大多数前期分析仍停留在定性的层面，定量地理解知识表示、神经元激活与模型输出之间的关系仍不清楚。在这种背景下，本研究通过使用事实探究数据集执行神经元级干预，首次揭示了神经元激活与输出标记概率之间的线性关系。我们将这种线性关系的梯度称为“神经元经验梯度”（Neuron Empirical Gradients, NEGs），并提出了一种高效的方法NeurGrad来计算这些梯度，以促进神经元的定量分析。接下来，我们调查了PLMs中神经元经验梯度是否编码了一般任务知识，为此我们引入了一个名为MCEval8k的多选择知识评估基准，覆盖六个类型和22项任务。实验结果证实，神经元经验梯度能够有效地捕捉知识，而技能神经元则表现出效率、通用性、包容性和相互依赖性。这些发现通过神经元经验梯度将知识与PLM输出联系起来，揭示了PLM存储知识的方式。代码和数据集已公开发布。 

---
# Factuality or Fiction? Benchmarking Modern LLMs on Ambiguous QA with Citations 

**Title (ZH)**: 事实还是虚构？现代大规模语言模型在有歧义的问答任务上的基准测试及引文分析 

**Authors**: Maya Patel, Aditi Anand  

**Link**: [PDF](https://arxiv.org/pdf/2412.18051)  

**Abstract**: Benchmarking modern large language models (LLMs) on complex and realistic tasks is critical to advancing their development. In this work, we evaluate the factual accuracy and citation performance of state-of-the-art LLMs on the task of Question Answering (QA) in ambiguous settings with source citations. Using three recently published datasets-DisentQA-DupliCite, DisentQA-ParaCite, and AmbigQA-Cite-featuring a range of real-world ambiguities, we analyze the performance of two leading LLMs, GPT-4o-mini and Claude-3.5. Our results show that larger, recent models consistently predict at least one correct answer in ambiguous contexts but fail to handle cases with multiple valid answers. Additionally, all models perform equally poorly in citation generation, with citation accuracy consistently at 0. However, introducing conflict-aware prompting leads to large improvements, enabling models to better address multiple valid answers and improve citation accuracy, while maintaining their ability to predict correct answers. These findings highlight the challenges and opportunities in developing LLMs that can handle ambiguity and provide reliable source citations. Our benchmarking study provides critical insights and sets a foundation for future improvements in trustworthy and interpretable QA systems. 

**Abstract (ZH)**: 在复杂且现实的任务中基准测试现代大型语言模型（LLMs）对于推动其发展至关重要。本研究旨在评估最先进LLMs在具有源引文的模糊环境中问答（QA）任务中的事实在准确性及引文性能。我们使用了三种最近发布的数据集——DisentQA-DupliCite、DisentQA-ParaCite 和 AmbigQA-Cite，涵盖了各种现实世界的模糊性。我们分析了两种领先的LLM，GPT-4o-mini 和 Claude-3.5，在这些数据集上的表现。研究结果表明，较大的、较新的模型在模糊环境中一致能够预测至少一个正确答案，但难以处理具有多个正确答案的情况。此外，所有模型在引文生成方面的表现都极其不佳，引文准确性始终为0。然而，引入冲突感知的提示对模型表现产生了显著改善，使模型更好地处理多重正确答案并提高引文准确性，同时保持预测正确答案的能力。这些发现突显了开发能够处理模糊性和提供可靠引文数据的LLMs所面临的挑战和机遇。我们的基准测试研究提供了重要的见解，并为未来的可信和可解释的QA系统改进奠定了基础。 

---
# Aligning AI Research with the Needs of Clinical Coding Workflows: Eight Recommendations Based on US Data Analysis and Critical Review 

**Title (ZH)**: 将AI研究与临床编码工作流程需求对齐：基于美国数据的分析与批判性审查的八大建议 

**Authors**: Yidong Gan, Maciej Rybinski, Ben Hachey, Jonathan K. Kummerfeld  

**Link**: [PDF](https://arxiv.org/pdf/2412.18043)  

**Abstract**: Clinical coding is crucial for healthcare billing and data analysis. Manual clinical coding is labour-intensive and error-prone, which has motivated research towards full automation of the process. However, our analysis, based on US English electronic health records and automated coding research using these records, shows that widely used evaluation methods are not aligned with real clinical contexts. For example, evaluations that focus on the top 50 most common codes are an oversimplification, as there are thousands of codes used in practice. This position paper aims to align AI coding research more closely with practical challenges of clinical coding. Based on our analysis, we offer eight specific recommendations, suggesting ways to improve current evaluation methods. Additionally, we propose new AI-based methods beyond automated coding, suggesting alternative approaches to assist clinical coders in their workflows. 

**Abstract (ZH)**: 临床编码对于医疗收费和数据分析至关重要。手工临床编码劳动密集且容易出错，因此推动了全自动化过程的研究。然而，基于对美国英语电子健康记录的分析以及使用这些记录进行的自动化编码研究，我们发现广泛采用的评估方法未能与实际临床情境保持一致。例如，仅针对最常见的前50个代码的评估过于简化，因为在实践中使用的代码多达数千个。本文旨在使人工智能编码研究更加贴近临床编码的实际挑战。基于我们的分析，我们提出了八项具体建议，旨在改进现有的评估方法。此外，我们还提出了超越自动化编码的新型人工智能方法，为临床编码者的日常工作流程提供替代方案。 

---
# Explainability in Neural Networks for Natural Language Processing Tasks 

**Title (ZH)**: 神经网络在自然语言处理任务中的可解释性 

**Authors**: Melkamu Mersha, Mingiziem Bitewa, Tsion Abay, Jugal Kalita  

**Link**: [PDF](https://arxiv.org/pdf/2412.18036)  

**Abstract**: Neural networks are widely regarded as black-box models, creating significant challenges in understanding their inner workings, especially in natural language processing (NLP) applications. To address this opacity, model explanation techniques like Local Interpretable Model-Agnostic Explanations (LIME) have emerged as essential tools for providing insights into the behavior of these complex systems. This study leverages LIME to interpret a multi-layer perceptron (MLP) neural network trained on a text classification task. By analyzing the contribution of individual features to model predictions, the LIME approach enhances interpretability and supports informed decision-making. Despite its effectiveness in offering localized explanations, LIME has limitations in capturing global patterns and feature interactions. This research highlights the strengths and shortcomings of LIME and proposes directions for future work to achieve more comprehensive interpretability in neural NLP models. 

**Abstract (ZH)**: 神经网络通常被视为黑箱模型，这为理解其内部工作机制带来了重大挑战，尤其是在自然语言处理（NLP）应用中。为了解决这一透明度问题，像局部可解释模型无偏解释（LIME）这样的模型解释技术成为了提供这些复杂系统行为见解的重要工具。本研究利用LIME来解释用于文本分类任务的多层感知机（MLP）神经网络。通过分析单个特征对模型预测的贡献，LIME方法提高了可解释性并支持了基于信息的决策。尽管LIME在提供局部解释方面效果显著，但它在捕捉全局模式和特征互作方面存在一定局限性。本研究突出了LIME的优点和不足，并提出了未来工作中实现神经NLP模型更全面可解释性的方向。 

---
# Same Company, Same Signal: The Role of Identity in Earnings Call Transcripts 

**Title (ZH)**: 相同公司，相同信号：身份在earnings call纪要中的作用 

**Authors**: Ding Yu, Zhuo Liu, Hangfeng He  

**Link**: [PDF](https://arxiv.org/pdf/2412.18029)  

**Abstract**: Post-earnings volatility prediction is critical for investors, with previous works often leveraging earnings call transcripts under the assumption that their rich semantics contribute significantly. To further investigate how transcripts impact volatility, we introduce DEC, a dataset featuring accurate volatility calculations enabled by the previously overlooked beforeAfterMarket attribute and dense ticker coverage. Unlike established benchmarks, where each ticker has only around two earnings, DEC provides 20 earnings records per ticker. Using DEC, we reveal that post-earnings volatility undergoes significant shifts, with each ticker displaying a distinct volatility distribution. To leverage historical post-earnings volatility and capture ticker-specific patterns, we propose two training-free baselines: Post-earnings Volatility (PEV) and Same-ticker Post-earnings Volatility (STPEV). These baselines surpass all transcripts-based models on DEC as well as on established benchmarks. Additionally, we demonstrate that current transcript representations predominantly capture ticker identity rather than offering financially meaningful insights specific to each earnings. This is evidenced by two key observations: earnings representations from the same ticker exhibit significantly higher similarity compared to those from different tickers, and predictions from transcript-based models show strong correlations with prior post-earnings volatility. 

**Abstract (ZH)**: 基于公告的波动率预测对于投资者至关重要，此前的研究往往依赖于财报电话会议纪要，假设其丰富的语义信息对波动率预测贡献显著。为进一步探讨电话会议纪要如何影响波动率，我们引入了DEC数据集，该数据集通过以前被忽视的beforeAfterMarket（公告前/后）属性和密集的股票覆盖率，实现了精确的波动率计算。与其他公认的基准数据集相比，DEC每只股票提供了20份财报记录。利用DEC数据集，我们发现基于公告的波动率经历了显著的变化，每只股票显示出独特的波动率分布。为了利用历史公告后的波动率并捕捉股票特定的模式，我们提出了两种无需训练的基准模型：后公告波动率（Post-earnings Volatility，PEV）和同股票标识的后公告波动率（Same-ticker Post-earnings Volatility，STPEV）。这些基准模型在DEC以及公认的基准数据集上都超过了基于公告的所有模型。此外，我们证明了当前的公告表示主要捕获了股票标识性信息，而不是提供了对每份财报具有财务意义的特定洞察。这一发现通过两个关键观察得到了证实：来自同一股票的财报表示显示出显著更高的相似性，而基于公告的模型的预测与之前的公告后波动率高度相关。 

---
# StructTest: Benchmarking LLMs' Reasoning through Compositional Structured Outputs 

**Title (ZH)**: StructTest: 通过组成结构化输出对LLMs推理能力进行基准测试 

**Authors**: Hailin Chen, Fangkai Jiao, Mathieu Ravaut, Nawshad Farruque, Xuan Phi Nguyen, Chengwei Qin, Manan Dey, Bosheng Ding, Caiming Xiong, Shafiq Joty, Yingbo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2412.18011)  

**Abstract**: The rapid development of large language models (LLMs) necessitates robust, unbiased, and scalable methods for evaluating their capabilities. However, human annotations are expensive to scale, model-based evaluations are prone to biases in answer style, while target-answer-based benchmarks are vulnerable to data contamination and cheating. To address these limitations, we propose StructTest, a novel benchmark that evaluates LLMs on their ability to produce compositionally specified structured outputs as an unbiased, cheap-to-run and difficult-to-cheat measure. The evaluation is done deterministically by a rule-based evaluator, which can be easily extended to new tasks. By testing structured outputs across diverse task domains -- including Summarization, Code, HTML and Math -- we demonstrate that StructTest serves as a good proxy for general reasoning abilities, as producing structured outputs often requires internal logical reasoning. We believe that StructTest offers a critical, complementary approach to objective and robust model evaluation. 

**Abstract (ZH)**: 大型语言模型（LLMs）的快速发展迫切需要稳健、无偏见且可扩展的评估方法来衡量其能力。然而，人工标注成本高昂，基于模型的评估容易受到答案风格偏见的影响，而基于目标答案的基准则容易受到数据污染和作弊的威胁。为解决这些限制，我们提出了一种名为StructTest的新基准方法，该方法通过一种无偏见、低成本且难以作弊的方式评估LLMs生成组合性结构化输出的能力。评估通过基于规则的评估器完成，可以轻松扩展到新的任务。通过在包括摘要、代码、HTML和数学在内的多个任务领域测试结构化输出，我们证明了StructTest可以作为一般推理能力的良好代理指标，因为生成结构化输出通常需要内部逻辑推理。我们认为，StructTest提供了客观且稳健模型评估的一种关键补充方法。 

---
# Correctness is not Faithfulness in RAG Attributions 

**Title (ZH)**: 正确性不等同于忠実性在RAG归因中 

**Authors**: Jonas Wallat, Maria Heuss, Maarten de Rijke, Avishek Anand  

**Link**: [PDF](https://arxiv.org/pdf/2412.18004)  

**Abstract**: Retrieving relevant context is a common approach to reduce hallucinations and enhance answer reliability. Explicitly citing source documents allows users to verify generated responses and increases trust. Prior work largely evaluates citation correctness - whether cited documents support the corresponding statements. But citation correctness alone is insufficient. To establish trust in attributed answers, we must examine both citation correctness and citation faithfulness. In this work, we first disentangle the notions of citation correctness and faithfulness, which have been applied inconsistently in previous studies. Faithfulness ensures that the model's reliance on cited documents is genuine, reflecting actual reference use rather than superficial alignment with prior beliefs, which we call post-rationalization. We design an experiment that reveals the prevalent issue of post-rationalization, which undermines reliable attribution and may result in misplaced trust. Our findings suggest that current attributed answers often lack citation faithfulness (up to 57 percent of the citations), highlighting the need to evaluate correctness and faithfulness for trustworthy attribution in language models. 

**Abstract (ZH)**: 检索相关上下文是减少幻觉和提高答案可靠性的一种常见方法。明确引用源文档可以让用户验证生成的答案并增加信任度。以往的工作主要评估引用的正确性——即所引用的文档是否支持相应的陈述。然而，引用的正确性本身是不够的。为了建立对归因答案的信任，我们必须同时评估引用的正确性和忠实性。在本研究中，我们首先区分了引用正确性和忠实性这两个概念，这两个概念在以往的研究中应用不一致。忠实性确保模型依赖引用的文档是真实的，反映了实际的参考使用，而不是表面化的与先验信念的一致性，这被称为后理性化。我们设计了一个实验，揭示了后理性化这一普遍存在的问题，它削弱了可靠的归因，并可能导致不适当的信任。我们的研究发现表明，当前的归因答案常常缺乏引用忠实性（高达57%的引用），突显了在语言模型中评估正确性和忠实性以实现可靠归因的必要性。 

---
# CARL-GT: Evaluating Causal Reasoning Capabilities of Large Language Models 

**Title (ZH)**: CARL-GT：评估大型语言模型的因果推理能力 

**Authors**: Ruibo Tu, Hedvig Kjellström, Gustav Eje Henter, Cheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.17970)  

**Abstract**: Causal reasoning capabilities are essential for large language models (LLMs) in a wide range of applications, such as education and healthcare. But there is still a lack of benchmarks for a better understanding of such capabilities. Current LLM benchmarks are mainly based on conversational tasks, academic math tests, and coding tests. Such benchmarks evaluate LLMs in well-regularized settings, but they are limited in assessing the skills and abilities to solve real-world problems. In this work, we provide a benchmark, named by CARL-GT, which evaluates CAusal Reasoning capabilities of large Language models using Graphs and Tabular data. The benchmark has a diverse range of tasks for evaluating LLMs from causal graph reasoning, knowledge discovery, and decision-making aspects. In addition, effective zero-shot learning prompts are developed for the tasks. In our experiments, we leverage the benchmark for evaluating open-source LLMs and provide a detailed comparison of LLMs for causal reasoning abilities. We found that LLMs are still weak in casual reasoning, especially with tabular data to discover new insights. Furthermore, we investigate and discuss the relationships of different benchmark tasks by analyzing the performance of LLMs. The experimental results show that LLMs have different strength over different tasks and that their performance on tasks in different categories, i.e., causal graph reasoning, knowledge discovery, and decision-making, shows stronger correlation than tasks in the same category. 

**Abstract (ZH)**: 因果推理能力对于大型语言模型（LLMs）在各种应用中，如教育和医疗领域，至关重要。然而，目前尚缺乏相关的基准测试以更好地理解这种能力。当前的LLM基准主要基于对话任务、学术数学测试和编程测试。这些基准测试在设定良好的规则环境中评估LLM的能力，但它们在评估解决实际问题所需技能和能力方面存在局限性。在本研究中，我们提供了一个名为CARL-GT的基准测试，该基准测试使用图形和表格数据评估大型语言模型的因果推理能力。基准测试涵盖了一系列多样化的任务，从因果图推理、知识发现和决策制定方面评估LLM。此外，还开发了有效的零样本学习提示以应用于这些任务。在实验中，我们利用基准测试评估开源LLM，并详细比较了LLM在因果推理能力方面的表现。我们发现LLM在因果推理方面仍然较弱，特别是在从表格数据中发现新见解方面表现不佳。此外，我们通过分析LLM在不同基准测试任务中的表现，探讨了不同任务之间的关系。实验结果表明，LLM在不同任务上的表现各异，并且它们在不同类别任务（因果图推理、知识发现和决策制定）上的表现相关性更强，而不是在同一个类别内的任务。 

---
# Path-of-Thoughts: Extracting and Following Paths for Robust Relational Reasoning with Large Language Models 

**Title (ZH)**: 基于思维路径：从大型语言模型中提取和追踪路径以实现稳健的关系推理 

**Authors**: Ge Zhang, Mohammad Ali Alomrani, Hongjian Gu, Jiaming Zhou, Yaochen Hu, Bin Wang, Qun Liu, Mark Coates, Yingxue Zhang, Jianye Hao  

**Link**: [PDF](https://arxiv.org/pdf/2412.17963)  

**Abstract**: Large language models (LLMs) possess vast semantic knowledge but often struggle with complex reasoning tasks, particularly in relational reasoning problems such as kinship or spatial reasoning. In this paper, we present Path-of-Thoughts (PoT), a novel framework designed to tackle relation reasoning by decomposing the task into three key stages: graph extraction, path identification, and reasoning. Unlike previous approaches, PoT efficiently extracts a task-agnostic graph that identifies crucial entities, relations, and attributes within the problem context. Subsequently, PoT identifies relevant reasoning chains within the graph corresponding to the posed question, facilitating inference of potential answers. Experimental evaluations on four benchmark datasets, demanding long reasoning chains, demonstrate that PoT surpasses state-of-the-art baselines by a significant margin (maximum 21.3%) without necessitating fine-tuning or extensive LLM calls. Furthermore, as opposed to prior neuro-symbolic methods, PoT exhibits improved resilience against LLM errors by leveraging the compositional nature of graphs. 

**Abstract (ZH)**: 大语言模型（LLMs）蕴含了广泛的语言知识，但在处理复杂的推理任务时常常表现不佳，特别是在亲缘关系或空间推理等关系推理问题上。本文提出了一种名为路径思维（Path-of-Thoughts, PoT）的新型框架，旨在通过将任务分解为三个关键阶段来应对关系推理问题：图提取、路径识别和推理。与之前的几种方法不同，PoT 有效地提取了一个任务无关的图，该图能够识别问题上下文中关键实体、关系和属性。随后，PoT 在图中识别出与提出的问题相关的推理链，从而有助于潜在答案的推断。通过在四个基准数据集上进行实证评估，这些数据集要求处理较长的推理链，结果显示PoT 在不需微调或大量调用LLM的情况下显著超越了现有最先进的基线方法（最高超过21.3%）。此外，与之前的神经符号方法相比，PoT 通过利用图的组合性质展示了更好的鲁棒性，能够更好地抵御LLM的错误。 

---
# IITR-CIOL@NLU of Devanagari Script Languages 2025: Multilingual Hate Speech Detection and Target Identification in Devanagari-Scripted Languages 

**Title (ZH)**: IITR-CIOL@NLU 2025 蒙德拉加里语系语言的多语言仇视言论检测与目标识别 

**Authors**: Siddhant Gupta, Siddh Singhal, Azmine Toushik Wasi  

**Link**: [PDF](https://arxiv.org/pdf/2412.17947)  

**Abstract**: This work focuses on two subtasks related to hate speech detection and target identification in Devanagari-scripted languages, specifically Hindi, Marathi, Nepali, Bhojpuri, and Sanskrit. Subtask B involves detecting hate speech in online text, while Subtask C requires identifying the specific targets of hate speech, such as individuals, organizations, or communities. We propose the MultilingualRobertaClass model, a deep neural network built on the pretrained multilingual transformer model ia-multilingual-transliterated-roberta, optimized for classification tasks in multilingual and transliterated contexts. The model leverages contextualized embeddings to handle linguistic diversity, with a classifier head for binary classification. We received 88.40% accuracy in Subtask B and 66.11% accuracy in Subtask C, in the test set. 

**Abstract (ZH)**: 本文专注于与印地语系语言（包括印地语、马拉地语、尼泊尔语、布贾普里语和梵语）相关的仇恨言论检测和目标识别的两个子任务。子任务 B 涉及在线文本中的仇恨言论检测，而子任务 C 则要求识别仇恨言论的具体目标，比如个人、组织或社区。我们提出了一种名为 MultilingualRobertaClass 的模型，该模型基于预训练的多语言变压器模型 ia-multilingual-transliterated-roberta，专门针对多语言和转写语境下的分类任务进行了优化。该模型利用上下文化嵌入处理语义多样性，并包含一个用于二分类的任务头。在测试集中，我们在子任务 B 中获得了 88.40% 的准确率，在子任务 C 中获得了 66.11% 的准确率。 

---
# BenCzechMark : A Czech-centric Multitask and Multimetric Benchmark for Large Language Models with Duel Scoring Mechanism 

**Title (ZH)**: BenCzechMark：一种基于捷克语的多任务和多指标基准测试，配备 Duel 评分机制的大语言模型 

**Authors**: Martin Fajcik, Martin Docekal, Jan Dolezal, Karel Ondrej, Karel Beneš, Jan Kapsa, Pavel Smrz, Alexander Polok, Michal Hradis, Zuzana Neverilova, Ales Horak, Radoslav Sabol, Michal Stefanik, Adam Jirkovsky, David Adamczyk, Petr Hyner, Jan Hula, Hynek Kydlicek  

**Link**: [PDF](https://arxiv.org/pdf/2412.17933)  

**Abstract**: We present BenCzechMark (BCM), the first comprehensive Czech language benchmark designed for large language models, offering diverse tasks, multiple task formats, and multiple evaluation metrics. Its scoring system is grounded in statistical significance theory and uses aggregation across tasks inspired by social preference theory. Our benchmark encompasses 50 challenging tasks, with corresponding test datasets, primarily in native Czech, with 11 newly collected ones. These tasks span 8 categories and cover diverse domains, including historical Czech news, essays from pupils or language learners, and spoken word.
Furthermore, we collect and clean BUT-Large Czech Collection, the largest publicly available clean Czech language corpus, and use it for (i) contamination analysis, (ii) continuous pretraining of the first Czech-centric 7B language model, with Czech-specific tokenization. We use our model as a baseline for comparison with publicly available multilingual models. Lastly, we release and maintain a leaderboard, with existing 44 model submissions, where new model submissions can be made at this https URL. 

**Abstract (ZH)**: 我们提出了一种名为BenCzechMark（BCM）的大规模语言模型基准测试，这是第一个全面的捷克语言基准测试，旨在为大型语言模型提供多样的任务、多种任务格式和多种评估指标。其评分系统基于统计显著性理论，并借鉴了社会偏好理论中的聚合方法。我们的基准测试包含50个具有挑战性的任务，其中11个是新收集的数据集，对应的任务集主要使用捷克本土语言，涵盖了8个类别，包括历史捷克新闻、学生或语言学习者的作文以及口语内容。

此外，我们收集并清理了BUT-Large捷克语集合，这是目前最大的公开可用的清洁捷克语言语料库，并将其用于（i）污染分析；（ii）连续预训练第一个以捷克为中心的7B语言模型，该模型具有捷克语特定的分词。我们使用该模型作为基线，与现有的多语言模型进行比较。最后，我们提供并维护了一个排行榜，已有44个模型提交（提交地址为：[此链接]）。新模型提交可通过上述链接进行。 

---
# The Power of Adaptation: Boosting In-Context Learning through Adaptive Prompting 

**Title (ZH)**: 适应的力量：通过自适应提示增强上下文学习 

**Authors**: Shuzhang Cai, Twumasi Mensah-Boateng, Xander Kuksov, Jing Yuan, Shaojie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2412.17891)  

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional abilities across a broad range of language-related tasks, including generating solutions to complex reasoning problems. An effective technique to enhance LLM performance is in-context learning, which encourages a step-by-step reasoning process by including explanatory examples to guide the model's responses. However, selecting appropriate exemplars for the model poses a challenge, as each dataset demands a distinct set of exemplars to enable the LLM to learn effectively and perform well on the test set. Current studies often rely on uncertainty- or diversity-based selection strategies to select exemplars for annotation and to improve model learning. However, these studies typically employ a non-adaptive approach, selecting a set of exemplars all at once. We argue that this non-adaptive strategy may result in a set of exemplars with high redundancy in terms of the knowledge covered, ultimately reducing their overall informativeness. To address this limitation, we propose \textsc{Adaptive-Prompt}, a novel method that adaptively selects exemplars by leveraging model feedback from previously chosen exemplars. Experimental results show that \textsc{Adaptive-Prompt} significantly enhances LLM performance across a variety of reasoning tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）在广泛的语言相关任务中展示了出色的能力，包括解决复杂推理问题。有效提高LLM性能的一种方法是内部上下文学习，这种方法通过提供解释性的示例来引导模型的回答，以促进逐步的推理过程。然而，为模型选择合适的示例是一个挑战，因为每个数据集都要求一组独特的示例来帮助LLM有效学习并在测试集中表现良好。当前的研究通常依赖于基于不确定性或多样性选择策略来选择需要标注的示例并改善模型的学习。然而，这些研究通常采用非适应性方法，一次性选择一组示例。我们认为这种方法可能会导致选择的一组示例在涵盖的知识方面具有高度冗余，从而降低其整体信息量。为了解决这一局限性，我们提出了一种名为\textsc{Adaptive-Prompt}的新方法，该方法通过利用之前选择的示例的模型反馈来适应性地选择示例。实验结果表明，\textsc{Adaptive-Prompt}显著提高了LLM在各种推理任务中的性能。 

---
# Evaluating LLM Reasoning in the Operations Research Domain with ORQA 

**Title (ZH)**: 使用ORQA评估大型语言模型在运筹学领域的推理能力 

**Authors**: Mahdi Mostajabdaveh, Timothy T. Yu, Samarendra Chandan Bindu Dash, Rindranirina Ramamonjison, Jabo Serge Byusa, Giuseppe Carenini, Zirui Zhou, Yong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.17874)  

**Abstract**: In this paper, we introduce and apply Operations Research Question Answering (ORQA), a new benchmark designed to assess the generalization capabilities of Large Language Models (LLMs) in the specialized technical domain of Operations Research (OR). This benchmark evaluates whether LLMs can emulate the knowledge and reasoning skills of OR experts when confronted with diverse and complex optimization problems. The dataset, developed by OR experts, features real-world optimization problems that demand multistep reasoning to construct their mathematical models. Our evaluations of various open source LLMs, such as LLaMA 3.1, DeepSeek, and Mixtral, reveal their modest performance, highlighting a gap in their ability to generalize to specialized technical domains. This work contributes to the ongoing discourse on LLMs generalization capabilities, offering valuable insights for future research in this area. The dataset and evaluation code are publicly available. 

**Abstract (ZH)**: 在这篇论文中，我们引入并应用了运筹学问答（ORQA），一个新的基准，旨在评估大型语言模型（LLM）在运筹学（OR）这一专门技术领域的泛化能力。该基准评估LLM是否能在面对多样化和复杂优化问题时，模仿运筹学专家的知识和推理技能。该数据集由运筹学专家精心设计，包含需要多步推理来构建其数学模型的实际优化问题。我们对多种开源LLM（如LLaMA 3.1、DeepSeek和Mixtral）的评估显示，它们的性能较为有限，这突显了它们在泛化到专门技术领域方面的能力差距。这项工作为LLM泛化能力的持续讨论做出了贡献，并为该领域的未来研究提供了宝贵见解。该数据集和评估代码已公开提供。 

---
# Joint Knowledge Editing for Information Enrichment and Probability Promotion 

**Title (ZH)**: 联合知识编辑以促进信息丰富和概率提升 

**Authors**: Wenhang Shi, Yiren Chen, Shuqing Bian, Xinyi Zhang, Zhe Zhao, Pengfei Hu, Wei Lu, Xiaoyong Du  

**Link**: [PDF](https://arxiv.org/pdf/2412.17872)  

**Abstract**: Knowledge stored in large language models requires timely updates to reflect the dynamic nature of real-world information. To update the knowledge, most knowledge editing methods focus on the low layers, since recent probes into the knowledge recall process reveal that the answer information is enriched in low layers. However, these probes only and could only reveal critical recall stages for the original answers, while the goal of editing is to rectify model's prediction for the target answers. This inconsistency indicates that both the probe approaches and the associated editing methods are deficient. To mitigate the inconsistency and identify critical editing regions, we propose a contrast-based probe approach, and locate two crucial stages where the model behavior diverges between the original and target answers: Information Enrichment in low layers and Probability Promotion in high layers. Building upon the insights, we develop the Joint knowledge Editing for information Enrichment and probability Promotion (JEEP) method, which jointly edits both the low and high layers to modify the two critical recall stages. Considering the mutual interference and growing forgetting due to dual modifications, JEEP is designed to ensure that updates to distinct regions share the same objectives and are complementary. We rigorously evaluate JEEP by editing up to thousands of facts on various models, i.e., GPT-J (6B) and LLaMA (7B), and addressing diverse editing objectives, i.e., adding factual and counterfactual knowledge. In all tested scenarios, JEEP achieves best performances, validating the effectiveness of the revealings of our probe approach and the designs of our editing method. Our code and data are available at this https URL. 

**Abstract (ZH)**: 大型语言模型中存储的知识需要及时更新，以反映现实世界信息的动态性质。为了更新知识，大多数知识编辑方法侧重于低层，因为最近对知识回忆过程的探查表明，答案信息丰富于低层。然而，这些探查只能揭示原始答案的关键回忆阶段，而编辑的目的是纠正模型对目标答案的预测。这种不一致表明，现有的探查方法和相关的编辑方法都是不足的。为了缓解这种不一致并识别关键的编辑区域，我们提出了一种对比探针方法，并定位了模型在原始答案与目标答案之间行为分歧的两个关键阶段：低层的信息丰富和高层的概率提升。基于这些见解，我们开发了联合知识编辑方法（Joint Knowledge Editing for Information Enhancement and Probability Promotion, JEEP），该方法同时编辑低层和高层，以修改这两个关键回忆阶段。考虑到由于双重修改导致的相互干扰和逐渐遗忘，JEEP 设计为确保不同区域的更新具有相同的目标并相互补充。我们通过在多种模型（如 GPT-J（6B）和 LLaMA（7B））上编辑多达数千个事实，并解决各种编辑目标（如添加事实性和反事实性知识）来严格评估 JEEP。在所有测试场景中，JEEP 均表现出最佳性能，验证了我们探针方法揭示的有效性和我们编辑方法设计的有效性。我们的代码和数据可通过以下链接获得：[这里](this https URL)。 

---
# Evaluating and Enhancing LLMs for Multi-turn Text-to-SQL with Multiple Question Types 

**Title (ZH)**: 评估并提升针对多轮文本到SQL转换的大型语言模型的性能，涵盖多种问题类型 

**Authors**: Ziming Guo, Chao Ma, Yinggang Sun, Tiancheng Zhao, Guangyao Wang, Hai Huang  

**Link**: [PDF](https://arxiv.org/pdf/2412.17867)  

**Abstract**: Recent advancements in large language models (LLMs) have significantly advanced text-to-SQL systems. However, most LLM-based methods often narrowly focus on SQL generation, neglecting the complexities of real-world conversational queries. This oversight can lead to unreliable responses, particularly for ambiguous questions that cannot be directly addressed with SQL. To bridge this gap, we propose MMSQL, a comprehensive test suite designed to evaluate the question classification and SQL generation capabilities of LLMs by simulating real-world scenarios with diverse question types and multi-turn Q\&A interactions. Using MMSQL, we assessed the performance of popular LLMs, including both open-source and closed-source models, and identified key factors impacting their performance in such scenarios. Moreover, we introduce an LLM-based multi-agent framework that employs specialized agents to identify question types and determine appropriate answering strategies. Our experiments demonstrate that this approach significantly enhances the model's ability to navigate the complexities of conversational dynamics, effectively handling the diverse and complex nature of user queries. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）的进展显著提升了文本到SQL系统的性能。然而，大多数基于LLM的方法往往仅专注于SQL生成，忽视了真实世界对话查询的复杂性。这种忽视可能导致不可靠的响应，特别是对于那些不能直接用SQL回答的含糊不清的问题。为了弥补这一差距，我们提出了MMSQL，这是一个全面的测试套件，通过模拟具有多种问题类型和多轮问答交互的真实场景，评估LLM的问题分类和SQL生成能力。通过MMSQL，我们评估了几种流行的LLM，包括开源和封闭源模型，并确定了影响其在这种情况下表现的关键因素。此外，我们引入了一种基于LLM的多代理框架，该框架使用专门的代理来识别问题类型并确定合适的回答策略。我们的实验表明，这种方法显著增强了模型适应对话动态复杂性的能力，有效处理了用户查询的多样性和复杂性。 

---
# Overview of the 2024 ALTA Shared Task: Detect Automatic AI-Generated Sentences for Human-AI Hybrid Articles 

**Title (ZH)**: 2024年ALTA共享任务概览：检测人工与AI混合文章中的自动AI生成句子 

**Authors**: Diego Mollá, Qiongkai Xu, Zijie Zeng, Zhuang Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.17848)  

**Abstract**: The ALTA shared tasks have been running annually since 2010. In 2024, the purpose of the task is to detect machine-generated text in a hybrid setting where the text may contain portions of human text and portions machine-generated. In this paper, we present the task, the evaluation criteria, and the results of the systems participating in the shared task. 

**Abstract (ZH)**: 自2010年起，ALTA共享任务已每年举行一次。2024年的任务目的是在半自动环境中检测机器生成的文字，该环境中可能包含部分人类撰写的文本和部分机器生成的文本。在这篇论文中，我们介绍了该任务、评估标准以及参与共享任务的系统的性能结果。 

---
# Enhancing Knowledge Distillation for LLMs with Response-Priming Prompting 

**Title (ZH)**: 用响应引导提示增强知识蒸馏以优化大规模语言模型 

**Authors**: Vijay Goyal, Mustafa Khan, Aprameya Tirupati, Harveer Saini, Michael Lam, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2412.17846)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable performance across a wide range of natural language processing (NLP) tasks. However, these models are often difficult to deploy due to significant computational requirements and resource constraints. Knowledge distillation (KD) is an effective technique for transferring the performance of larger LLMs to smaller models. Traditional KD methods primarily focus on the direct output of the teacher model, with little emphasis on the role of prompting during knowledge transfer. In this paper, we propose a set of novel response-priming prompting strategies applied in the knowledge distillation pipeline to enhance the performance of student models. Our approach fine-tunes a smaller Llama 3.1 8B Instruct model by distilling knowledge from a quantized Llama 3.1 405B Instruct teacher model. We apply LoRA optimization and evaluate on the GSM8K benchmark. Experimental results demonstrate that integrating reasoning-eliciting prompting into the proposed KD pipeline significantly improves student model performance, offering an efficient way to deploy powerful models in resource-constrained environments. We find that Ground Truth prompting results in a 55\% performance increase on GSM8K for a distilled Llama 3.1 8B Instruct compared to the same model distilled without prompting. A thorough investigation into the self-attention layers of the student models indicates that the more successful prompted models tend to exhibit certain positive behaviors inside their attention heads which can be tied to their increased accuracy. Our implementation can be found at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在广泛的语言处理任务中表现出色。然而，这些模型由于计算需求大和资源限制，往往难以部署。知识蒸馏（KD）是一种有效的方法，可以将大型LLM的性能转移到较小的模型中。传统的方法主要关注教师模型的直接输出，而对知识转移过程中提示的作用关注较少。本文提出了一套新颖的响应促进提示策略，应用于知识蒸馏管道中，以提高学生模型的性能。我们的方法通过对量化后的Llama 3.1 405B Instruct教师模型进行知识蒸馏，来微调一个较小的Llama 3.1 8B Instruct学生模型，并应用LoRA优化，在GSM8K基准上进行评估。实验结果表明，将推理促进提示整合到所提出的KD管道中，显著提高了学生模型的性能，为在资源受限环境中部署强大模型提供了有效途径。我们发现，使用Ground Truth提示可以将蒸馏后的Llama 3.1 8B Instruct模型在GSM8K上的性能提高55%。通过对学生模型的自我注意层进行深入研究，发现更成功的提示模型通常在其注意头中表现出某些积极的行为，这些行为与其更高的准确性有关。我们的实现代码可以在这个链接访问：[这里提供一个链接]。 

---
# Evaluating the Capabilities of Large Language Models for Multi-label Emotion Understanding 

**Title (ZH)**: 评估大型语言模型在多标签情感理解方面的能力 

**Authors**: Tadesse Destaw Belay, Israel Abebe Azime, Abinew Ali Ayele, Grigori Sidorov, Dietrich Klakow, Philipp Slusallek, Olga Kolesnikova, Seid Muhie Yimam  

**Link**: [PDF](https://arxiv.org/pdf/2412.17837)  

**Abstract**: Large Language Models (LLMs) show promising learning and reasoning abilities. Compared to other NLP tasks, multilingual and multi-label emotion evaluation tasks are under-explored in LLMs. In this paper, we present EthioEmo, a multi-label emotion classification dataset for four Ethiopian languages, namely, Amharic (amh), Afan Oromo (orm), Somali (som), and Tigrinya (tir). We perform extensive experiments with an additional English multi-label emotion dataset from SemEval 2018 Task 1. Our evaluation includes encoder-only, encoder-decoder, and decoder-only language models. We compare zero and few-shot approaches of LLMs to fine-tuning smaller language models. The results show that accurate multi-label emotion classification is still insufficient even for high-resource languages such as English, and there is a large gap between the performance of high-resource and low-resource languages. The results also show varying performance levels depending on the language and model type. EthioEmo is available publicly to further improve the understanding of emotions in language models and how people convey emotions through various languages. 

**Abstract (ZH)**: 大规模语言模型（LLMs）展示了有前景的学习和推理能力。与其它NLP任务相比，跨语言和多标签情感评估任务在LLMs中的探索仍然不足。在本文中，我们提出了EthioEmo数据集，这是一个包含四门埃塞俄比亚语言的多标签情感分类数据集，具体来说，是阿姆哈拉语（amh）、阿法恩奥罗摩语（orm）、索马里语（som）和提格里尼亚语（tir）。我们还使用SemEval 2018 Task 1中的额外多标签情感分类的英语数据集进行了广泛的实验。我们的评估包括仅编码器、编码器-解码器和仅解码器语言模型。我们将大语言模型的零样本和少样本方法与微调较小的模型进行了比较。结果表明，即使对于高资源语言如英语，精确的多标签情感分类仍然不够充分，高资源和低资源语言之间的性能差距巨大。结果还显示，不同语言和模型类型下的表现水平也有所不同。EthioEmo数据集已经公开，旨在进一步提高对语言模型中情感理解以及人们通过各种语言传达情感方式的认识。 

---
# Look Ahead Text Understanding and LLM Stitching 

**Title (ZH)**: 展望文本理解与大规模语言模型整合 

**Authors**: Junlin Julian Jiang, Xin Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.17836)  

**Abstract**: This paper proposes a look ahead text understanding problem with look ahead section identification (LASI) as an example. This problem may appear in generative AI as well as human interactions, where we want to understand the direction of a developing text or conversation. We tackle the problem using transformer-based LLMs. We show that LASI is more challenging than classic section identification (SI). We argue that both bidirectional contextual information (e.g., BERT) and unidirectional predictive ability (e.g., GPT) will benefit the task. We propose two approaches to stitch together BERT and GPT. Experiments show that our approach outperforms the established models, especially when there is noise in the text (which is often the case for developing text in generative AI). Our paper sheds light on other look ahead text understanding tasks that are important to social media, such as look ahead sentiment classification, and points out the opportunities to leverage pre-trained LLMs through stitching. 

**Abstract (ZH)**: 本文以前瞻段落识别（LASI）为例，提出了一个前瞻文本理解问题。该问题可能出现在生成型AI和人际交往中，我们希望通过理解正在发展的文本或对话的方向来解决这一问题。我们使用基于变换器的大型语言模型（LLM）来解决这一问题。研究表明，LASI 比经典的段落识别（SI）更具挑战性。我们认为，双向上下文信息（如 BERT）和单向预测能力（如 GPT）都将有助于这一任务。我们提出了两种方法将BERT和GPT结合在一起。实验结果表明，我们的方法在文本存在噪声（在生成型AI中的发展文本通常存在这种情况）的情况下优于现有模型。我们的研究为社交媒体中的其他前瞻文本理解任务提供了新的视角，如前瞻情感分类，并指出了通过结合预训练的大型语言模型来利用这些机会的可能性。 

---
# Leveraging Sentiment for Offensive Text Classification 

**Title (ZH)**: 利用情感分析进行冒犯性文本分类 

**Authors**: Khondoker Ittehadul Islam  

**Link**: [PDF](https://arxiv.org/pdf/2412.17825)  

**Abstract**: In this paper, we conduct experiment to analyze whether models can classify offensive texts better with the help of sentiment. We conduct this experiment on the SemEval 2019 task 6, OLID, dataset. First, we utilize pre-trained language models to predict the sentiment of each instance. Later we pick the model that achieved the best performance on the OLID test set, and train it on the augmented OLID set to analyze the performance. Results show that utilizing sentiment increases the overall performance of the model. 

**Abstract (ZH)**: 在本文中，我们进行实验以分析情感分析是否有助于模型更好地分类具有攻击性的文本。我们在此实验中使用SemEval 2019 任务6 (OLID) 数据集。首先，我们利用预训练的语言模型来预测每个实例的情感。随后，我们选择在OLID测试集上表现最好的模型，并在扩增后的OLID数据集上对其进行训练以分析其性能。实验结果表明，利用情感分析可以提高模型的整体性能。 

---
# The Rosetta Paradox: Domain-Specific Performance Inversions in Large Language Models 

**Title (ZH)**: 罗塞塔悖论：大型语言模型中的领域特定性能反转 

**Authors**: Basab Jha, Ujjwal Puri  

**Link**: [PDF](https://arxiv.org/pdf/2412.17821)  

**Abstract**: While large language models, such as GPT and BERT, have already demonstrated unprecedented skills in everything from natural language processing to domain-specific applications, there came an unexplored phenomenon we term the Rosetta Paradox. The Rosetta Paradox characterizes the counterintuitive performance inversions across domains of knowledge. This paradox captures how such LLMs can excel in highly specialized fields but do poorly on tasks which require general, everyday knowledge. This paper formalizes the definition of the Rosetta Paradox and introduces a panoramic analysis framework that includes both a Domain Specificity Index (DSI) and a Performance Inversion Metric (PIM) for consistent quantification of domain-specific behavior in LLMs.
We adopt this paradox and conduct a series of investigations through extensive experiments across diverse models and knowledge domains, ranging from rich technical areas to common-sense reasoning. Our findings indicate that the Rosetta Paradox is likely not a mere artifact of data distribution but an intrinsic architectural and emergent property of deep neural networks. We present comparative analyses across different model architectures, sizes, and training methodologies that shed light into the peculiar ways this paradox manifests itself and challenge the standard evaluation metrics. 

**Abstract (ZH)**: 尽管诸如GPT和BERT这样的大规模语言模型已经在自然语言处理以及领域特定应用中展现了前所未有的技能，但其中出现了一种未被探索的现象，我们称之为罗塞塔悖论。罗塞塔悖论指出了知识领域间非直观的性能倒置现象。这种悖论揭示了这些语言模型在高度专业化领域表现出色，但在需要广泛日常知识的任务上则表现不佳。本文正式定义了罗塞塔悖论，并引入了一个全景分析框架，该框架包括领域特定性指数（DSI）和性能倒置度量（PIM），用于量化LLM在不同领域的特定行为。

我们采用了这一悖论，并通过广泛的实验，对多种模型和知识领域（从丰富的技术领域到常识推理）进行了深入研究。我们的研究结果表明，罗塞塔悖论很可能不是数据分布的偶然现象，而是一种深层次神经网络架构的固有和新兴属性。我们对不同模型架构、规模和训练方法进行了比较分析，揭示了这一悖论表现出来的独特方式，并挑战了传统的评估标准。 

---
# Inductive Linguistic Reasoning with Large Language Models 

**Title (ZH)**: 使用大规模语言模型进行归纳语言推理 

**Authors**: Raghav Ramji, Keshav Ramji  

**Link**: [PDF](https://arxiv.org/pdf/2412.17819)  

**Abstract**: Evaluating large language models (LLMs) on their linguistic reasoning capabilities is an important task to understand the gaps in their skills that may surface during large-scale adoption. In this work, we investigate the abilities of such models to perform abstract multilingual reasoning through the lens of linguistic puzzles on extremely low-resource languages. As these translation tasks involve inductive and deductive reasoning from reference instances, we examine whether diverse auxiliary demonstrations can be automatically induced from seed exemplars, through analogical prompting. We employ a two-stage procedure, first generating analogical exemplars with a language model, and then applying them in-context along with provided target language exemplars. Our results on the modeLing dataset show that analogical prompting is effective in eliciting models' knowledge of language grammar similarities, boosting the performance of GPT-4o by as much as 8.1% and Llama-3.1-405B-Instruct by 5.9% over chain-of-thought approaches. These gains are attributable to the analogical demonstrations, both when self-generated as well as when produced by weaker multilingual models. Furthermore, we demonstrate that our method generalizes to other tasks present in Linguistics Olympiad competitions, achieving sizable improvements across all problem types and difficulty levels included in the LINGOLY dataset with GPT-4o. We also report several findings about interesting phenomena which drive linguistic reasoning performance, suggesting that such puzzles are a valuable benchmark for new reasoning methods. 

**Abstract (ZH)**: 评估大型语言模型（LLMs）的语言推理能力是一项重要任务，有助于理解其在大规模应用中可能暴露出的能力差距。在本研究中，我们通过极端低资源语言的语言谜题，探讨这些模型在进行抽象多语言推理方面的能力。由于这些翻译任务涉及归纳和演绎推理，我们研究是否可以从种子示例中自动诱导出多样性的辅助示范，通过类比提示进行。我们采用两阶段的过程，首先使用语言模型生成类比示例，然后在提供目标语言示例的上下文中应用它们。我们在modeLing数据集上的结果显示，类比提示在激发模型对语言语法相似性的知识方面是有效的，提高了GPT-4o的性能8.1%，以及Llama-3.1-405B-Instruct的性能5.9%，超过了基于链式思考的方法。这些提升归因于无论是自动生成的还是由较弱的多语言模型生成的类比示范。此外，我们展示了本方法在其他包括在Linguistics Olympiad竞赛中的任务上的泛化能力，在LINGOLY数据集中的所有问题类型和难度级别上，GPT-4o均实现了显著的改进。我们还报告了一些关于影响语言推理性能有趣现象的研究发现，这些发现表明该类谜题是评估新推理方法的有价值的基准。 

---
# How Well Do LLMs Generate Code for Different Application Domains? Benchmark and Evaluation 

**Title (ZH)**: 不同应用领域中大语言模型生成代码的效果如何？基准测试与评估 

**Authors**: Dewu Zheng, Yanlin Wang, Ensheng Shi, Hongyu Zhang, Zibin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2412.18573)  

**Abstract**: Recently, an increasing number of AI-driven programming assistants powered by code LLMs have been integrated into various real-world software development environments, significantly boosting developer productivity. However, existing code generation benchmarks primarily focus on general-purpose scenarios, leaving the code generation performance of LLMs for specific application domains largely unknown. In this paper, we introduce a new benchmark, MultiCodeBench, to fill this gap. MultiCodeBench comprises 2,400 programming tasks, covering 12 popular software development domains and 15 programming languages. Specifically, we perform in-depth research to identify these 12 application domains. Given that each domain may involve multiple technical frameworks, and that different frameworks present distinct challenges in the coding process, we categorize the commonly used frameworks and platforms within each domain. We then sample programming problems from GitHub repositories related to these subdomains. To ensure the quality of the tasks and mitigate data leakage issues, we invite annotators to rewrite the docstrings for each task in MultiCodeBench. Additionally, we build a static analysis-based dependency parsing tool to extract the dependencies in the ground truth for each task, enabling deeper performance analysis. Through extensive experiments on MultiCodeBench with eleven representative mainstream LLMs, we reveal the code generation performance of the LLMs across different application domains, providing practical insights for developers in downstream fields when selecting LLMs. Furthermore, we analyze the reasons behind the models' failures in completing software application development tasks, offering guidance for model developers to enhance domain-specific code generation capabilities. 

**Abstract (ZH)**: 近年来，越来越多由代码LLM驱动的AI编程助手被集成到各种实际软件开发环境中，显著提升了开发者的生产力。然而，现有的代码生成基准主要关注通用场景，使得LLM在特定应用领域中的代码生成性能仍然知之甚少。在本文中，我们介绍了一个新的基准——MultiCodeBench，以填补这一空白。MultiCodeBench 包含2,400个编程任务，涵盖了12个流行的软件开发领域和15种编程语言。具体而言，我们深入研究了这些12个应用领域的选择。鉴于每个领域可能涉及多种技术框架，而不同框架在编码过程中会带来不同的挑战，我们对每个领域的常用框架和平台进行了分类。然后，我们从与这些子领域相关的GitHub仓库中抽取编程问题。为了确保任务的质量并减少数据泄露的问题，我们邀请注释者为MultiCodeBench中的每个任务重写文档字符串。此外，我们建立了一种基于静态分析的依赖关系解析工具，以提取每个任务的真实依赖关系，从而实现更深入的性能分析。通过在MultiCodeBench上对11种主流的代表LLM进行广泛的实验，我们揭示了LLM在不同应用领域的代码生成性能，为下游开发者在选择LLM时提供了实用的洞察。进一步地，我们分析了模型在完成软件应用开发任务时失败的原因，为模型开发者提供指导以增强领域特定的代码生成能力。 

---
# Consistency Checks for Language Model Forecasters 

**Title (ZH)**: 语言模型预测器的一致性检查 

**Authors**: Daniel Paleka, Abhimanyu Pallavi Sudhir, Alejandro Alvarez, Vineeth Bhat, Adam Shen, Evan Wang, Florian Tramèr  

**Link**: [PDF](https://arxiv.org/pdf/2412.18544)  

**Abstract**: Forecasting is a task that is difficult to evaluate: the ground truth can only be known in the future. Recent work showing LLM forecasters rapidly approaching human-level performance begs the question: how can we benchmark and evaluate these forecasters instantaneously? Following the consistency check framework, we measure the performance of forecasters in terms of the consistency of their predictions on different logically-related questions. We propose a new, general consistency metric based on arbitrage: for example, if a forecasting AI illogically predicts that both the Democratic and Republican parties have 60% probability of winning the 2024 US presidential election, an arbitrageur can trade against the forecaster's predictions and make a profit. We build an automated evaluation system that generates a set of base questions, instantiates consistency checks from these questions, elicits the predictions of the forecaster, and measures the consistency of the predictions. We then build a standard, proper-scoring-rule forecasting benchmark, and show that our (instantaneous) consistency metrics correlate with LLM forecasters' ground truth Brier scores (which are only known in the future). We also release a consistency benchmark that resolves in 2028, providing a long-term evaluation tool for forecasting. 

**Abstract (ZH)**: 预测是一项难以评估的任务：真实情况只有在未来才能得知。最近的研究表明，大型语言模型（LLM）预测器正在迅速接近人类水平的表现，这引发了另一个问题：我们如何能够即时地对这些预测器进行基准测试和评估？我们遵循一致性检验框架，衡量预测器在其对不同逻辑相关问题的预测中的一致性。我们提出了一种基于套利的新的一般一致性度量方法：例如，如果一个预测AI逻辑上错误地预测民主党与共和党在2024年美国总统大选中获胜的概率均为60%，那么套利者可以通过对预测的反向操作获利。我们构建了一个自动评估系统，该系统生成一组基础问题，基于这些问题实例化一致性检验，征求预测器的预测，并衡量这些预测的一致性。然后，我们构建了一个标准的、符合评分规则的预测基准，并展示了我们的一致性度量指标与LLM预测器的未来已知真实布雷尔评分（Brier scores）之间的相关性。此外，我们还发布了一个将在2028年揭晓的一致性基准，为预测提供了一个长期评估工具。 

---
# Characterizations of Language Generation With Breadth 

**Title (ZH)**: 语言生成的广度characterization 

**Authors**: Alkis Kalavasis, Anay Mehrotra, Grigoris Velegkas  

**Link**: [PDF](https://arxiv.org/pdf/2412.18530)  

**Abstract**: We study language generation in the limit, introduced by Kleinberg and Mullainathan [KM24], building on classical works of Gold [Gol67] and Angluin [Ang79]. [KM24] proposed an algorithm that generates strings from any countable language collection in the limit. While their algorithm eventually outputs strings from the target language $K$, it sacrifices breadth, i.e., the ability to generate all strings in $K$. A key open question in [KM24] is whether this trade-off between consistency and breadth is inherrent.
Recent works proposed different notions of consistent generation with breadth. Kalavasis, Mehrotra, and Velegkas [KVM24] introduced three definitions: generation with exact breadth, approximate breadth, and unambiguous generation. Concurrently and independently, Charikar and Pabbaraju [CP24a] proposed exhaustive generation. Both works examined when generation with these notions of breadth is possible.
Building on [CP24a, KVM24], we fully characterize language generation for these notions and their natural combinations. For exact breadth, we provide an unconditional lower bound, removing a technical condition from [KVM24] and extending the result of [CP24a] that holds for specific collections of languages. We show that generation with exact breadth is characterized by Angluin's condition for identification. We further introduce a weaker version of Angluin's condition that tightly characterizes both approximate breadth and exhaustive generation, proving their equivalence. Additionally, we show that unambiguous generation is also characterized by Angluin's condition as a special case of a broader result. Finally, we strengthen [KVM24] by giving unconditional lower bounds for stable generators, showing that Angluin's condition characterizes the previous breadth notions for stable generators. This shows a separation between stable and unstable generation with approximate breadth. 

**Abstract (ZH)**: 我们研究了Kleinberg和Mullainathan [KM24]引入的极限语言生成问题，该问题基于Gold [Gol67]和Angluin [Ang79]的经典工作。[KM24]提出了一种算法，能够从任何可数语言集合中生成字符串。虽然该算法最终会生成目标语言$K$中的字符串，但牺牲了广度，即生成$K$中所有字符串的能力。[KM24]中的一个关键开放问题是这种一致性和广度之间的权衡是否固有。

最近的一些研究提出了不同形式的广义一致生成概念。Kalavasis、Mehrotra和Velegkas [KVM24]引入了三种定义：精确广度生成、近似广度生成和非模糊生成。Charikar和Pabbaraju [CP24a]同时独立地提出了穷尽生成。这两项研究分别探讨了在这些广度概念下生成是可能的情况。

基于[CP24a, KVM24]的工作，我们全面刻画了这些广度概念及其自然组合下的语言生成问题。对于精确广度，我们提供了无条件的下限，去除了[KVM24]中的一项技术条件，并将[CP24a]的结论推广至特定语言集合。我们证明了精确广度生成由Angluin的识别条件完全定义。进一步地，我们引入了一个比Angluin条件更弱但能够紧密定义近似广度和穷尽生成的版本，证明了它们的等价性。此外，我们还展示了非模糊生成在一般结果的特殊情况下也由Angluin条件定义。最后，我们通过提供非模糊生成器的无条件下限加强了[KVM24]的工作，表明Angluin条件定义了所有先前的广度概念对非模糊生成器的情况。这展示了在近似广度下稳定生成与非稳定生成之间的分离。 

---
# Explainable Multi-Modal Data Exploration in Natural Language via LLM Agent 

**Title (ZH)**: 通过大规模语言模型代理进行可解释的多模态数据语言探索 

**Authors**: Farhad Nooralahzadeh, Yi Zhang, Jonathan Furst, Kurt Stockinger  

**Link**: [PDF](https://arxiv.org/pdf/2412.18428)  

**Abstract**: International enterprises, organizations, or hospitals collect large amounts of multi-modal data stored in databases, text documents, images, and videos. While there has been recent progress in the separate fields of multi-modal data exploration as well as in database systems that automatically translate natural language questions to database query languages, the research challenge of querying database systems combined with other unstructured modalities such as images in natural language is widely unexplored.
In this paper, we propose XMODE - a system that enables explainable, multi-modal data exploration in natural language. Our approach is based on the following research contributions: (1) Our system is inspired by a real-world use case that enables users to explore multi-modal information systems. (2) XMODE leverages a LLM-based agentic AI framework to decompose a natural language question into subtasks such as text-to-SQL generation and image analysis. (3) Experimental results on multi-modal datasets over relational data and images demonstrate that our system outperforms state-of-the-art multi-modal exploration systems, excelling not only in accuracy but also in various performance metrics such as query latency, API costs, planning efficiency, and explanation quality, thanks to the more effective utilization of the reasoning capabilities of LLMs. 

**Abstract (ZH)**: 国际企业、组织或医院收集了大量的多模态数据，这些数据存储在数据库、文本文件、图像和视频中。虽然在多模态数据探索和自动将自然语言问题转换为数据库查询语言的数据库系统方面已经取得了进展，但将数据库系统与其他未结构化的模态（如图像）结合起来使用自然语言进行查询的研究挑战仍未得到广泛探索。

本文提出了一种名为XMODE的系统，该系统能够使用自然语言进行可解释的多模态数据探索。我们的方法基于以下几个研究贡献：（1）我们的系统受到了实际应用场景的启发，使得用户能够探索多模态信息系统。（2）XMODE利用基于大模型的代理型AI框架，将自然语言问题分解为子任务，例如文本到SQL生成和图像分析。（3）在关系数据和图像的多模态数据集上的实验结果表明，我们的系统在准确性和查询延迟、API费用、规划效率和解释质量等多个性能指标上都明显优于现有的多模态探索系统，这得益于对大模型推理能力更有效的利用。 

---
# LongDocURL: a Comprehensive Multimodal Long Document Benchmark Integrating Understanding, Reasoning, and Locating 

**Title (ZH)**: 长文档URL：一个综合多模态长文档基准，涵盖理解、推理和定位能力 

**Authors**: Chao Deng, Jiale Yuan, Pi Bu, Peijie Wang, Zhong-Zhi Li, Jian Xu, Xiao-Hui Li, Yuan Gao, Jun Song, Bo Zheng, Cheng-Lin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.18424)  

**Abstract**: Large vision language models (LVLMs) have improved the document understanding capabilities remarkably, enabling the handling of complex document elements, longer contexts, and a wider range of tasks. However, existing document understanding benchmarks have been limited to handling only a small number of pages and fail to provide a comprehensive analysis of layout elements locating. In this paper, we first define three primary task categories: Long Document Understanding, numerical Reasoning, and cross-element Locating, and then propose a comprehensive benchmark, LongDocURL, integrating above three primary tasks and comprising 20 sub-tasks categorized based on different primary tasks and answer evidences. Furthermore, we develop a semi-automated construction pipeline and collect 2,325 high-quality question-answering pairs, covering more than 33,000 pages of documents, significantly outperforming existing benchmarks. Subsequently, we conduct comprehensive evaluation experiments on both open-source and closed-source models across 26 different configurations, revealing critical performance gaps in this field. 

**Abstract (ZH)**: 大视觉语言模型（LVLMs）显著提升了文档理解能力，使其能够处理复杂的文档元素、更长的上下文以及更广泛的任务。然而，现有的文档理解基准在处理多页文档时存在局限性，未能提供全面的布局元素定位分析。本文首先定义了三个主要任务类别：长文档理解、数值推理和跨元素定位，并提出了一个综合基准——LongDocURL，该基准集成了上述三个主要任务，包含20个基于不同主要任务和答案证据分类的子任务。此外，我们开发了一种半自动构建管道，收集了2,325个高质量的问题-答案对，涵盖了超过33,000页的文档，显著优于现有基准。随后，我们在来源开放和封闭的26种不同配置的模型上进行了全面的评估实验，揭示了该领域中关键的性能差距。 

---
# DeepCRCEval: Revisiting the Evaluation of Code Review Comment Generation 

**Title (ZH)**: DeepCRCEval：重新审视代码审查评论生成的评估方法 

**Authors**: Junyi Lu, Xiaojia Li, Zihan Hua, Lei Yu, Shiqi Cheng, Li Yang, Fengjun Zhang, Chun Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2412.18291)  

**Abstract**: Code review is a vital but demanding aspect of software development, generating significant interest in automating review comments. Traditional evaluation methods for these comments, primarily based on text similarity, face two major challenges: inconsistent reliability of human-authored comments in open-source projects and the weak correlation of text similarity with objectives like enhancing code quality and detecting defects.
This study empirically analyzes benchmark comments using a novel set of criteria informed by prior research and developer interviews. We then similarly revisit the evaluation of existing methodologies. Our evaluation framework, DeepCRCEval, integrates human evaluators and Large Language Models (LLMs) for a comprehensive reassessment of current techniques based on the criteria set. Besides, we also introduce an innovative and efficient baseline, LLM-Reviewer, leveraging the few-shot learning capabilities of LLMs for a target-oriented comparison.
Our research highlights the limitations of text similarity metrics, finding that less than 10% of benchmark comments are high quality for automation. In contrast, DeepCRCEval effectively distinguishes between high and low-quality comments, proving to be a more reliable evaluation mechanism. Incorporating LLM evaluators into DeepCRCEval significantly boosts efficiency, reducing time and cost by 88.78% and 90.32%, respectively. Furthermore, LLM-Reviewer demonstrates significant potential of focusing task real targets in comment generation. 

**Abstract (ZH)**: 代码审查是软件开发中一个重要但富有挑战性的环节，引起了对自动化审查评论的广泛关注。传统对这些评论的评估方法，主要基于文本相似性，面临两大主要挑战：开源项目中人工编写的评论的一致性和可靠性不足，以及文本相似性与提升代码质量、检测缺陷等目标之间的弱关联性。

本研究通过采用一套新的评价标准，结合先前研究和开发者访谈的结果进行实证分析，重新评估现有的评估方法。我们的评估框架DeepCRCEval结合了人力评估者和大型语言模型（LLM），基于提出的标准对现有技术进行全面的重新评估。此外，我们还引入了一种创新且高效的基线模型LLM-Reviewer，利用LLM的少量示例学习能力进行目标导向的比较。

我们的研究揭示了文本相似性度量的局限性，发现基准评论中仅有不到10%的评论适合自动化。相比之下，DeepCRCEval能够有效地区分高质量和低质量的评论，证明了其作为更可靠的评估机制的有效性。将LLM评估者纳入DeepCRCEval显著提高了效率，分别降低了88.78%的时间和90.32%的成本。此外，LLM-Reviewer展示了在评论生成中集中目标任务的巨大潜力。 

---
# ICM-Assistant: Instruction-tuning Multimodal Large Language Models for Rule-based Explainable Image Content Moderation 

**Title (ZH)**: ICM-Assistant：基于指令调优的多模态大型语言模型在基于规则的可解释图像内容审核中的应用 

**Authors**: Mengyang Wu, Yuzhi Zhao, Jialun Cao, Mingjie Xu, Zhongming Jiang, Xuehui Wang, Qinbin Li, Guangneng Hu, Shengchao Qin, Chi-Wing Fu  

**Link**: [PDF](https://arxiv.org/pdf/2412.18216)  

**Abstract**: Controversial contents largely inundate the Internet, infringing various cultural norms and child protection standards. Traditional Image Content Moderation (ICM) models fall short in producing precise moderation decisions for diverse standards, while recent multimodal large language models (MLLMs), when adopted to general rule-based ICM, often produce classification and explanation results that are inconsistent with human moderators. Aiming at flexible, explainable, and accurate ICM, we design a novel rule-based dataset generation pipeline, decomposing concise human-defined rules and leveraging well-designed multi-stage prompts to enrich short explicit image annotations. Our ICM-Instruct dataset includes detailed moderation explanation and moderation Q-A pairs. Built upon it, we create our ICM-Assistant model in the framework of rule-based ICM, making it readily applicable in real practice. Our ICM-Assistant model demonstrates exceptional performance and flexibility. Specifically, it significantly outperforms existing approaches on various sources, improving both the moderation classification (36.8\% on average) and moderation explanation quality (26.6\% on average) consistently over existing MLLMs. Code/Data is available at this https URL. 

**Abstract (ZH)**: 互联网上大量的争议性内容侵犯了各种文化规范和儿童保护标准。传统的图像内容审核（ICM）模型在针对多样化的标准时无法做出精确的审核决策，而最近的多模态大型语言模型（MLLMs），当应用于通用规则基础的ICM时，往往会产出与人工审核者不一致的分类和解释结果。为实现灵活、可解释且准确的ICM，我们设计了一种新颖的基于规则的数据集生成流水线，将简洁的人工定义规则分解，并利用精心设计的多阶段提示来丰富短显式的图像标注。我们的ICM-Instruct数据集包括详细的审核解释和审核问答对。在此基础上，我们在基于规则的ICM框架中构建了ICM-Assistant模型，使其在实际应用中易于实施。ICM-Assistant模型展示了出色的表现和灵活性。具体而言，它在多种来源上显著优于现有方法，一致地提高了审核分类（平均36.8%）和审核解释质量（平均26.6%），并且优于现有的MLLMs。相关代码/数据可在以下链接获取：this https URL。 

---
# VLABench: A Large-Scale Benchmark for Language-Conditioned Robotics Manipulation with Long-Horizon Reasoning Tasks 

**Title (ZH)**: VLABench：一种用于长时序推理任务的基于语言条件的机器人操纵大规模基准测试 

**Authors**: Shiduo Zhang, Zhe Xu, Peiju Liu, Xiaopeng Yu, Yuan Li, Qinghui Gao, Zhaoye Fei, Zhangyue Yin, Zuxuan Wu, Yu-Gang Jiang, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2412.18194)  

**Abstract**: General-purposed embodied agents are designed to understand the users' natural instructions or intentions and act precisely to complete universal tasks. Recently, methods based on foundation models especially Vision-Language-Action models (VLAs) have shown a substantial potential to solve language-conditioned manipulation (LCM) tasks well. However, existing benchmarks do not adequately meet the needs of VLAs and relative algorithms. To better define such general-purpose tasks in the context of LLMs and advance the research in VLAs, we present VLABench, an open-source benchmark for evaluating universal LCM task learning. VLABench provides 100 carefully designed categories of tasks, with strong randomization in each category of task and a total of 2000+ objects. VLABench stands out from previous benchmarks in four key aspects: 1) tasks requiring world knowledge and common sense transfer, 2) natural language instructions with implicit human intentions rather than templates, 3) long-horizon tasks demanding multi-step reasoning, and 4) evaluation of both action policies and language model capabilities. The benchmark assesses multiple competencies including understanding of mesh\&texture, spatial relationship, semantic instruction, physical laws, knowledge transfer and reasoning, etc. To support the downstream finetuning, we provide high-quality training data collected via an automated framework incorporating heuristic skills and prior information. The experimental results indicate that both the current state-of-the-art pretrained VLAs and the workflow based on VLMs face challenges in our tasks. 

**Abstract (ZH)**: 通用型躯体化代理旨在理解和执行用户的自然指令或意图，以精确地完成通用任务。近期，基于基础模型的方法，尤其是视觉-语言-动作模型（VLAs），在解决语言条件下的操作任务（LCM）方面展现了巨大的潜力。然而，现有的基准测试并没有充分满足VLAs及其相关算法的需求。为更好地在大语言模型（LLMs）的背景下定义这类通用任务，并推进VLAs的研究，我们提出了VLABench，这是一个开源基准测试，用于评估通用LCM任务的学习。VLABench 提供了100个精心设计的任务类别，每个类别都有较强的随机化，并包含了超过2000个对象。与之前的基准测试相比，VLABench 在四个方面脱颖而出：1) 要求世界知识和常识的转移；2) 自然语言指令中隐含着人类意图，而非模板；3) 需要多步推理的长期任务；4) 评估动作策略和语言模型能力。基准测试评估了多个能力，包括网格与纹理的理解、空间关系、语义指令、物理定律、知识转移和推理等。为了支持下游微调，我们提供了一套高质量的训练数据，这些数据是通过结合启发式技能和先验信息的自动化框架收集的。实验结果表明，当前最先进的预训练VLAs和基于VLMs的工作流程在我们的任务中都面临着挑战。 

---
# scReader: Prompting Large Language Models to Interpret scRNA-seq Data 

**Title (ZH)**: scReader：引导大型语言模型解释单细胞RNA测序数据 

**Authors**: Cong Li, Qingqing Long, Yuanchun Zhou, Meng Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2412.18156)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable advancements, primarily due to their capabilities in modeling the hidden relationships within text sequences. This innovation presents a unique opportunity in the field of life sciences, where vast collections of single-cell omics data from multiple species provide a foundation for training foundational models. However, the challenge lies in the disparity of data scales across different species, hindering the development of a comprehensive model for interpreting genetic data across diverse organisms. In this study, we propose an innovative hybrid approach that integrates the general knowledge capabilities of LLMs with domain-specific representation models for single-cell omics data interpretation. We begin by focusing on genes as the fundamental unit of representation. Gene representations are initialized using functional descriptions, leveraging the strengths of mature language models such as LLaMA-2. By inputting single-cell gene-level expression data with prompts, we effectively model cellular representations based on the differential expression levels of genes across various species and cell types. In the experiments, we constructed developmental cells from humans and mice, specifically targeting cells that are challenging to annotate. We evaluated our methodology through basic tasks such as cell annotation and visualization analysis. The results demonstrate the efficacy of our approach compared to other methods using LLMs, highlighting significant improvements in accuracy and interoperability. Our hybrid approach enhances the representation of single-cell data and offers a robust framework for future research in cross-species genetic analysis. 

**Abstract (ZH)**: 大型语言模型（LLMs）在模拟文本序列中隐藏的关系方面取得了显著的进步。这种创新为生命科学领域带来了独特的机遇，其中来自多个物种的大量单细胞组学数据为训练基础模型提供了基础。然而，不同物种间数据规模的差异阻碍了跨物种遗传数据解释的全面模型的发展。在此研究中，我们提出了一种创新的融合方法，将LLMs的一般知识能力与针对单细胞组学数据的领域特定表示模型相结合，用于解释基因数据。我们首先以基因作为表示的基本单元进行研究。基因表示利用功能描述初始化，利用成熟语言模型如LLaMA-2的优势。通过输入单细胞基因水平表达数据并结合提示，我们有效构建了基于不同物种和细胞类型之间基因差异表达水平的细胞表示。在实验中，我们构建了来自人类和小鼠的发育细胞，并专门针对那些难以标注的细胞进行研究。我们通过基本任务如细胞注释和可视化分析评估了该方法。结果表明，与使用LLMs的其他方法相比，我们的方法具有更高的准确性和互操作性。我们的融合方法提高了单细胞数据的表示能力，并为跨物种遗传分析提供了稳健的框架。 

---
# GeneSUM: Large Language Model-based Gene Summary Extraction 

**Title (ZH)**: GeneSUM：基于大型语言模型的基因摘要提取 

**Authors**: Zhijian Chen, Chuan Hu, Min Wu, Qingqing Long, Xuezhi Wang, Yuanchun Zhou, Meng Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2412.18154)  

**Abstract**: Emerging topics in biomedical research are continuously expanding, providing a wealth of information about genes and their function. This rapid proliferation of knowledge presents unprecedented opportunities for scientific discovery and formidable challenges for researchers striving to keep abreast of the latest advancements. One significant challenge is navigating the vast corpus of literature to extract vital gene-related information, a time-consuming and cumbersome task. To enhance the efficiency of this process, it is crucial to address several key challenges: (1) the overwhelming volume of literature, (2) the complexity of gene functions, and (3) the automated integration and generation. In response, we propose GeneSUM, a two-stage automated gene summary extractor utilizing a large language model (LLM). Our approach retrieves and eliminates redundancy of target gene literature and then fine-tunes the LLM to refine and streamline the summarization process. We conducted extensive experiments to validate the efficacy of our proposed framework. The results demonstrate that LLM significantly enhances the integration of gene-specific information, allowing more efficient decision-making in ongoing research. 

**Abstract (ZH)**: 生物医学研究中的新兴领域正在不断发展，为基因及其功能提供了丰富的信息。这种知识的快速增长为科学发现提供了前所未有的机遇，同时也给研究人员带来了巨大挑战，他们需要努力跟上最新的进展。一个重要的挑战是如何有效导航大量的文献，提取关键的基因相关信息，这是一项耗时且繁琐的工作。为了提高这一过程的效率，必须解决几个关键问题：(1) 文献的庞大数量，(2) 基因功能的复杂性，以及(3) 自动化的整合与生成能力。为此，我们提出了一种名为GeneSUM的两阶段自动化基因摘要提取器，该工具利用大型语言模型（LLM）。我们的方法首先检索并去除目标基因文献中的冗余信息，然后通过微调LLM来优化和简化摘要生成过程。我们进行了大量实验来验证我们所提出的框架的有效性。结果表明，大型语言模型显著增强了基因特定信息的整合能力，使研究人员在当前研究中能够更高效地作出决策。 

---
# Are We in the AI-Generated Text World Already? Quantifying and Monitoring AIGT on Social Media 

**Title (ZH)**: 我们已经进入了由AI生成文本的世界了吗？量化和监控社交媒体上的AI生成文本 

**Authors**: Zhen Sun, Zongmin Zhang, Xinyue Shen, Ziyi Zhang, Yule Liu, Michael Backes, Yang Zhang, Xinlei He  

**Link**: [PDF](https://arxiv.org/pdf/2412.18148)  

**Abstract**: Social media platforms are experiencing a growing presence of AI-Generated Texts (AIGTs). However, the misuse of AIGTs could have profound implications for public opinion, such as spreading misinformation and manipulating narratives. Despite its importance, a systematic study to assess the prevalence of AIGTs on social media is still lacking. To address this gap, this paper aims to quantify, monitor, and analyze the AIGTs on online social media platforms. We first collect a dataset (SM-D) with around 2.4M posts from 3 major social media platforms: Medium, Quora, and Reddit. Then, we construct a diverse dataset (AIGTBench) to train and evaluate AIGT detectors. AIGTBench combines popular open-source datasets and our AIGT datasets generated from social media texts by 12 LLMs, serving as a benchmark for evaluating mainstream detectors. With this setup, we identify the best-performing detector (OSM-Det). We then apply OSM-Det to SM-D to track AIGTs over time and observe different trends of AI Attribution Rate (AAR) across social media platforms from January 2022 to October 2024. Specifically, Medium and Quora exhibit marked increases in AAR, rising from 1.77% to 37.03% and 2.06% to 38.95%, respectively. In contrast, Reddit shows slower growth, with AAR increasing from 1.31% to 2.45% over the same period. Our further analysis indicates that AIGTs differ from human-written texts across several dimensions, including linguistic patterns, topic distributions, engagement levels, and the follower distribution of authors. We envision our analysis and findings on AIGTs in social media can shed light on future research in this domain. 

**Abstract (ZH)**: 社交媒体平台上的人工智能生成文本（AIGTs）正经历快速增长。然而，AIGTs 的不当使用可能对公共舆论产生深远影响，例如传播错误信息和操控叙事。尽管其重要性不言而喻，但对社交媒体上AIGTs 的系统性研究仍然缺乏。为填补这一空白，本文旨在量化、监控并分析社交媒体平台上的AIGTs。首先，我们从Medium、Quora和Reddit三大社交媒体平台收集了一个包含约240万条帖子的数据集（SM-D）。然后，我们构建了一个多样化的数据集（AIGTBench）来训练和评估AIGT检测器。AIGTBench结合了流行的开源数据集和从社交媒体文本生成的、由12个语言模型（LLMs）创建的AIGT数据集，作为主流检测器的基准。

借助这一设置，我们识别出表现最佳的检测器（OSM-Det）。随后，我们应用OSM-Det到SM-D数据集中，以监测随着时间推移AIGTs的变化，并观察从2022年1月到2024年10月期间不同社交媒体平台上的AI归属率（AAR）趋势。具体而言，Medium和Quora的AAR显著上升，分别从1.77%升至37.03%和从2.06%升至38.95%。相比之下，Reddit的增长速度较慢，AAR在同一时期内从1.31%升至2.45%。进一步分析表明，AIGTs在语言模式、主题分布、互动水平以及作者粉丝分布等多个维度上与人类撰写的文本存在差异。我们希望我们的分析和对社交媒体上AIGTs 的研究能够启发未来在这个领域内的研究工作。 

---
# AEIOU: A Unified Defense Framework against NSFW Prompts in Text-to-Image Models 

**Title (ZH)**: AEIOU：一种针对文本到图像模型中不适当提示的统一防御框架 

**Authors**: Yiming Wang, Jiahao Chen, Qingming Li, Xing Yang, Shouling Ji  

**Link**: [PDF](https://arxiv.org/pdf/2412.18123)  

**Abstract**: As text-to-image (T2I) models continue to advance and gain widespread adoption, their associated safety issues are becoming increasingly prominent. Malicious users often exploit these models to generate Not-Safe-for-Work (NSFW) images using harmful or adversarial prompts, highlighting the critical need for robust safeguards to ensure the integrity and compliance of model outputs. Current internal safeguards frequently degrade image quality, while external detection methods often suffer from low accuracy and inefficiency.
In this paper, we introduce AEIOU, a defense framework that is Adaptable, Efficient, Interpretable, Optimizable, and Unified against NSFW prompts in T2I models. AEIOU extracts NSFW features from the hidden states of the model's text encoder, utilizing the separable nature of these features to detect NSFW prompts. The detection process is efficient, requiring minimal inference time. AEIOU also offers real-time interpretation of results and supports optimization through data augmentation techniques. The framework is versatile, accommodating various T2I architectures. Our extensive experiments show that AEIOU significantly outperforms both commercial and open-source moderation tools, achieving over 95% accuracy across all datasets and improving efficiency by at least tenfold. It effectively counters adaptive attacks and excels in few-shot and multi-label scenarios. 

**Abstract (ZH)**: 随着文本-to-图像（T2I）模型不断发展并广泛采用，它们所伴随的安全问题变得越来越显著。恶意用户经常利用这些模型通过有害或对抗性的提示生成不适合工作环境（NSFW）的图像，这凸显了必须实施强大保护措施以确保模型输出的完整性和合规性的迫切性。当前的内部保护措施往往会降低图像质量，而外部检测方法往往存在准确率低和效率低的问题。

在本文中，我们引入了AEIOU，这是一种针对T2I模型中的NSFW提示具有可适应性、高效性、可解释性、优化性和统一性的防御框架。AEIOU从模型文本编码器的隐藏状态中提取NSFW特征，并利用这些特征的分离性质来检测NSFW提示。检测过程高效且几乎不需要额外的推理时间。AEIOU还提供实时结果解释，并支持通过数据增强技术优化。该框架具有较强的通用性，可以适用于各种T2I架构。我们广泛的实验结果显示，AEIOU在所有数据集上的准确率均超过95%，并且其效率至少提高了十倍。它有效地抵御了适应性攻击，并在少量样本和多标签场景中表现出色。 

---
# Generating Traffic Scenarios via In-Context Learning to Learn Better Motion Planner 

**Title (ZH)**: 通过情境学习生成交通场景以改善运动规划器性能 

**Authors**: Aizierjiang Aiersilan  

**Link**: [PDF](https://arxiv.org/pdf/2412.18086)  

**Abstract**: Motion planning is a crucial component in autonomous driving. State-of-the-art motion planners are trained on meticulously curated datasets, which are not only expensive to annotate but also insufficient in capturing rarely seen critical scenarios. Failing to account for such scenarios poses a significant risk to motion planners and may lead to incidents during testing. An intuitive solution is to manually compose such scenarios by programming and executing a simulator (e.g., CARLA). However, this approach incurs substantial human costs. Motivated by this, we propose an inexpensive method for generating diverse critical traffic scenarios to train more robust motion planners. First, we represent traffic scenarios as scripts, which are then used by the simulator to generate traffic scenarios. Next, we develop a method that accepts user-specified text descriptions, which a Large Language Model (LLM) translates into scripts using in-context learning. The output scripts are sent to the simulator that produces the corresponding traffic scenarios. As our method can generate abundant safety-critical traffic scenarios, we use them as synthetic training data for motion planners. To demonstrate the value of generated scenarios, we train existing motion planners on our synthetic data, real-world datasets, and a combination of both. Our experiments show that motion planners trained with our data significantly outperform those trained solely on real-world data, showing the usefulness of our synthetic data and the effectiveness of our data generation method. Our source code is available at this https URL. 

**Abstract (ZH)**: 自主驾驶中的运动规划是其关键组成部分。最新的运动规划器在精心策划的数据集上进行训练，这些数据集不仅标注成本高昂，而且难以涵盖罕见但至关重要的应用场景。未能考虑这些场景会显著增加运动规划器的风险，并可能在测试过程中引发事故。一个直观的解决方案是通过编程和执行模拟器（例如CARLA）手动组合这些场景。然而，这种方法会带来显著的人工成本。受到这一问题的启发，我们提出了一种低成本的方法，用于生成多样化的关键交通场景，以训练更稳健的运动规划器。首先，我们将交通场景表示为脚本，然后将这些脚本用于模拟器生成交通场景。接着，我们开发了一种方法，该方法接受用户指定的文本描述，由大型语言模型（LLM）通过上下文学习将其转化为脚本。生成的脚本随后发送给模拟器，以生成相应的交通场景。由于我们的方法能够生成大量的安全关键交通场景，我们将其用作运动规划器的合成训练数据。为了展示生成场景的价值，我们使用了我们的合成数据、真实世界数据，以及两者的组合来训练现有的运动规划器。实验结果显示，使用我们数据训练的运动规划器在性能上显著优于仅使用真实世界数据进行训练的运动规划器，这表明了我们合成数据的有效性以及生成方法的有效性。我们的源代码可以在以下网址获取：这个 https URL。 

---
# MMFactory: A Universal Solution Search Engine for Vision-Language Tasks 

**Title (ZH)**: MMFactory：视觉-语言任务的通用解决方案搜索引擎 

**Authors**: Wan-Cyuan Fan, Tanzila Rahman, Leonid Sigal  

**Link**: [PDF](https://arxiv.org/pdf/2412.18072)  

**Abstract**: With advances in foundational and vision-language models, and effective fine-tuning techniques, a large number of both general and special-purpose models have been developed for a variety of visual tasks. Despite the flexibility and accessibility of these models, no single model is able to handle all tasks and/or applications that may be envisioned by potential users. Recent approaches, such as visual programming and multimodal LLMs with integrated tools aim to tackle complex visual tasks, by way of program synthesis. However, such approaches overlook user constraints (e.g., performance / computational needs), produce test-time sample-specific solutions that are difficult to deploy, and, sometimes, require low-level instructions that maybe beyond the abilities of a naive user. To address these limitations, we introduce MMFactory, a universal framework that includes model and metrics routing components, acting like a solution search engine across various available models. Based on a task description and few sample input-output pairs and (optionally) resource and/or performance constraints, MMFactory can suggest a diverse pool of programmatic solutions by instantiating and combining visio-lingual tools from its model repository. In addition to synthesizing these solutions, MMFactory also proposes metrics and benchmarks performance / resource characteristics, allowing users to pick a solution that meets their unique design constraints. From the technical perspective, we also introduced a committee-based solution proposer that leverages multi-agent LLM conversation to generate executable, diverse, universal, and robust solutions for the user. Experimental results show that MMFactory outperforms existing methods by delivering state-of-the-art solutions tailored to user problem specifications. Project page is available at this https URL. 

**Abstract (ZH)**: 随着基础模型和视觉语言模型的发展，以及有效微调技术的应用，已经开发出大量适用于多种视觉任务的通用和专用模型。尽管这些模型具有灵活性和易用性，但没有单一模型能够应对所有可能被潜在用户设想的任务和/或应用。最近的方法，如视觉编程和集成工具的多模态大语言模型（multimodal LLMs），通过程序合成旨在解决复杂视觉任务。然而，这些方法忽略了用户的约束条件（例如，性能/计算需求），生成了在测试时针对特定样本的解决方案，这些解决方案难以部署，并且有时需要低级别的指令，这可能超出普通用户的操作能力。为了克服这些限制，我们引入了MMFactory，这是一个通用框架，包括模型路由和度量路由组件，充当在各种可用模型中搜索解决方案的引擎。基于任务描述、少数样本输入输出对以及（可选地）资源和/或性能限制，MMFactory可以建议通过实例化和组合其模型库中的视觉语言工具来实现多样化的程序解决方案。除了合成这些解决方案外，MMFactory还建议适当的度量标准并评估性能/资源特性，使用户能够选择符合其特定设计约束的解决方案。从技术角度来看，我们还引入了一种基于多智能体大语言模型对话的委员会解决方案提名人，以生成可供执行的、多样化的、通用的和稳健的解决方案。实验结果表明，MMFactory通过交付符合用户问题规格的领先解决方案，超越了现有方法。项目页面可通过以下链接访问：this https URL。 

---
# Lla-VAP: LSTM Ensemble of Llama and VAP for Turn-Taking Prediction 

**Title (ZH)**: Lla-VAP：基于Llama和VAP的LSTM集成模型用于交互转换预测 

**Authors**: Hyunbae Jeon, Frederic Guintu, Rayvant Sahni  

**Link**: [PDF](https://arxiv.org/pdf/2412.18061)  

**Abstract**: Turn-taking prediction is the task of anticipating when the speaker in a conversation will yield their turn to another speaker to begin speaking. This project expands on existing strategies for turn-taking prediction by employing a multi-modal ensemble approach that integrates large language models (LLMs) and voice activity projection (VAP) models. By combining the linguistic capabilities of LLMs with the temporal precision of VAP models, we aim to improve the accuracy and efficiency of identifying TRPs in both scripted and unscripted conversational scenarios. Our methods are evaluated on the In-Conversation Corpus (ICC) and Coached Conversational Preference Elicitation (CCPE) datasets, highlighting the strengths and limitations of current models while proposing a potentially more robust framework for enhanced prediction. 

**Abstract (ZH)**: 轮流发言预测是预测对话中的说话人何时会将发言权让给其他说话人的任务。本项目在此基础上扩展了现有的轮流发言预测策略，采用了多模态集成方法，结合了大型语言模型（LLMs）和语音活动投影（VAP）模型。通过将LLMs的语言能力与VAP模型的时间精度相结合，我们旨在在既定和非既定对话场景中改善TRP（Turn-Right-Part）识别的准确性和效率。我们的方法在In-Conversation Corpus (ICC) 和 Coached Conversational Preference Elicitation (CCPE) 数据集上进行了评估，突显了当前模型的优势和局限性，并提出了一种可能更稳健的框架以增强预测能力。 

---
# Emoji Retrieval from Gibberish or Garbled Social Media Text: A Novel Methodology and A Case Study 

**Title (ZH)**: 从乱码或 garbled 社交媒体文本中提取表情符号：一种新颖的方法论和案例研究

注释：在这里，“gibberish”一词通常指的是无法理解的文字或乱码。在翻译时，根据上下文，我将其译为“乱码”。如果您有更具体的技术术语偏好，也可以告知我调整。 

**Authors**: Shuqi Cui, Nirmalya Thakur, Audrey Poon  

**Link**: [PDF](https://arxiv.org/pdf/2412.18046)  

**Abstract**: Emojis are widely used across social media platforms but are often lost in noisy or garbled text, posing challenges for data analysis and machine learning. Conventional preprocessing approaches recommend removing such text, risking the loss of emojis and their contextual meaning. This paper proposes a three-step reverse-engineering methodology to retrieve emojis from garbled text in social media posts. The methodology also identifies reasons for the generation of such text during social media data mining. To evaluate its effectiveness, the approach was applied to 509,248 Tweets about the Mpox outbreak, a dataset referenced in about 30 prior works that failed to retrieve emojis from garbled text. Our method retrieved 157,748 emojis from 76,914 Tweets. Improvements in text readability and coherence were demonstrated through metrics such as Flesch Reading Ease, Flesch-Kincaid Grade Level, Coleman-Liau Index, Automated Readability Index, Dale-Chall Readability Score, Text Standard, and Reading Time. Additionally, the frequency of individual emojis and their patterns of usage in these Tweets were analyzed, and the results are presented. 

**Abstract (ZH)**: 表情符号在社交媒体平台中广泛使用，但在嘈杂或错误拼写的文本中往往会丢失，这给数据解析和机器学习带来了挑战。传统预处理方法建议移除这类文本，从而可能会导致表情符号及其上下文意义的丢失。本文提出了一种三步逆向工程方法，用于从社交媒体帖子中的错误拼写文本中检索表情符号。该方法还识别了在社交媒体数据挖掘过程中产生此类文本的原因。为了验证其有效性，该方法应用于包含509,248条关于猴痘爆发的推文的数据集，该数据集在大约30篇先前的研究中被引用，但这些研究未能从错误拼写的文本中检索出表情符号。我们的方法成功从76,914条推文中检索出了157,748个表情符号。通过诸如弗利舍可读性分数、弗利舍-金凯德阅读难度等级、科尔曼-利奥利指数、自动可读性指数、戴尔-查尔可读性分数、文本标准和阅读时间等指标，展示了文本的可读性和连贯性的增强。此外，还分析了这些推文中个别表情符号的频率及其使用模式，并将结果进行了呈现。 

---
# Theoretical Constraints on the Expressive Power of $\mathsf{RoPE}$-based Tensor Attention Transformers 

**Title (ZH)**: 基于RoPE的张量注意力变换器的表达能力的理论约束 

**Authors**: Xiaoyu Li, Yingyu Liang, Zhenmei Shi, Zhao Song, Mingda Wan  

**Link**: [PDF](https://arxiv.org/pdf/2412.18040)  

**Abstract**: Tensor Attention extends traditional attention mechanisms by capturing high-order correlations across multiple modalities, addressing the limitations of classical matrix-based attention. Meanwhile, Rotary Position Embedding ($\mathsf{RoPE}$) has shown superior performance in encoding positional information in long-context scenarios, significantly enhancing transformer models' expressiveness. Despite these empirical successes, the theoretical limitations of these technologies remain underexplored. In this study, we analyze the circuit complexity of Tensor Attention and $\mathsf{RoPE}$-based Tensor Attention, showing that with polynomial precision, constant-depth layers, and linear or sublinear hidden dimension, they cannot solve fixed membership problems or $(A_{F,r})^*$ closure problems, under the assumption that $\mathsf{TC}^0 \neq \mathsf{NC}^1$. These findings highlight a gap between the empirical performance and theoretical constraints of Tensor Attention and $\mathsf{RoPE}$-based Tensor Attention Transformers, offering insights that could guide the development of more theoretically grounded approaches to Transformer model design and scaling. 

**Abstract (ZH)**: 本文将下面的论文内容或标题翻译成中文，并确保符合学术规范：

张量注意机制通过在多种模态间捕捉高阶相关性，扩展了传统注意机制，解决了经典矩阵基注意机制的局限性。同时，旋转型位置嵌入（$\mathsf{RoPE}$）已经在长上下文场景中展示了优越的性能，在编码位置信息方面显著增强了Transformer模型的表达能力。尽管这些技术在实际应用中取得了成功，但它们的理论局限性仍然有待进一步探索。本研究分析了张量注意机制和基于$\mathsf{RoPE}$的张量注意机制的电路复杂度，表明在多项式精度、常深度层以及线性或亚线性隐藏维数的假设下，它们无法解决固定成员问题或$(A_{F,r})^*$闭包问题，前提是假设$\mathsf{TC}^0 \neq \mathsf{NC}^1$。这些发现揭示了张量注意机制和基于$\mathsf{RoPE}$的张量注意机制Transformer在实际性能和理论约束之间的差距，为指导更具有理论基础的Transformer模型设计和扩展提供了见解。

具体翻译如下：
- Tensor Attention通过在多种模态间捕捉高阶相关性，扩展了传统注意机制，解决了经典矩阵基注意机制的局限性。
- Rotary Position Embedding（$\mathsf{RoPE}$）已经在长上下文场景中展示了优越的性能，在编码位置信息方面显著增强了Transformer模型的表达能力。
- 尽管这些技术在实际应用中取得了成功，但它们的理论局限性仍然有待进一步探索。
- 在本研究中，我们分析了Tensor Attention和基于$\mathsf{RoPE}$的Tensor Attention的电路复杂度，表明在多项式精度、常深度层以及线性或亚线性隐藏维数的假设下，它们无法解决固定成员问题或$(A_{F,r})^*$闭包问题，前提是假设$\mathsf{TC}^0 \neq \mathsf{NC}^1$。
- 这些发现揭示了Tensor Attention和基于$\mathsf{RoPE}$的Tensor Attention Transformer在实际性能和理论约束之间的差距，为指导更具有理论基础的Transformer模型设计和扩展提供了见解。 

---
# VITRO: Vocabulary Inversion for Time-series Representation Optimization 

**Title (ZH)**: VITRO: 时间序列表示优化中的词汇反转 

**Authors**: Filippos Bellos, Nam H. Nguyen, Jason J. Corso  

**Link**: [PDF](https://arxiv.org/pdf/2412.17921)  

**Abstract**: Although LLMs have demonstrated remarkable capabilities in processing and generating textual data, their pre-trained vocabularies are ill-suited for capturing the nuanced temporal dynamics and patterns inherent in time series. The discrete, symbolic nature of natural language tokens, which these vocabularies are designed to represent, does not align well with the continuous, numerical nature of time series data. To address this fundamental limitation, we propose VITRO. Our method adapts textual inversion optimization from the vision-language domain in order to learn a new time series per-dataset vocabulary that bridges the gap between the discrete, semantic nature of natural language and the continuous, numerical nature of time series data. We show that learnable time series-specific pseudo-word embeddings represent time series data better than existing general language model vocabularies, with VITRO-enhanced methods achieving state-of-the-art performance in long-term forecasting across most datasets. 

**Abstract (ZH)**: 尽管大规模语言模型（LLMs）在处理和生成文本数据方面表现出色，但它们的预训练词汇表难以捕捉时间序列数据中内在的细腻的时间动态和模式。这些词汇表设计用于表示的自然语言符号是离散的，而时间序列数据的连续数值性质与之不匹配。为了克服这一根本限制，我们提出了VITRO方法。该方法从视觉语言领域引入文本反转优化，以学习一种新的针对每种数据集的时间序列词汇表，该词汇表能够弥合自然语言的离散语义性质与时间序列数据的连续数值性质之间的差距。我们证明，学习的时间序列特定伪词嵌入比现有的通用语言模型词汇表更好地表示时间序列数据，并且通过VITRO增强的方法在大多数数据集上实现了长期预测的最先进性能。 

---
# A Multimodal Emotion Recognition System: Integrating Facial Expressions, Body Movement, Speech, and Spoken Language 

**Title (ZH)**: 多模态情感识别系统：融合面部表情、身体动作、语音和口语情感 

**Authors**: Kris Kraack  

**Link**: [PDF](https://arxiv.org/pdf/2412.17907)  

**Abstract**: Traditional psychological evaluations rely heavily on human observation and interpretation, which are prone to subjectivity, bias, fatigue, and inconsistency. To address these limitations, this work presents a multimodal emotion recognition system that provides a standardised, objective, and data-driven tool to support evaluators, such as psychologists, psychiatrists, and clinicians. The system integrates recognition of facial expressions, speech, spoken language, and body movement analysis to capture subtle emotional cues that are often overlooked in human evaluations. By combining these modalities, the system provides more robust and comprehensive emotional state assessment, reducing the risk of mis- and overdiagnosis. Preliminary testing in a simulated real-world condition demonstrates the system's potential to provide reliable emotional insights to improve the diagnostic accuracy. This work highlights the promise of automated multimodal analysis as a valuable complement to traditional psychological evaluation practices, with applications in clinical and therapeutic settings. 

**Abstract (ZH)**: 传统的心理评估主要依赖于人类的观察和解释，这容易导致主观性、偏见、疲劳和不一致性。为了解决这些问题，本研究提出了一种多模态情感识别系统，该系统提供了一种标准化、客观和数据驱动的工具，以支持评估者，如心理学家、精神科医生和临床医生。该系统通过综合面部表情、言语、口头语言和身体动作分析，来捕捉人类评估中常被忽略的细微情感线索。通过结合这些模态，系统提供了一种更为稳健和全面的情绪状态评估，降低了误诊和过度诊断的风险。在模拟真实世界条件下的初步测试表明，该系统有可能提供可靠的情感洞察，以提高诊断准确性。本研究突显了自动化多模态分析在补充传统心理评估实践方面的潜力，其应用范围涵盖了临床和治疗领域。 

---
# Bridging the Data Provenance Gap Across Text, Speech and Video 

**Title (ZH)**: 跨文本、语音和视频的数据来源鸿沟桥梁构建 

**Authors**: Shayne Longpre, Nikhil Singh, Manuel Cherep, Kushagra Tiwary, Joanna Materzynska, William Brannon, Robert Mahari, Manan Dey, Mohammed Hamdy, Nayan Saxena, Ahmad Mustafa Anis, Emad A. Alghamdi, Vu Minh Chien, Naana Obeng-Marnu, Da Yin, Kun Qian, Yizhi Li, Minnie Liang, An Dinh, Shrestha Mohanty, Deividas Mataciunas, Tobin South, Jianguo Zhang, Ariel N. Lee, Campbell S. Lund, Christopher Klamm, Damien Sileo, Diganta Misra, Enrico Shippole, Kevin Klyman, Lester JV Miranda, Niklas Muennighoff, Seonghyeon Ye, Seungone Kim, Vipul Gupta, Vivek Sharma, Xuhui Zhou, Caiming Xiong, Luis Villa, Stella Biderman, Alex Pentland, Sara Hooker, Jad Kabbara  

**Link**: [PDF](https://arxiv.org/pdf/2412.17847)  

**Abstract**: Progress in AI is driven largely by the scale and quality of training data. Despite this, there is a deficit of empirical analysis examining the attributes of well-established datasets beyond text. In this work we conduct the largest and first-of-its-kind longitudinal audit across modalities--popular text, speech, and video datasets--from their detailed sourcing trends and use restrictions to their geographical and linguistic representation. Our manual analysis covers nearly 4000 public datasets between 1990-2024, spanning 608 languages, 798 sources, 659 organizations, and 67 countries. We find that multimodal machine learning applications have overwhelmingly turned to web-crawled, synthetic, and social media platforms, such as YouTube, for their training sets, eclipsing all other sources since 2019. Secondly, tracing the chain of dataset derivations we find that while less than 33% of datasets are restrictively licensed, over 80% of the source content in widely-used text, speech, and video datasets, carry non-commercial restrictions. Finally, counter to the rising number of languages and geographies represented in public AI training datasets, our audit demonstrates measures of relative geographical and multilingual representation have failed to significantly improve their coverage since 2013. We believe the breadth of our audit enables us to empirically examine trends in data sourcing, restrictions, and Western-centricity at an ecosystem-level, and that visibility into these questions are essential to progress in responsible AI. As a contribution to ongoing improvements in dataset transparency and responsible use, we release our entire multimodal audit, allowing practitioners to trace data provenance across text, speech, and video. 

**Abstract (ZH)**: 人工智能的进步主要受到训练数据的数量和质量的推动。尽管如此，目前对非文本数据集的属性进行实证分析仍然存在不足。本文中，我们进行了迄今为止覆盖范围最广的跨模态纵向审计，从数据的详细来源趋势和使用限制，到地理和语言的代表性。我们的手动分析涵盖了1990年至2024年间近4000个公开数据集，涉及608种语言、798个数据源、659个组织和67个国家。

我们发现，多模态机器学习应用程序已经大量转向依赖网络抓取、合成和社交媒体平台（如YouTube）的训练集，自2019年以来其他来源的数据已经黯然失色。其次，通过追溯数据集的来源链，我们发现虽然只有不到33%的数据集受到限制性许可，但广泛使用中的文本、语音和视频数据集中超过80%的原始内容带有非商业限制。最后，尽管公共AI训练数据集代表的语言和地理数量不断增加，我们的审计表明，自2013年以来，相对地理和多语言方面的代表性覆盖率并没有显著改善。

我们认为，我们审计的广度使我们能够从生态系统层面实证地考察数据来源、限制以及西方中心主义的趋势。对这些问题的透明度认识对于负责任的人工智能的进步至关重要。作为提高数据集透明度和负责使用的一个贡献，我们发布了整个多模态审计的详细信息，允许实践者追踪文本、语音和视频数据的来源。 

---
# Ensemble Machine Learning Model for Inner Speech Recognition: A Subject-Specific Investigation 

**Title (ZH)**: 针对个体差异的内部言语识别 ensemble 机器学习模型研究 

**Authors**: Shahamat Mustavi Tasin, Muhammad E. H. Chowdhury, Shona Pedersen, Malek Chabbouh, Diala Bushnaq, Raghad Aljindi, Saidul Kabir, Anwarul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2412.17824)  

**Abstract**: Inner speech recognition has gained enormous interest in recent years due to its applications in rehabilitation, developing assistive technology, and cognitive assessment. However, since language and speech productions are a complex process, for which identifying speech components has remained a challenging task. Different approaches were taken previously to reach this goal, but new approaches remain to be explored. Also, a subject-oriented analysis is necessary to understand the underlying brain dynamics during inner speech production, which can bring novel methods to neurological research. A publicly available dataset, Thinking Out Loud Dataset, has been used to develop a Machine Learning (ML)-based technique to classify inner speech using 128-channel surface EEG signals. The dataset is collected on a Spanish cohort of ten subjects while uttering four words (Arriba, Abajo, Derecha, and Izquierda) by each participant. Statistical methods were employed to detect and remove motion artifacts from the Electroencephalography (EEG) signals. A large number (191 per channel) of time-, frequency- and time-frequency-domain features were extracted. Eight feature selection algorithms are explored, and the best feature selection technique is selected for subsequent evaluations. The performance of six ML algorithms is evaluated, and an ensemble model is proposed. Deep Learning (DL) models are also explored, and the results are compared with the classical ML approach. The proposed ensemble model, by stacking the five best logistic regression models, generated an overall accuracy of 81.13% and an F1 score of 81.12% in the classification of four inner speech words using surface EEG signals. The proposed framework with the proposed ensemble of classical ML models shows promise in the classification of inner speech using surface EEG signals. 

**Abstract (ZH)**: 近年来，内语言识别因其在康复、辅助技术发展和认知评估中的应用而引起了广泛关注。然而，由于语言和言语产生是一个复杂的过程，识别言语成分仍是一个挑战。先前采取了不同的方法来实现这一目标，但仍需探索新的方法。此外，主题导向的分析对于理解内语言产生过程中的脑动态至关重要，这可以为神经科学研究带来新的方法。已公布的数据集“Thinking Out Loud Dataset”被用于开发基于机器学习（ML）的技术，以使用128通道表面EEG信号分类内语言。该数据集由西班牙语语境下的十个被试产生，每个被试说出四个词语（Arriba、Abajo、Derecha、和Izquierda）。统计方法被用于检测和去除EEG信号中的运动伪迹。提取了大量的时间域、频率域和时频域特征（每个通道191个）。探索了八种特征选择算法，并从中选择了最佳特征选择技术进行后续评估。评估了六种ML算法的性能，并提出了一个集成模型。还探索了深度学习（DL）模型，并将其结果与经典的ML方法进行了比较。提出的集成模型通过堆叠五种最佳逻辑回归模型，产生了81.13%的整体准确率和81.12%的F1分数，在表面EEG信号分类四个内语言词语上的表现。所提出的框架与提出的经典ML模型集成显示出了在使用表面EEG信号分类内语言方面的潜力。 

---
# Large Language Model Safety: A Holistic Survey 

**Title (ZH)**: 大型语言模型安全性：一篇全面的综述 

**Authors**: Dan Shi, Tianhao Shen, Yufei Huang, Zhigen Li, Yongqi Leng, Renren Jin, Chuang Liu, Xinwei Wu, Zishan Guo, Linhao Yu, Ling Shi, Bojian Jiang, Deyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2412.17686)  

**Abstract**: The rapid development and deployment of large language models (LLMs) have introduced a new frontier in artificial intelligence, marked by unprecedented capabilities in natural language understanding and generation. However, the increasing integration of these models into critical applications raises substantial safety concerns, necessitating a thorough examination of their potential risks and associated mitigation strategies.
This survey provides a comprehensive overview of the current landscape of LLM safety, covering four major categories: value misalignment, robustness to adversarial attacks, misuse, and autonomous AI risks. In addition to the comprehensive review of the mitigation methodologies and evaluation resources on these four aspects, we further explore four topics related to LLM safety: the safety implications of LLM agents, the role of interpretability in enhancing LLM safety, the technology roadmaps proposed and abided by a list of AI companies and institutes for LLM safety, and AI governance aimed at LLM safety with discussions on international cooperation, policy proposals, and prospective regulatory directions.
Our findings underscore the necessity for a proactive, multifaceted approach to LLM safety, emphasizing the integration of technical solutions, ethical considerations, and robust governance frameworks. This survey is intended to serve as a foundational resource for academy researchers, industry practitioners, and policymakers, offering insights into the challenges and opportunities associated with the safe integration of LLMs into society. Ultimately, it seeks to contribute to the safe and beneficial development of LLMs, aligning with the overarching goal of harnessing AI for societal advancement and well-being. A curated list of related papers has been publicly available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）的快速开发和部署为人工智能领域开辟了一个新的前沿，其在自然语言理解和生成方面展现了前所未有的能力。然而，这些模型日益整合到关键应用中引发了重大的安全关切，需要全面审视其潜在风险及其缓解策略。

本文综述了当前LLM安全的现状，涵盖了四大主要类别：价值偏差、对抗性攻击的稳健性、误用和自主人工智能的风险。除了对这些四个方面进行全面的缓解方法和评估资源的回顾之外，我们还探讨了与LLM安全相关的四个主题：LLM代理的安全影响、提高LLM安全性的可解释性作用、一系列AI公司和机构为LLM安全提出的和遵守的技术路线图，以及旨在促进LLM安全的治理措施，包括国际合作、政策建议和潜在的监管方向。

我们的研究强调了对LLM安全采取积极、多维度方法的必要性，强调了技术解决方案、伦理考量和稳健治理框架的集成。本文旨在为学术研究人员、产业从业者和政策制定者提供一个基础资源，帮助他们了解将LLM安全地融入社会所面临的挑战和机遇。最终，本文旨在为LLM的安全和有益发展做出贡献，与整体目标一致，即利用人工智能促进社会进步和福祉。

相关论文的精选清单已在此处 https://this-url 提供给公众。 

---
