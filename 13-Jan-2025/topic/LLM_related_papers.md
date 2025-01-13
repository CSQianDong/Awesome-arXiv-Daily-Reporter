# Navigating Tomorrow: Reliably Assessing Large Language Models Performance on Future Event Prediction 

**Title (ZH)**: 导航未来：可靠评估大规模语言模型在预测未来事件方面的性能 

**Authors**: Petraq Nako, Adam Jatowt  

**Link**: [PDF](https://arxiv.org/pdf/2501.05925)  

**Abstract**: Predicting future events is an important activity with applications across multiple fields and domains. For example, the capacity to foresee stock market trends, natural disasters, business developments, or political events can facilitate early preventive measures and uncover new opportunities. Multiple diverse computational methods for attempting future predictions, including predictive analysis, time series forecasting, and simulations have been proposed. This study evaluates the performance of several large language models (LLMs) in supporting future prediction tasks, an under-explored domain. We assess the models across three scenarios: Affirmative vs. Likelihood questioning, Reasoning, and Counterfactual analysis. For this, we create a dataset1 by finding and categorizing news articles based on entity type and its popularity. We gather news articles before and after the LLMs training cutoff date in order to thoroughly test and compare model performance. Our research highlights LLMs potential and limitations in predictive modeling, providing a foundation for future improvements. 

**Abstract (ZH)**: 预测未来事件是一项重要的活动，具有跨多个领域和行业的广泛应用。例如，预见股市走势、自然灾害、企业发展或政治事件的能力可以促进早期预防措施并发现新的机遇。多种不同的计算方法被提出以尝试进行未来预测，包括预测分析、时间序列预测和模拟等。本研究评估了几种大规模语言模型（LLMs）在支持未来预测任务中的表现，这是一个尚未广泛探索的领域。我们通过基于实体类型及其流行度来查找并分类新闻文章来构建数据集，并收集了LLMs训练截止日期前后发布的新闻文章，以全面测试和比较模型性能。我们的研究突出了LLMs在预测建模中的潜力和局限性，为未来改进奠定了基础。 

---
# Retrieval-Augmented Generation by Evidence Retroactivity in LLMs 

**Title (ZH)**: 基于证据回溯的LLM中检索增强生成 

**Authors**: Liang Xiao, Wen Dai, Shuai Chen, Bin Qin, Chongyang Shi, Haopeng Jing, Tianyu Guo  

**Link**: [PDF](https://arxiv.org/pdf/2501.05475)  

**Abstract**: Retrieval-augmented generation has gained significant attention due to its ability to integrate relevant external knowledge, enhancing the accuracy and reliability of the LLMs' responses. Most of the existing methods apply a dynamic multiple retrieval-generating process, to address multi-hop complex questions by decomposing them into sub-problems. However, these methods rely on an unidirectional forward reasoning paradigm, where errors from insufficient reasoning steps or inherent flaws in current retrieval systems are irreversible, potentially derailing the entire reasoning chain. For the first time, this work introduces Retroactive Retrieval-Augmented Generation (RetroRAG), a novel framework to build a retroactive reasoning paradigm. RetroRAG revises and updates the evidence, redirecting the reasoning chain to the correct direction. RetroRAG constructs an evidence-collation-discovery framework to search, generate, and refine credible evidence. It synthesizes inferential evidence related to the key entities in the question from the existing source knowledge and formulates search queries to uncover additional information. As new evidence is found, RetroRAG continually updates and organizes this information, enhancing its ability to locate further necessary evidence. Paired with an Answerer to generate and evaluate outputs, RetroRAG is capable of refining its reasoning process iteratively until a reliable answer is obtained. Empirical evaluations show that RetroRAG significantly outperforms existing methods. 

**Abstract (ZH)**: 检索增强生成由于其整合相关外部知识的能力，在提高大语言模型（LLM）响应的准确性和可靠性方面获得了广泛关注。目前大多数现有方法采用动态多轮检索-生成过程，通过将多跳复杂问题分解为子问题来解决问题。然而，这些方法依赖于单向前向推理模式，其中由于推理步骤不足或当前检索系统中的内在缺陷导致的错误是不可逆的，可能会导致整个推理链的偏离。首次提出，本工作引入了回溯检索增强生成（RetroRAG），一种新颖的框架以构建回溯推理范式。RetroRAG 能够修正和更新证据，重新引导推理链的方向。RetroRAG 构建了一个证据收集发现框架，用于搜索、生成和完善可信证据。它从现有来源知识中合成与问题关键实体相关的推断性证据，并制定搜索查询以发现额外信息。随着新证据的发现，RetroRAG 不断更新和组织这些信息，增强其定位进一步所需的证据的能力。结合一个回答器生成和评估输出，RetroRAG 能够迭代地改进其推理过程，直到获得可靠的答案。实证评估表明，RetroRAG 显著优于现有方法。 

---
# LLM-MedQA: Enhancing Medical Question Answering through Case Studies in Large Language Models 

**Title (ZH)**: LLM-MedQA：通过大型语言模型案例研究增强医学问答能力 

**Authors**: Hang Yang, Hao Chen, Hui Guo, Yineng Chen, Ching-Sheng Lin, Shu Hu, Jinrong Hu, Xi Wu, Xin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.05464)  

**Abstract**: Accurate and efficient question-answering systems are essential for delivering high-quality patient care in the medical field. While Large Language Models (LLMs) have made remarkable strides across various domains, they continue to face significant challenges in medical question answering, particularly in understanding domain-specific terminologies and performing complex reasoning. These limitations undermine their effectiveness in critical medical applications. To address these issues, we propose a novel approach incorporating similar case generation within a multi-agent medical question-answering (MedQA) system. Specifically, we leverage the Llama3.1:70B model, a state-of-the-art LLM, in a multi-agent architecture to enhance performance on the MedQA dataset using zero-shot learning. Our method capitalizes on the model's inherent medical knowledge and reasoning capabilities, eliminating the need for additional training data. Experimental results show substantial performance gains over existing benchmark models, with improvements of 7% in both accuracy and F1-score across various medical QA tasks. Furthermore, we examine the model's interpretability and reliability in addressing complex medical queries. This research not only offers a robust solution for medical question answering but also establishes a foundation for broader applications of LLMs in the medical domain. 

**Abstract (ZH)**: 准确且高效的问答系统对于提供高质量的医疗服务至关重要。尽管大型语言模型（LLMs）已经在各个领域取得了显著进展，但在医疗领域的问答任务中，它们仍然面临重大挑战，特别是在理解和处理领域特定术语以及进行复杂推理方面。这些限制削弱了它们在关键医疗应用中的效果。为了解决这些问题，我们提出了一种新颖的方法，即在多智能体医疗问答（MedQA）系统中引入相似病例生成技术。具体而言，我们利用最先进的大型语言模型Llama3.1:70B，在多智能体架构中进行零样本学习，以增强MedQA数据集上的性能。我们的方法充分利用了模型内置的医学知识和推理能力，无需额外的训练数据。实验结果显示，与现有基准模型相比，我们的方法在各种医学问答任务中实现了显著的性能提升，准确率和F1分数分别提高了7%。此外，我们还考察了该模型在处理复杂医学查询时的可解释性和可靠性。这项研究不仅提供了一种稳健的解决方案以应对医学问答任务，还为我们探索大型语言模型在医疗领域的更广泛应用奠定了基础。 

---
# Semantic Exploration with Adaptive Gating for Efficient Problem Solving with Language Models 

**Title (ZH)**: 面向语言模型高效问题求解的自适应门控语义探索 

**Authors**: Sungjae Lee, Hyejin Park, Jaechang Kim, Jungseul Ok  

**Link**: [PDF](https://arxiv.org/pdf/2501.05752)  

**Abstract**: Recent advancements in large language models (LLMs) have shown remarkable potential in various complex tasks requiring multi-step reasoning methods like tree search to explore diverse reasoning paths. However, existing methods often suffer from computational inefficiency and redundancy. First, they overlook the diversity of task difficulties, leading to unnecessarily extensive searches even for easy tasks. Second, they neglect the semantics of reasoning paths, resulting in redundant exploration of semantically identical paths. To address these limitations, we propose Semantic Exploration with Adaptive Gating (SEAG), a computationally efficient method. SEAG employs an adaptive gating mechanism that dynamically decides whether to conduct a tree search, based on the confidence level of answers from a preceding simple reasoning method. Furthermore, its tree-based exploration consolidates semantically identical reasoning steps, reducing redundant explorations while maintaining or even improving accuracy. Our extensive experiments demonstrate that SEAG significantly improves accuracy by 4.3% on average while requiring only 31% of computational costs compared to existing tree search-based methods on complex reasoning benchmarks including GSM8K and ARC with diverse language models such as Llama2, Llama3, and Mistral. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）在需要多步推理方法的任务中，如树搜索以探索多种推理路径方面，展现出了显著的潜力。然而，现有的方法常常存在计算效率低下和冗余的问题。首先，这些方法忽视了任务难度的多样性，导致即使是简单的任务也需要进行不必要的广泛搜索。其次，它们忽略了推理路径的意义，导致对语义相同的路径进行了冗余探索。为了克服这些局限性，我们提出了一种计算高效的Semantic Exploration with Adaptive Gating（SEAG），即自适应门控的语义探索方法。SEAG采用了一种自适应门控机制，根据先前简单推理方法提供的答案置信度水平，动态决定是否进行树搜索。此外，其基于树的探索能够合并具有相同语义的推理步骤，从而减少冗余探索，同时保持或甚至提高准确性。我们的大量实验表明，在包括GSM8K和ARC等复杂推理基准测试中，使用各种语言模型（如Llama2、Llama3和Mistral）时，SEAG相比现有的基于树搜索的方法，在计算成本降低31%的同时，平均提高了4.3%的准确性。 

---
# Facilitate Collaboration between Large Language Model and Task-specific Model for Time Series Anomaly Detection 

**Title (ZH)**: 促进大型语言模型与任务特定模型在时间序列异常检测中的合作 

**Authors**: Feiyi Chen, Leilei Zhang, Guansong Pang, Roger Zimmermann, Shuiguang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2501.05675)  

**Abstract**: In anomaly detection, methods based on large language models (LLMs) can incorporate expert knowledge, while task-specific smaller models excel at extracting normal patterns and detecting value fluctuations. Inspired by the human nervous system, where the brain stores expert knowledge and the peripheral nervous system and spinal cord handle specific tasks like withdrawal and knee-jerk reflexes, we propose CoLLaTe, a framework designed to facilitate collaboration between LLMs and task-specific models, leveraging the strengths of both.
In this work, we first formulate the collaboration process and identify two key challenges in the collaboration between LLMs and task-specific models: (1) the misalignment between the expression domains of LLMs and smaller models, and (2) error accumulation arising from the predictions of both models.
To address these challenges, we introduce two key components in CoLLaTe: the alignment module and the collaborative loss function. Through theoretical analysis and experimental validation, we demonstrate that these components effectively mitigate the identified challenges and achieve better performance than LLM based methods and task-specific smaller model. 

**Abstract (ZH)**: 在异常检测领域，基于大型语言模型（LLMs）的方法可以融入专家知识，而针对特定任务的小型模型则擅长提取正常模式和检测价值波动。受到人类神经系统启发，大脑存储专家知识而外周神经系统和脊髓处理特定任务（如收缩反射和膝跳反射），我们提出了一种名为CoLLaTe的框架，旨在促进LLMs与特定任务模型之间的合作，充分利用双方的优势。

在这项工作中，我们首先定义了合作过程，并指出了LLMs与特定任务模型之间合作中的两个关键挑战：（1）LLMs和小型模型的表达域不匹配，以及（2）两种模型预测过程中的误差累积。

为解决这些挑战，我们在CoLLaTe中引入了两个关键组成部分：对齐模块和协作损失函数。通过理论分析和实验验证，我们展示了这些组成部分有效地缓解了上述挑战，并在性能上优于基于LLM的方法和特定任务的小型模型。 

---
# Contextual ASR Error Handling with LLMs Augmentation for Goal-Oriented Conversational AI 

**Title (ZH)**: 基于上下文的ASR错误处理：通过LLM增强的目标导向会话AI中的应用 

**Authors**: Yuya Asano, Sabit Hassan, Paras Sharma, Anthony Sicilia, Katherine Atwell, Diane Litman, Malihe Alikhani  

**Link**: [PDF](https://arxiv.org/pdf/2501.06129)  

**Abstract**: General-purpose automatic speech recognition (ASR) systems do not always perform well in goal-oriented dialogue. Existing ASR correction methods rely on prior user data or named entities. We extend correction to tasks that have no prior user data and exhibit linguistic flexibility such as lexical and syntactic variations. We propose a novel context augmentation with a large language model and a ranking strategy that incorporates contextual information from the dialogue states of a goal-oriented conversational AI and its tasks. Our method ranks (1) n-best ASR hypotheses by their lexical and semantic similarity with context and (2) context by phonetic correspondence with ASR hypotheses. Evaluated in home improvement and cooking domains with real-world users, our method improves recall and F1 of correction by 34% and 16%, respectively, while maintaining precision and false positive rate. Users rated .8-1 point (out of 5) higher when our correction method worked properly, with no decrease due to false positives. 

**Abstract (ZH)**: 通用的自动语音识别（ASR）系统在目标导向对话中并不总是表现出色。现有的ASR纠错方法依赖于用户的先验数据或命名实体。我们扩展了纠错到那些没有用户先验数据且表现出语文灵活性的任务，如词汇和句法变化。我们提出了一种新的上下文增强方法，该方法结合了大规模语言模型和一种排名策略，该策略综合了目标导向对话AI及其任务在对话状态中的上下文信息。我们的方法通过（1）按与上下文的词汇和语义相似度对n-best ASR假设进行排名，以及（2）通过音素对应关系对上下文进行排名。在家庭装修和烹饪领域通过真实用户进行了评估，我们的方法在召回率和F1值上分别提高了34%和16%，同时保持了精确度和假阳性率。当我们的纠错方法正常工作时，用户评分提高了0.8至1分（满分5分），并且没有因假阳性而降低评分。 

---
# Addressing speaker gender bias in large scale speech translation systems 

**Title (ZH)**: 大规模语音翻译系统中讲话者性别偏见的应对方法 

**Authors**: Shubham Bansal, Vikas Joshi, Harveen Chadha, Rupeshkumar Mehta, Jinyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.05989)  

**Abstract**: This study addresses the issue of speaker gender bias in Speech Translation (ST) systems, which can lead to offensive and inaccurate translations. The masculine bias often found in large-scale ST systems is typically perpetuated through training data derived from Machine Translation (MT) systems. Our approach involves two key steps. First, we employ Large Language Models (LLMs) to rectify translations based on the speaker's gender in a cost-effective manner. Second, we fine-tune the ST model with the corrected data, enabling the model to generate gender-specific translations directly from audio cues, without the need for explicit gender input. Additionally, we propose a three-mode fine-tuned model for scenarios where the speaker's gender is either predefined or should not be inferred from speech cues. We demonstrate a 70% improvement in translations for female speakers compared to our baseline and other large-scale ST systems, such as Seamless M4T and Canary, on the MuST-SHE test set. 

**Abstract (ZH)**: 本研究针对语音翻译（ST）系统中的说话人口音偏差问题进行了研究，这一问题可能导致冒犯性和不准确的翻译。大规模ST系统中常见的男性偏见通常通过源于机器翻译（MT）系统的训练数据得以延续。我们的方法包括两个关键步骤。首先，我们利用大型语言模型（LLMs）以经济高效的方式纠正翻译中的性别偏差。其次，我们使用修正后的数据微调ST模型，使模型能够直接从音频提示中生成性别特定的翻译，而无需明确输入性别信息。此外，我们还提出了一种三模式微调模型，适用于说话人口音已预定义或不应从语音提示中推断的情况。我们使用MuST-SHE测试集展示了与基线及其他大规模ST系统（如Seamless M4T和Canary）相比，对于女性说话人的翻译改进率达到70%。 

---
# Affordably Fine-tuned LLMs Provide Better Answers to Course-specific MCQs 

**Title (ZH)**: 经济实惠地微调的大型语言模型提供了更好的课程特定选择题答案 

**Authors**: Bianca Raimondi, Saverio Giallorenzo, Maurizio Gabbrielli  

**Link**: [PDF](https://arxiv.org/pdf/2501.05891)  

**Abstract**: In education, the capability of generating human-like text of Large Language Models (LLMs) inspired work on how they can increase the efficiency of learning and teaching. We study the affordability of these models for educators and students by investigating how LLMs answer multiple-choice questions (MCQs) with respect to hardware constraints and refinement techniques. We explore this space by using generic pre-trained LLMs (the 7B, 13B, and 70B variants of LLaMA-2) to answer 162 undergraduate-level MCQs from a course on Programming Languages (PL) -- the MCQ dataset is a contribution of this work, which we make publicly available. Specifically, we dissect how different factors, such as using readily-available material -- (parts of) the course's textbook -- for fine-tuning and quantisation (to decrease resource usage) can change the accuracy of the responses. The main takeaway is that smaller textbook-based fine-tuned models outperform generic larger ones (whose pre-training requires conspicuous resources), making the usage of LLMs for answering MCQs resource- and material-wise affordable. 

**Abstract (ZH)**: 在教育领域，大型语言模型（LLMs）生成类人类文本的能力激发了它们如何能提高学习和教学效率的研究。我们通过探讨硬件限制和改进技术，研究这些模型对教育工作者和学生来说的经济性。我们使用通用预训练的LLM（LLaMA-2 7B、13B和70B变体）来回答一门编程语言（PL）课程中的162道本科水平的多项选择题（MCQs），并因此构建了一个MCQ数据集，该数据集已公开提供。具体来说，我们分析了使用现成的课程教材的部分内容进行微调和量化（以减少资源使用）等因素如何影响模型的响应准确性。主要结论是，基于教材的较小模型的性能优于通用的大型模型（其预训练需要大量资源），这使得使用LLMs回答MCQs从资源和材料的角度来看是可行且经济的。 

---
# Enabling Scalable Oversight via Self-Evolving Critic 

**Title (ZH)**: 通过自我进化的评论者实现 scalable 监督 

**Authors**: Zhengyang Tang, Ziniu Li, Zhenyang Xiao, Tian Ding, Ruoyu Sun, Benyou Wang, Dayiheng Liu, Fei Huang, Tianyu Liu, Bowen Yu, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2501.05727)  

**Abstract**: Despite their remarkable performance, the development of Large Language Models (LLMs) faces a critical challenge in scalable oversight: providing effective feedback for tasks where human evaluation is difficult or where LLMs outperform humans. While there is growing interest in using LLMs for critique, current approaches still rely on human annotations or more powerful models, leaving the issue of enhancing critique capabilities without external supervision unresolved. We introduce SCRIT (Self-evolving CRITic), a framework that enables genuine self-evolution of critique abilities. Technically, SCRIT self-improves by training on synthetic data, generated by a contrastive-based self-critic that uses reference solutions for step-by-step critique, and a self-validation mechanism that ensures critique quality through correction outcomes. Implemented with Qwen2.5-72B-Instruct, one of the most powerful LLMs, SCRIT achieves up to a 10.3\% improvement on critique-correction and error identification benchmarks. Our analysis reveals that SCRIT's performance scales positively with data and model size, outperforms alternative approaches, and benefits critically from its self-validation component. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）表现出色，但其发展面临着可扩展监督的关键挑战：为那些难以进行人类评估或LLMs超越人类的任务提供有效反馈。虽然越来越多的研究正在利用LLMs进行批评，但当前的方法仍然依赖于人工注释或更强大的模型，这使得在缺乏外部监督的情况下增强批评能力的问题仍未解决。我们提出了SCRIT（Self-evolving CRITic）框架，该框架能够真正实现批评能力的自我进化。从技术上讲，SCRIT通过利用基于对比的方法生成的合成数据进行自我改进，这些数据源自一个使用参考解决方案进行逐步批评的自我批评机制，以及一个自我验证机制，该机制通过纠正结果确保批评质量。采用目前最强大的LLM之一Qwen2.5-72B-Instruct实现后，SCRIT在批评-纠正和错误识别基准上的性能提升了10.3%。我们的分析表明，SCRIT的性能随着数据和模型规模的增加而正向扩展，超越了其他替代方法，并且严重受益于其自我验证组件。 

---
# Multiagent Finetuning: Self Improvement with Diverse Reasoning Chains 

**Title (ZH)**: 多智能体微调：通过多样化推理链实现自我提升 

**Authors**: Vighnesh Subramaniam, Yilun Du, Joshua B. Tenenbaum, Antonio Torralba, Shuang Li, Igor Mordatch  

**Link**: [PDF](https://arxiv.org/pdf/2501.05707)  

**Abstract**: Large language models (LLMs) have achieved remarkable performance in recent years but are fundamentally limited by the underlying training data. To improve models beyond the training data, recent works have explored how LLMs can be used to generate synthetic data for autonomous self-improvement. However, successive steps of self-improvement can reach a point of diminishing returns. In this work, we propose a complementary approach towards self-improvement where finetuning is applied to a multiagent society of language models. A group of language models, all starting from the same base model, are independently specialized by updating each one using data generated through multiagent interactions among the models. By training each model on independent sets of data, we illustrate how this approach enables specialization across models and diversification over the set of models. As a result, our overall system is able to preserve diverse reasoning chains and autonomously improve over many more rounds of fine-tuning than single-agent self-improvement methods. We quantitatively illustrate the efficacy of the approach across a wide suite of reasoning tasks. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）在多个任务中取得了显著的性能，但其根本上仍受限于训练数据。为了超越训练数据的限制，近期的研究探索了如何利用LLMs生成合成数据以促进自主自适应。然而，连续的自适应步骤可能会达到边际收益递减的阶段。在本文中，我们提出了一种互补的自适应方法，在该方法中，对多智能体社会中的语言模型进行微调。一群从同一基础模型出发的语言模型通过模型间的多智能体交互生成数据，各自独立地进行专业化训练。通过让每个模型在独立的数据集上进行训练，本文展示了这种方法如何在模型之间实现专业化，并在模型集合中实现多样化。因此，我们的整体系统能够在多轮次的微调中实现更自主的进步，并且比单智能体自适应方法能够进行更多轮次的优化。我们通过广泛的心理任务验证了该方法的有效性。 

---
# Cascaded Self-Evaluation Augmented Training for Efficient Multimodal Large Language Models 

**Title (ZH)**: 级联自评估增强训练以提高高效多模态大型语言模型的效果 

**Authors**: Zheqi Lv, Wenkai Wang, Jiawei Wang, Shengyu Zhang, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2501.05662)  

**Abstract**: Efficient Multimodal Large Language Models (EMLLMs) have rapidly advanced recently. Incorporating Chain-of-Thought (CoT) reasoning and step-by-step self-evaluation has improved their performance. However, limited parameters often hinder EMLLMs from effectively using self-evaluation during inference. Key challenges include synthesizing evaluation data, determining its quantity, optimizing training and inference strategies, and selecting appropriate prompts.
To address these issues, we introduce Self-Evaluation Augmented Training (SEAT). SEAT uses more powerful EMLLMs for CoT reasoning, data selection, and evaluation generation, then trains EMLLMs with the synthesized data. However, handling long prompts and maintaining CoT reasoning quality are problematic. Therefore, we propose Cascaded Self-Evaluation Augmented Training (Cas-SEAT), which breaks down lengthy prompts into shorter, task-specific cascaded prompts and reduces costs for resource-limited settings. During data synthesis, we employ open-source 7B-parameter EMLLMs and annotate a small dataset with short prompts.
Experiments demonstrate that Cas-SEAT significantly boosts EMLLMs' self-evaluation abilities, improving performance by 19.68%, 55.57%, and 46.79% on the MathVista, Math-V, and We-Math datasets, respectively. Additionally, our Cas-SEAT Dataset serves as a valuable resource for future research in enhancing EMLLM self-evaluation. 

**Abstract (ZH)**: 高效多模态大型语言模型（EMLLMs）最近取得了迅速的进步。引入链式思维（CoT）推理和逐步自我评估提高了其性能。然而，有限的参数往往妨碍EMLLMs在推理过程中有效地利用自我评估。关键挑战包括合成评估数据、确定其数量、优化训练和推理策略，以及选择合适的提示。

为了解决这些问题，我们提出了自我评估增强训练（SEAT）。SEAT 使用更强大的EMLLMs进行CoT推理、数据选择和评估生成，然后使用合成的数据训练EMLLMs。然而，处理长提示并保持CoT推理质量存在困难。因此，我们提出了分步自我评估增强训练（Cas-SEAT），该方法将长提示分解为更短、任务特定的分步提示，以降低资源有限环境下成本。在数据合成过程中，我们使用开源的7B参数EMLLMs，并对短提示进行了标注。

实验表明，Cas-SEAT 显著提高了EMLLMs的自我评估能力，在MathVista、Math-V和We-Math数据集上的表现分别提高了19.68%、55.57%和46.79%。此外，我们提出的Cas-SEAT数据集为未来增强EMLLM自我评估的研究提供了宝贵的资源。 

---
# Collaboration of Large Language Models and Small Recommendation Models for Device-Cloud Recommendation 

**Title (ZH)**: 将大型语言模型与小型推荐模型相结合的设备-云推荐协作方法 

**Authors**: Zheqi Lv, Tianyu Zhan, Wenjie Wang, Xinyu Lin, Shengyu Zhang, Wenqiao Zhang, Jiwei Li, Kun Kuang, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2501.05647)  

**Abstract**: Large Language Models (LLMs) for Recommendation (LLM4Rec) is a promising research direction that has demonstrated exceptional performance in this field. However, its inability to capture real-time user preferences greatly limits the practical application of LLM4Rec because (i) LLMs are costly to train and infer frequently, and (ii) LLMs struggle to access real-time data (its large number of parameters poses an obstacle to deployment on devices). Fortunately, small recommendation models (SRMs) can effectively supplement these shortcomings of LLM4Rec diagrams by consuming minimal resources for frequent training and inference, and by conveniently accessing real-time data on devices.
In light of this, we designed the Device-Cloud LLM-SRM Collaborative Recommendation Framework (LSC4Rec) under a device-cloud collaboration setting. LSC4Rec aims to integrate the advantages of both LLMs and SRMs, as well as the benefits of cloud and edge computing, achieving a complementary synergy. We enhance the practicability of LSC4Rec by designing three strategies: collaborative training, collaborative inference, and intelligent request. During training, LLM generates candidate lists to enhance the ranking ability of SRM in collaborative scenarios and enables SRM to update adaptively to capture real-time user interests. During inference, LLM and SRM are deployed on the cloud and on the device, respectively. LLM generates candidate lists and initial ranking results based on user behavior, and SRM get reranking results based on the candidate list, with final results integrating both LLM's and SRM's scores. The device determines whether a new candidate list is needed by comparing the consistency of the LLM's and SRM's sorted lists. Our comprehensive and extensive experimental analysis validates the effectiveness of each strategy in LSC4Rec. 

**Abstract (ZH)**: 大语言模型（LLMs）在推荐任务中的应用（LLM4Rec）是极具前景的研究方向，已经在推荐领域展示了出色的效果。然而，LLM4Rec 无法捕捉实时用户偏好，极大地限制了其实际应用，主要原因包括（i）LLM 训练和推理成本高昂，且（ii）LLM 难以访问实时数据（其参数量庞大是部署在设备上的障碍）。幸运的是，小型推荐模型（SRMs）能够有效补充 LLM4Rec 的这些不足，通过消耗最少资源频繁训练和推理，并方便地在设备上访问实时数据。

为了解决这些问题，我们设计了在设备-云协作框架下的设备-云LLM-SRM协作推荐框架（LSC4Rec）。LSC4Rec 的目标是融合 LLM 和 SRM 的优势，以及云计算和边缘计算的优势，实现互补协同。我们通过设计三种策略来增强 LSC4Rec 的实用性：协同训练、协同推理和智能请求。在训练过程中，LLM 生成候选列表，以增强 SRM 在协作场景下的排名能力，并使 SRM 能够根据用户偏好进行自适应更新以捕捉实时用户兴趣。在推理过程中，LLM 和 SRM 分别部署在云和设备上。LLM 根据用户行为生成候选列表和初步排名结果，SRM 基于候选列表生成二次排序结果，最终结果结合了 LLM 和 SRM 的分数。设备通过比较 LLM 和 SRM 排序列表的一致性来决定是否需要生成新的候选列表。我们进行了全面而广泛的实验分析，验证了 LSC4Rec 中每种策略的有效性。 

---
# LLMQuoter: Enhancing RAG Capabilities Through Efficient Quote Extraction From Large Contexts 

**Title (ZH)**: LLMQuoter：通过高效提取大规模语境中的引用来增强RAG能力 

**Authors**: Yuri Facanha Bezerra, Li Weigang  

**Link**: [PDF](https://arxiv.org/pdf/2501.05554)  

**Abstract**: We introduce LLMQuoter, a lightweight, distillation-based model designed to enhance Retrieval Augmented Generation (RAG) by extracting the most relevant textual evidence for downstream reasoning tasks. Built on the LLaMA-3B architecture and fine-tuned with Low-Rank Adaptation (LoRA) on a 15,000-sample subset of HotpotQA, LLMQuoter adopts a "quote-first-then-answer" strategy, efficiently identifying key quotes before passing curated snippets to reasoning models. This workflow reduces cognitive overhead and outperforms full-context approaches like Retrieval-Augmented Fine-Tuning (RAFT), achieving over 20-point accuracy gains across both small and large language models. By leveraging knowledge distillation from a high-performing teacher model, LLMQuoter achieves competitive results in a resource-efficient fine-tuning setup. It democratizes advanced RAG capabilities, delivering significant performance improvements without requiring extensive model retraining. Our results highlight the potential of distilled quote-based reasoning to streamline complex workflows, offering a scalable and practical solution for researchers and practitioners alike. 

**Abstract (ZH)**: 我们介绍了LLMQuoter，这是一个轻量级的蒸馏模型，旨在通过提取与下游推理由证任务最相关的文本证据来增强检索增强生成（RAG）。该模型基于LLaMA-3B架构，并在HotpotQA的一个包含15,000个样本的子集中使用低秩适应（LoRA）进行了微调。LLMQuoter采用“先引述再回答”的策略，在确定关键引述后，将精炼片段传递给推理模型。这种工作流程降低了认知负担，并在各个方面（无论是小型还是大型语言模型）都超越了全文上下文方法，如检索增强微调（RAFT），实现了超过20个百分点的准确性提升。通过利用高性能教师模型的知识蒸馏，LLMQuoter在资源高效微调设置中取得了竞争力的结果。它使高级RAG能力更加普及，能够在不进行大量模型重新训练的情况下实现显著的性能提升。我们的研究结果突显了知识蒸馏引述推理的潜力，可以通过简化复杂的工作流程来提供一种可扩展且实用的解决方案，对于研究者和实践者都具有重要意义。 

---
# The dynamics of meaning through time: Assessment of Large Language Models 

**Title (ZH)**: 意义随时间的动态演变：大型语言模型的评估 

**Authors**: Mohamed Taher Alrefaie, Fatty Salem, Nour Eldin Morsy, Nada Samir, Mohamed Medhat Gaber  

**Link**: [PDF](https://arxiv.org/pdf/2501.05552)  

**Abstract**: Understanding how large language models (LLMs) grasp the historical context of concepts and their semantic evolution is essential in advancing artificial intelligence and linguistic studies. This study aims to evaluate the capabilities of various LLMs in capturing temporal dynamics of meaning, specifically how they interpret terms across different time periods. We analyze a diverse set of terms from multiple domains, using tailored prompts and measuring responses through both objective metrics (e.g., perplexity and word count) and subjective human expert evaluations. Our comparative analysis includes prominent models like ChatGPT, GPT-4, Claude, Bard, Gemini, and Llama. Findings reveal marked differences in each model's handling of historical context and semantic shifts, highlighting both strengths and limitations in temporal semantic understanding. These insights offer a foundation for refining LLMs to better address the evolving nature of language, with implications for historical text analysis, AI design, and applications in digital humanities. 

**Abstract (ZH)**: 理解大型语言模型（LLMs）如何掌握概念的历史背景及其语义演变，对于推动人工智能和语言学研究至关重要。本研究旨在评估不同LLMs在捕捉意义的 temporal 动态方面的能力，特别是它们如何在不同时间时期解释术语。我们分析了来自多个领域的多种术语，并采用定制的提示和通过客观指标（如困惑度和词数）以及主观的人类专家评估来衡量响应。本研究的比较分析包括著名的模型，如ChatGPT、GPT-4、Claude、Bard、Gemini和Llama。研究发现，每个模型在处理历史背景和语义转变方面存在显著差异，突显了在时间语义理解方面的优势和局限性。这些见解为改进LLMs以更好地应对语言演变奠定了基础，并对历史文本分析、AI设计以及数字人文的应用具有重要意义。 

---
# RTLSquad: Multi-Agent Based Interpretable RTL Design 

**Title (ZH)**: RTLSquad: 基于多代理的可解释RTL设计 

**Authors**: Bowei Wang, Qi Xiong, Zeqing Xiang, Lei Wang, Renzhi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.05470)  

**Abstract**: Optimizing Register-Transfer Level (RTL) code is crucial for improving hardware PPA performance. Large Language Models (LLMs) offer new approaches for automatic RTL code generation and optimization. However, existing methods often lack decision interpretability (sufficient, understandable justification for decisions), making it difficult for hardware engineers to trust the generated results, thus preventing these methods from being integrated into the design process. To address this, we propose RTLSquad, a novel LLM-Based Multi-Agent system for interpretable RTL code generation. RTLSquad divides the design process into exploration, implementation, and verification & evaluation stages managed by specialized agent squads, generating optimized RTL code through inter-agent collaboration, and providing decision interpretability through the communication process. Experiments show that RTLSquad excels in generating functionally correct RTL code and optimizing PPA performance, while also having the capability to provide decision paths, demonstrating the practical value of our system. 

**Abstract (ZH)**: 优化寄存器传输级（RTL）代码对于提高硬件性能参数（PPA）至关重要。大型语言模型（LLMs）为自动RTL代码生成和优化提供了新的方法。然而，现有方法通常缺乏决策解释性（即，充分且可理解的决策依据），这使得硬件工程师难以信任生成的结果，从而导致这些方法难以集成到设计过程中。为解决这一问题，我们提出了一种名为RTLSquad的新型基于LLM的多智能体系统，用于可解释的RTL代码生成。RTLSquad将设计过程分为探索、实现、验证与评估阶段，并由专门的智能体战队管理这些阶段，通过智能体间的协作生成优化的RTL代码，并通过通信过程提供决策解释性。实验结果表明，RTLSquad在生成功能正确的RTL代码和优化PPA性能方面表现出色，同时还能提供决策路径，验证了我们系统的实际应用价值。 

---
# From Conversation to Automation: Leveraging Large Language Models to Analyze Strategies in Problem Solving Therapy 

**Title (ZH)**: 从对话到自动化：利用大规模语言模型分析问题解决疗法中的策略 

**Authors**: Elham Aghakhani, Lu Wang, Karla T. Washington, George Demiris, Jina Huh-Yoo, Rezvaneh Rezapour  

**Link**: [PDF](https://arxiv.org/pdf/2501.06101)  

**Abstract**: Problem-solving therapy (PST) is a structured psychological approach that helps individuals manage stress and resolve personal issues by guiding them through problem identification, solution brainstorming, decision-making, and outcome evaluation. As mental health care increasingly integrates technologies like chatbots and large language models (LLMs), understanding how PST can be effectively automated is important. This study leverages anonymized therapy transcripts to analyze and classify therapeutic interventions using various LLMs and transformer-based models. Our results show that GPT-4o achieved the highest accuracy (0.76) in identifying PST strategies, outperforming other models. Additionally, we introduced a new dimension of communication strategies that enhances the current PST framework, offering deeper insights into therapist-client interactions. This research demonstrates the potential of LLMs to automate complex therapeutic dialogue analysis, providing a scalable, efficient tool for mental health interventions. Our annotation framework can enhance the accessibility, effectiveness, and personalization of PST, supporting therapists in real-time with more precise, targeted interventions. 

**Abstract (ZH)**: 问题解决疗法（PST）是一种结构化的心理治疗方法，通过引导个体识别问题、 brainstorm 解决方案、做出决策和评估结果，帮助他们管理压力和解决个人问题。随着心理健康护理越来越多地结合聊天机器人和大型语言模型（LLMs）等技术，了解如何有效自动化 PST 是至关重要的。本研究利用匿名的心理治疗转录材料，使用各种 LLM 和基于变换器的模型进行分析和分类。研究结果表明，GPT-4o 在识别 PST 策略方面取得了最高准确率（0.76），超越了其他模型。此外，我们还引入了一种新的沟通策略维度，以增强现有的 PST 框架，提供对咨询师与求助者互动更深层次的洞察。本研究证明了 LLMs 在自动化复杂治疗对话分析方面的潜力，提供了一种可扩展且高效的工具，用于支持心理健康干预。我们的注解框架可以增强 PST 的可访问性、有效性及个性化，支持咨询师实时提供更加精准和针对性的干预措施。 

---
# ConSim: Measuring Concept-Based Explanations' Effectiveness with Automated Simulatability 

**Title (ZH)**: ConSim：基于概念的解释的自动可模拟性评估方法 

**Authors**: Antonin Poché, Alon Jacovi, Agustin Martin Picard, Victor Boutin, Fanny Jourdan  

**Link**: [PDF](https://arxiv.org/pdf/2501.05855)  

**Abstract**: Concept-based explanations work by mapping complex model computations to human-understandable concepts. Evaluating such explanations is very difficult, as it includes not only the quality of the induced space of possible concepts but also how effectively the chosen concepts are communicated to users. Existing evaluation metrics often focus solely on the former, neglecting the latter. We introduce an evaluation framework for measuring concept explanations via automated simulatability: a simulator's ability to predict the explained model's outputs based on the provided explanations. This approach accounts for both the concept space and its interpretation in an end-to-end evaluation. Human studies for simulatability are notoriously difficult to enact, particularly at the scale of a wide, comprehensive empirical evaluation (which is the subject of this work). We propose using large language models (LLMs) as simulators to approximate the evaluation and report various analyses to make such approximations reliable. Our method allows for scalable and consistent evaluation across various models and datasets. We report a comprehensive empirical evaluation using this framework and show that LLMs provide consistent rankings of explanation methods. Code available at this https URL 

**Abstract (ZH)**: 基于概念的解释通过将复杂的模型计算映射到人类可理解的概念来工作。评估这种解释非常困难，因为它不仅涉及生成的概念空间的质量，还包括所选概念向用户传达的效果。现有的评估指标往往只关注前者，忽视了后者。我们引入了一种基于自动仿真性测量概念解释的评估框架：仿真器根据提供的解释预测解释模型输出的能力。这种方法在端到端的评估中考虑了概念空间及其解释。在大规模、全面的实证评估中（这是本工作的内容），使人类实现仿真性研究尤为困难。我们提出使用大规模语言模型（LLMs）作为仿真器来近似评估，并报告各种分析以确保这种近似可靠。我们的方法可以在各种模型和数据集上实现可扩展且一致的评估。我们使用该框架进行了一项全面的实证评估，并证明LLMs可以一致地评估解释方法的效果。代码可在此处访问：https://github.com/... 

---
# Controlling Large Language Models Through Concept Activation Vectors 

**Title (ZH)**: 通过概念激活向量控制大型语言模型 

**Authors**: Hanyu Zhang, Xiting Wang, Chengao Li, Xiang Ao, Qing He  

**Link**: [PDF](https://arxiv.org/pdf/2501.05764)  

**Abstract**: As large language models (LLMs) are widely deployed across various domains, the ability to control their generated outputs has become more critical. This control involves aligning LLMs outputs with human values and ethical principles or customizing LLMs on specific topics or styles for individual users. Existing controlled generation methods either require significant computational resources and extensive trial-and-error or provide coarse-grained control. In this paper, we propose Generation with Concept Activation Vector (GCAV), a lightweight model control framework that ensures accurate control without requiring resource-extensive fine-tuning. Specifically, GCAV first trains a concept activation vector for specified concepts to be controlled, such as toxicity. During inference, GCAV steers the concept vector in LLMs, for example, by removing the toxicity concept vector from the activation layers. Control experiments from different perspectives, including toxicity reduction, sentiment control, linguistic style, and topic control, demonstrate that our framework achieves state-of-the-art performance with granular control, allowing for fine-grained adjustments of both the steering layers and the steering magnitudes for individual samples. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在各个领域中的广泛应用，对其生成输出的控制能力变得越来越重要。这种控制涉及将LLMs的输出与人类价值观和伦理原则对齐，或者针对特定主题或风格对LLMs进行个性化定制。现有的控制生成方法要么需要大量的计算资源和广泛的试错，要么提供的控制粒度较粗。本文提出了一种名为Concept Activation Vector生成（GCAV）的轻量级模型控制框架，该框架确保了精确控制，而无需进行资源密集型的微调。具体来说，GCAV首先为需要控制的概念（如毒性）训练一个概念激活向量，在推理过程中，GCAV通过从激活层中移除相应概念的向量等方式来引导这些概念。从减少毒性、控制情感、语言风格和主题控制等多个角度进行的控制实验表明，我们的框架实现了最先进的性能，并且具有细粒度控制能力，允许对操作层和控制幅度进行个体样本级别的精细调整。 

---
# Exploring Large Language Models for Translating Romanian Computational Problems into English 

**Title (ZH)**: 探索大型语言模型在将罗马尼亚计算问题翻译成英语中的应用 

**Authors**: Adrian Marius Dumitran, Adrian-Catalin Badea, Stefan-Gabriel Muscalu, Angela-Liliana Dumitran, Stefan-Cosmin Dascalescu, Radu-Sebastian Amarie  

**Link**: [PDF](https://arxiv.org/pdf/2501.05601)  

**Abstract**: Recent studies have suggested that large language models (LLMs) underperform on mathematical and computer science tasks when these problems are translated from Romanian into English, compared to their original Romanian format. Accurate translation is critical for applications ranging from automatic translations in programming competitions to the creation of high-quality educational materials, as well as minimizing errors or fraud in human translations. This study shows that robust large language models (LLMs) can maintain or even enhance their performance in translating less common languages when given well-structured prompts. Our findings suggest that LLMs, with appropriate supervision, can be reliably used for the automatic translation of IOI (International Olympiad in Informatics)-style tasks. We evaluate several translation methods across multiple LLMs, including OpenRoLLM, Llama 3.1 8B, Llama 3.2 3B and GPT-4o, assessing their translation accuracy and performance stability through repeated runs. Additionally, we augment the OJI (Romanian County-Level Informatics Olympiad) Romanian dataset with accurate English translations, enhancing its utility for future LLM training and evaluation. Through detailed syntactic and semantic analyses, we confirm that with human oversight, LLMs can serve as a viable solution for multilingual problem-solving. We also compare the translation quality of LLMs against human translators, as evaluated by a certified expert, underscoring the potential of LLMs in realworld scenarios. 

**Abstract (ZH)**: 近年来的研究表明，当数学和计算机科学问题从罗马尼亚语翻译成英语时，大型语言模型（LLMs）的表现逊于其原始的罗马尼亚语格式。准确的翻译对于编程竞赛中的自动翻译、高质量教育资源的创建以及减少人工翻译中的错误或欺诈至关重要。本研究展示了，在给定良好结构的提示时，强大的LLMs可以在翻译较不常见的语言时维持甚至增强其性能。我们的研究结果表明，在适当的监督下，LLMs可以可靠地用于IOI（国际信息学奥林匹克）风格任务的自动翻译。我们评估了几种翻译方法在多个LLMs（包括OpenRoLLM、Llama 3.1 8B、Llama 3.2 3B和GPT-4o）上的表现，通过多次运行评估其翻译准确性和性能稳定性。此外，我们还扩展了OJI（罗马尼亚县级信息学奥林匹克）的罗马尼亚语数据集，增加了准确的英语翻译，增强了其在未来LLM训练和评估中的实用性。通过详细的句法和语义分析，我们确认，在人类监督下，LLMs可以作为一种可行的多语言问题解决解决方案。我们也对比了LLMs与专业人工译者的翻译质量，强调了LLMs在实际应用场景中的潜力。 

---
# The Future of AI: Exploring the Potential of Large Concept Models 

**Title (ZH)**: 人工智能的未来：探究大型概念模型的潜力 

**Authors**: Hussain Ahmad, Diksha Goel  

**Link**: [PDF](https://arxiv.org/pdf/2501.05487)  

**Abstract**: The field of Artificial Intelligence (AI) continues to drive transformative innovations, with significant progress in conversational interfaces, autonomous vehicles, and intelligent content creation. Since the launch of ChatGPT in late 2022, the rise of Generative AI has marked a pivotal era, with the term Large Language Models (LLMs) becoming a ubiquitous part of daily life. LLMs have demonstrated exceptional capabilities in tasks such as text summarization, code generation, and creative writing. However, these models are inherently limited by their token-level processing, which restricts their ability to perform abstract reasoning, conceptual understanding, and efficient generation of long-form content. To address these limitations, Meta has introduced Large Concept Models (LCMs), representing a significant shift from traditional token-based frameworks. LCMs use concepts as foundational units of understanding, enabling more sophisticated semantic reasoning and context-aware decision-making. Given the limited academic research on this emerging technology, our study aims to bridge the knowledge gap by collecting, analyzing, and synthesizing existing grey literature to provide a comprehensive understanding of LCMs. Specifically, we (i) identify and describe the features that distinguish LCMs from LLMs, (ii) explore potential applications of LCMs across multiple domains, and (iii) propose future research directions and practical strategies to advance LCM development and adoption. 

**Abstract (ZH)**: 人工智能（AI）领域继续推动着变革性创新，尤其是在对话界面、自动驾驶车辆和智能内容生成方面取得了显著进展。自2022年底ChatGPT发布以来，生成型AI的兴起标志着一个关键时期的到来，大型语言模型（LLMs）这一术语已成为日常生活的一部分。LLMs在文本总结、代码生成和创意写作等任务中展示了卓越的能力。然而，这些模型受到其基于令牌级别的处理的内在限制，这限制了它们进行抽象推理、概念理解以及高效生成长篇内容的能力。为了克服这些限制，Meta引入了大型概念模型（LCMs），代表了从传统基于令牌框架到基于概念框架的重大转变。LCMs以概念作为理解的基本单位，能够实现更高级的语义推理和上下文感知决策。由于对该新兴技术的研究有限，我们的研究旨在通过收集、分析和综合现有灰色文献来弥合知识差距，以全面理解LCMs。具体来说，我们将（i）识别并描述区分LCMs和LLMs的特征，（ii）探索LCMs在多个领域的潜在应用，以及（iii）提出未来的研究方向和实用策略，以促进LCMs的发展和应用。 

---
# HP-BERT: A framework for longitudinal study of Hinduphobia on social media via LLMs 

**Title (ZH)**: HP-BERT：通过大语言模型 longitudinally 研究社交媒体上 Hinduphobia 的框架 

**Authors**: Ashutosh Singh, Rohitash Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2501.05482)  

**Abstract**: During the COVID-19 pandemic, community tensions intensified, fuelling Hinduphobic sentiments and discrimination against individuals of Hindu descent within India and worldwide. Large language models (LLMs) have become prominent in natural language processing (NLP) tasks and social media analysis, enabling longitudinal studies of platforms like X (formerly Twitter) for specific issues during COVID-19. We present an abuse detection and sentiment analysis framework that offers a longitudinal analysis of Hinduphobia on X (Twitter) during and after the COVID-19 pandemic. This framework assesses the prevalence and intensity of Hinduphobic discourse, capturing elements such as derogatory jokes and racist remarks through sentiment analysis and abuse detection from pre-trained and fine-tuned LLMs. Additionally, we curate and publish a "Hinduphobic COVID-19 X (Twitter) Dataset" of 8,000 tweets annotated for Hinduphobic abuse detection, which is used to fine-tune a BERT model, resulting in the development of the Hinduphobic BERT (HP-BERT) model. We then further fine-tune HP-BERT using the SenWave dataset for multi-label sentiment analysis. Our study encompasses approximately 27.4 million tweets from six countries, including Australia, Brazil, India, Indonesia, Japan, and the United Kingdom. Our findings reveal a strong correlation between spikes in COVID-19 cases and surges in Hinduphobic rhetoric, highlighting how political narratives, misinformation, and targeted jokes contributed to communal polarisation. These insights provide valuable guidance for developing strategies to mitigate communal tensions in future crises, both locally and globally. We advocate implementing automated monitoring and removal of such content on social media to curb divisive discourse. 

**Abstract (ZH)**: 在COVID-19疫情期间，社区紧张局势加剧，促进了对印度教徒的恐惧情绪和针对印度教背景个体的歧视。大规模语言模型（LLMs）在自然语言处理（NLP）任务和社会媒体分析中发挥了重要作用，使我们能够对诸如X（原名Twitter）等平台在COVID-19疫情期间的特定问题进行纵向研究。我们提出了一种虐待检测和情感分析框架，该框架对X（Twitter）上的印度教恐惧主义进行了纵向分析，涵盖疫情之前和之后的时期。该框架评估了印度教恐惧主义言论的普遍性和强度，通过情感分析和虐待检测利用预训练和微调的LLMs捕捉贬低的笑话和种族主义评论等元素。此外，我们整理并发布了包含8,000条标注了印度教恐惧主义虐待的“印度教恐惧主义COVID-19 X（Twitter）数据集”，用以微调BERT模型，从而开发出印度教恐惧主义BERT（HP-BERT）模型。随后，我们使用SenWave数据集进一步微调HP-BERT，进行多标签情感分析。本研究涵盖了来自六个不同国家（澳大利亚、巴西、印度、印度尼西亚、日本和英国）共计约2740万条推特。研究发现表明，在COVID-19病例激增期间印度教恐惧主义言论的激增之间存在强烈关联，突显了政治叙事、虚假信息和针对印度教徒的针对性笑话在引发社区分裂方面的角色。这些见解提供了宝贵的信息，为企业和组织在未来的危机中制定缓解社区紧张局势的战略提供了指导，无论是在本地还是全球范围内。我们建议实施自动化监控和移除此类内容，以遏制分化性言论。 

---
# LatteReview: A Multi-Agent Framework for Systematic Review Automation Using Large Language Models 

**Title (ZH)**: LatteReview：使用大规模语言模型进行系统评价自动化的一种多代理框架 

**Authors**: Pouria Rouzrokh, Moein Shariatnia  

**Link**: [PDF](https://arxiv.org/pdf/2501.05468)  

**Abstract**: Systematic literature reviews and meta-analyses are essential for synthesizing research insights, but they remain time-intensive and labor-intensive due to the iterative processes of screening, evaluation, and data extraction. This paper introduces and evaluates LatteReview, a Python-based framework that leverages large language models (LLMs) and multi-agent systems to automate key elements of the systematic review process. Designed to streamline workflows while maintaining rigor, LatteReview utilizes modular agents for tasks such as title and abstract screening, relevance scoring, and structured data extraction. These agents operate within orchestrated workflows, supporting sequential and parallel review rounds, dynamic decision-making, and iterative refinement based on user feedback. LatteReview's architecture integrates LLM providers, enabling compatibility with both cloud-based and locally hosted models. The framework supports features such as Retrieval-Augmented Generation (RAG) for incorporating external context, multimodal reviews, Pydantic-based validation for structured inputs and outputs, and asynchronous programming for handling large-scale datasets. The framework is available on the GitHub repository, with detailed documentation and an installable package. 

**Abstract (ZH)**: 系统文献综述和元分析是综合研究洞察力的重要工具，但由于筛选、评估和数据提取的迭代过程，它们仍耗时且劳动密集。本文介绍了并评估了LatteReview，这是一个基于Python的框架，利用大型语言模型（LLMs）和多智能体系统自动化系统综述过程中的关键要素。设计时旨在简化工作流程同时保持严格性，LatteReview使用模块化的代理程序来执行诸如标题和摘要筛选、相关性评分和结构化数据提取等任务。这些代理程序在协调的流程中运作，支持顺序和并行的审查轮次、动态决策和基于用户反馈的迭代改进。LatteReview的架构整合了LLM提供者，使得既兼容基于云的模型也兼容本地托管的模型。该框架支持诸如检索增强生成（RAG）、多模态审查、基于Pydantic的验证以确保输入和输出的结构化，以及异步编程以处理大规模数据集等功能。该框架在GitHub存储库中可用，配备了详细的文档和可安装的包。 

---
