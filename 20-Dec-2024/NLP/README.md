# LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks 

**Title (ZH)**: 长 Bench v2：在现实长上下文多任务理解与推理方面的更深入探究 

**Authors**: Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xiaozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu, Lei Hou, Yuxiao Dong, Jie Tang, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.15204)  

**Abstract**: This paper introduces LongBench v2, a benchmark designed to assess the ability of LLMs to handle long-context problems requiring deep understanding and reasoning across real-world multitasks. LongBench v2 consists of 503 challenging multiple-choice questions, with contexts ranging from 8k to 2M words, across six major task categories: single-document QA, multi-document QA, long in-context learning, long-dialogue history understanding, code repository understanding, and long structured data understanding. To ensure the breadth and the practicality, we collect data from nearly 100 highly educated individuals with diverse professional backgrounds. We employ both automated and manual review processes to maintain high quality and difficulty, resulting in human experts achieving only 53.7% accuracy under a 15-minute time constraint. Our evaluation reveals that the best-performing model, when directly answers the questions, achieves only 50.1% accuracy. In contrast, the o1-preview model, which includes longer reasoning, achieves 57.7%, surpassing the human baseline by 4%. These results highlight the importance of enhanced reasoning ability and scaling inference-time compute to tackle the long-context challenges in LongBench v2. The project is available at this https URL. 

**Abstract (ZH)**: 以下是翻译后的论文内容或标题，符合学术规范：

本文介绍了LongBench v2，一个用于评估大模型处理需要深入理解和推理的长上下文问题的基准测试。LongBench v2 包含503个具有挑战性的选择题，上下文长度从8,000词到2,000,000词不等，涵盖六个主要任务类别：单文档问答（Single-document QA）、多文档问答（Multi-document QA）、长上下文学习（Long in-context learning）、长期对话历史理解（Long-dialogue history understanding）、代码库理解（Code repository understanding）和长结构化数据理解（Long structured data understanding）。为了确保覆盖面和实用性，我们从近100名背景多样的高学历人士中收集数据。我们采用自动和手动审查流程，确保高质量和难度，最终结果显示，即使在15分钟的时间限制内，人类专家的准确率也仅为53.7%。我们的评估表明，当最优秀的模型直接回答问题时，其准确率仅为50.1%。相比之下，o1-preview模型，该模型包含更长的推理过程，达到了57.7%的准确率，比人类基准高出4%。这些结果强调了增强推理能力和扩展推理时的计算资源对于解决LongBench v2中的长上下文挑战的重要性。该项目的详细内容可在以下网址获取：[此 https URLs]。 

---
# MMLU-CF: A Contamination-free Multi-task Language Understanding Benchmark 

**Title (ZH)**: MMLU-CF：一个无污染的多任务语言理解基准测试 

**Authors**: Qihao Zhao, Yangyu Huang, Tengchao Lv, Lei Cui, Qinzheng Sun, Shaoguang Mao, Xin Zhang, Ying Xin, Qiufeng Yin, Scarlett Li, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2412.15194)  

**Abstract**: Multiple-choice question (MCQ) datasets like Massive Multitask Language Understanding (MMLU) are widely used to evaluate the commonsense, understanding, and problem-solving abilities of large language models (LLMs). However, the open-source nature of these benchmarks and the broad sources of training data for LLMs have inevitably led to benchmark contamination, resulting in unreliable evaluation results. To alleviate this issue, we propose a contamination-free and more challenging MCQ benchmark called MMLU-CF. This benchmark reassesses LLMs' understanding of world knowledge by averting both unintentional and malicious data leakage. To avoid unintentional data leakage, we source data from a broader domain and design three decontamination rules. To prevent malicious data leakage, we divide the benchmark into validation and test sets with similar difficulty and subject distributions. The test set remains closed-source to ensure reliable results, while the validation set is publicly available to promote transparency and facilitate independent verification. Our evaluation of mainstream LLMs reveals that the powerful GPT-4o achieves merely a 5-shot score of 73.4% and a 0-shot score of 71.9% on the test set, which indicates the effectiveness of our approach in creating a more rigorous and contamination-free evaluation standard. The GitHub repository is available at this https URL and the dataset refers to this https URL. 

**Abstract (ZH)**: 多选题数据集（MCQ数据集），如大规模多任务语言理解（MMLU），广泛用于评估大型语言模型（LLMs）的常识、理解和问题解决能力。然而，这些基准的开源性质以及LLMs广泛的数据来源不可避免地导致了基准数据的污染，从而产生了不可靠的评估结果。为了解决这一问题，我们提出了一种无污染且更具挑战性的MCQ基准——MMLU-CF。该基准通过避免无意和恶意的数据泄漏，重新评估LLMs对世界知识的理解能力。为了防止无意数据泄漏，我们从更广泛的领域获取数据，并设计了三个去污染规则。为了防止恶意数据泄漏，我们将基准划分为具有类似难度和主题分布的验证集和测试集。测试集保持封闭源代码状态，以确保结果的可靠性，验证集则公开发布，促进透明度并方便独立验证。我们的评估显示，强大的GPT-4o在测试集上的5-shot得分为73.4%，0-shot得分为71.9%，这表明了我们方法在建立更为严格和无污染的评价标准方面的有效性。GitHub仓库地址为：[这里](https://github.com/)，数据集请参考：[这里](https://dataset.com/)。 

---
# Face the Facts! Evaluating RAG-based Fact-checking Pipelines in Realistic Settings 

**Title (ZH)**: 正视现实！在实际场景中评估基于RAG的事实核查管道 

**Authors**: Daniel Russo, Stefano Menini, Jacopo Staiano, Marco Guerini  

**Link**: [PDF](https://arxiv.org/pdf/2412.15189)  

**Abstract**: Natural Language Processing and Generation systems have recently shown the potential to complement and streamline the costly and time-consuming job of professional fact-checkers. In this work, we lift several constraints of current state-of-the-art pipelines for automated fact-checking based on the Retrieval-Augmented Generation (RAG) paradigm. Our goal is to benchmark, under more realistic scenarios, RAG-based methods for the generation of verdicts - i.e., short texts discussing the veracity of a claim - evaluating them on stylistically complex claims and heterogeneous, yet reliable, knowledge bases. Our findings show a complex landscape, where, for example, LLM-based retrievers outperform other retrieval techniques, though they still struggle with heterogeneous knowledge bases; larger models excel in verdict faithfulness, while smaller models provide better context adherence, with human evaluations favouring zero-shot and one-shot approaches for informativeness, and fine-tuned models for emotional alignment. 

**Abstract (ZH)**: 自然语言处理（NLP）和生成系统在最近显示出有望补充和简化专业事实核查人员繁重且耗时的工作。在本文中，我们基于检索增强生成（RAG）范式，改进了现有最先进的自动事实核查流程中的多项约束。我们的目标是在更现实的情景下对基于RAG的方法进行基准测试，这些方法用于生成裁定——即讨论声明真伪的简短文本——并通过风格复杂且多样的知识库对其进行评估。我们的研究发现了一个复杂的情景，例如，基于大型语言模型（LLM）的检索器在检索性能上优于其他检索技术，尽管它们仍然难以处理异质性知识库；较大的模型在裁定的可信度方面表现出色，而较小的模型在保持上下文一致性方面表现更好。从人类评价来看，零样本和单样本方法在信息丰富性方面受到青睐，而微调模型在情感一致性方面表现更好。 

---
# LlamaFusion: Adapting Pretrained Language Models for Multimodal Generation 

**Title (ZH)**: LlamaFusion：适配预训练语言模型的多模态生成 

**Authors**: Weijia Shi, Xiaochuang Han, Chunting Zhou, Weixin Liang, Xi Victoria Lin, Luke Zettlemoyer, Lili Yu  

**Link**: [PDF](https://arxiv.org/pdf/2412.15188)  

**Abstract**: We present LlamaFusion, a framework for empowering pretrained text-only large language models (LLMs) with multimodal generative capabilities, enabling them to understand and generate both text and images in arbitrary sequences. LlamaFusion leverages existing Llama-3's weights for processing texts autoregressively while introducing additional and parallel transformer modules for processing images with diffusion. During training, the data from each modality is routed to its dedicated modules: modality-specific feedforward layers, query-key-value projections, and normalization layers process each modality independently, while the shared self-attention layers allow interactions across text and image features. By freezing the text-specific modules and only training the image-specific modules, LlamaFusion preserves the language capabilities of text-only LLMs while developing strong visual understanding and generation abilities. Compared to methods that pretrain multimodal generative models from scratch, our experiments demonstrate that, LlamaFusion improves image understanding by 20% and image generation by 3.6% using only 50% of the FLOPs while maintaining Llama-3's language capabilities. We also demonstrate that this framework can adapt existing vision-language models with multimodal generation ability. Overall, this framework not only leverages existing computational investments in text-only LLMs but also enables the parallel development of language and vision capabilities, presenting a promising direction for efficient multimodal model development. 

**Abstract (ZH)**: 我们提出了LlamaFusion框架，该框架赋予了仅限文本的大规模语言模型（LLMs）多模态生成能力，使其能够理解和生成任意序列中的文本和图像。LlamaFusion利用了现有Llama-3的权重进行文本的自回归处理，同时引入了额外的并行变压器模块来处理通过扩散方法处理的图像。在训练过程中，每个模态的数据都被导向其专用模块：特定于模态的前馈层、查询-键-值投影以及归一化层独立处理每个模态，而共享的自注意力层则允许跨文本和图像特征的交互。通过冻结特定于文本的模块，仅训练特定于图像的模块，LlamaFusion保留了仅文本LLM的语言能力，同时增强了其强大的视觉理解和生成能力。与从头开始预训练多模态生成模型的方法相比，我们的实验表明，LlamaFusion仅使用50%的FLOPs，就能提高图像理解20%和图像生成3.6%，同时保留Llama-3的语言能力。此外，我们展示了该框架可以适应现有的具备多模态生成能力的视觉语言模型。总体而言，该框架不仅利用了仅文本LLM的现有计算投资，还促进了语言和视觉能力的并行发展，为高效的多模态模型开发提供了有希望的方向。 

---
# Language Models as Continuous Self-Evolving Data Engineers 

**Title (ZH)**: 语言模型作为连续自进化的数据工程师 

**Authors**: Peidong Wang, Ming Wang, Zhiming Ma, Xiaocui Yang, Shi Feng, Daling Wang, Yifei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.15151)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities on various tasks, while the further evolvement is limited to the lack of high-quality training data. In addition, traditional training approaches rely too much on expert-labeled data, setting an upper limit on the performance of LLMs. To address this issue, we propose a novel paradigm that enables LLMs to train itself by autonomously generating, cleaning, reviewing, and annotating data with preference information, named LANCE. Our approach demonstrates that LLMs can serve as continuous self-evolving data engineers, significantly reducing the time and cost of the post-training data construction process. Through iterative fine-tuning on different variants of the Qwen2, we validate the effectiveness of LANCE across various tasks, showing that it can continuously improve model performance and maintain high-quality data generation. Across eight benchmark dimensions, LANCE resulted in an average score enhancement of 3.36 for Qwen2-7B and 2.70 for Qwen2-7B-Instruct. This training paradigm with autonomous data construction not only reduces the reliance on human experts or external models but also ensures that the data aligns with human values and preferences, paving the way for the development of future superintelligent systems that can exceed human capabilities. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在多种任务上展示了卓越的能力，但进一步的发展受限于高质量训练数据的缺乏。此外，传统的训练方法过度依赖于专家标记的数据，从而限制了LLMs的性能上限。为解决这一问题，我们提出了一种新的范式，使LLMs能够在自主生成、清理、审核和注释数据（含偏好信息）的过程中自我训练，这一范式命名为LANCE。我们的方法证明了LLMs可以作为持续自我进化的数据工程师，显著减少了后训练数据构建过程的时间和成本。通过在Qwen2的不同变体上进行迭代微调，我们验证了LANCE在各种任务中的有效性，展示了其对模型性能的持续改进和高质量数据生成的维持能力。在八个基准维度上，LANCE分别提高了Qwen2-7B和Qwen2-7B-Instruct的平均分数3.36和2.70。这种自主数据构建的训练范式不仅减少了对人类专家或外部模型的依赖，还确保了数据与人类价值观和偏好保持一致，为开发未来超越人类能力的超级智能系统奠定了基础。 

---
# Adaptive Pruning for Large Language Models with Structural Importance Awareness 

**Title (ZH)**: 具有结构重要性意识的大型语言模型自适应剪枝 

**Authors**: Haotian Zheng, Jinke Ren, Yushan Sun, Ruichen Zhang, Wenbo Zhang, Zhen Li, Dusit Niyato, Shuguang Cui, Yatong Han  

**Link**: [PDF](https://arxiv.org/pdf/2412.15127)  

**Abstract**: The recent advancements in large language models (LLMs) have significantly improved language understanding and generation capabilities. However, it is difficult to deploy LLMs on resource-constrained edge devices due to their high computational and storage resource demands. To address this issue, we propose a novel LLM model pruning method, namely structurally-aware adaptive pruning (SAAP), to significantly reduce the computational and memory costs while maintaining model performance. We first define an adaptive importance fusion metric to evaluate the importance of all coupled structures in LLMs by considering their homoscedastic uncertainty. Then, we rank the importance of all modules to determine the specific layers that should be pruned to meet particular performance requirements. Furthermore, we develop a new group fine-tuning strategy to improve the inference efficiency of LLMs. Finally, we evaluate the proposed SAAP method on multiple LLMs across two common tasks, i.e., zero-shot classification and text generation. Experimental results show that our SAAP method outperforms several state-of-the-art baseline methods, achieving 2.17%, 2.37%, and 2.39% accuracy gains on LLaMA-7B, Vicuna-7B, and LLaMA-13B. Additionally, SAAP improves the token generation speed by 5%, showcasing its practical advantages in resource-constrained scenarios. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）在语言理解和生成能力方面取得了显著进步。然而，由于LLMs对计算和存储资源的要求较高，将其部署在资源受限的边缘设备上颇具挑战性。为应对这一问题，我们提出了一种新的LLM模型剪枝方法，即结构感知自适应剪枝（SAAP），旨在大幅降低计算和内存成本，同时保持模型性能。首先，我们定义了一个自适应重要性融合度量方法，通过考虑耦合结构的同方差不确定性来评估LLMs中所有耦合结构的重要性。然后，我们根据各模块的重要性对所有模块进行排序，以确定应剪枝的具体层，以满足特定性能要求。此外，我们开发了一种新的分组微调策略，以提高LLMs的推理效率。最后，我们在两种常见任务——零样本分类和文本生成——上对多个LLMs进行了SAAP方法的评估。实验结果表明，我们的SAAP方法在多个模型上都优于几种最先进的基准方法，在LLaMA-7B、Vicuna-7B和LLaMA-13B上分别实现了2.17%、2.37%和2.39%的准确率提升。此外，SAAP还使令牌生成速度提高了5%，展示了其在资源受限场景中的实用优势。 

---
# Outcome-Refining Process Supervision for Code Generation 

**Title (ZH)**: 代码生成中的目标细化过程监督 

**Authors**: Zhuohao Yu, Weizheng Gu, Yidong Wang, Zhengran Zeng, Jindong Wang, Wei Ye, Shikun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.15118)  

**Abstract**: Large Language Models have demonstrated remarkable capabilities in code generation, yet they often struggle with complex programming tasks that require deep algorithmic reasoning. While process supervision through learned reward models shows promise in guiding reasoning steps, it requires expensive training data and suffers from unreliable evaluation. We propose Outcome-Refining Process Supervision, a novel paradigm that treats outcome refinement itself as the process to be supervised. Our framework leverages concrete execution signals to ground the supervision of reasoning steps, while using tree-structured exploration to maintain multiple solution trajectories simultaneously. Experiments demonstrate that our approach enables even smaller models to achieve high success accuracy and performance metrics on competitive programming tasks, creates more reliable verification than traditional reward models without requiring training PRMs. Our approach achieves significant improvements across 5 models and 3 datasets: an average of 26.9% increase in correctness and 42.2% in efficiency. The results suggest that providing structured reasoning space with concrete verification signals is crucial for solving complex programming tasks. We open-source all our code and data at: this https URL 

**Abstract (ZH)**: 大规模语言模型在代码生成方面展示了令人瞩目的能力，但在处理需要深入算法推理的复杂编程任务时往往表现不佳。虽然通过学习奖励模型的过程监督有潜力引导推理步骤，但这种方法需要昂贵的训练数据，并且在评估的可靠性方面存在问题。我们提出了一种名为Outcome-Refining Process Supervision的新范式，将结果细化本身作为需要监督的过程。我们的框架利用具体的执行信号将监督聚焦于推理步骤，同时使用基于树结构的探索来同时维持多个解决方案轨迹。实验表明，我们的方法使较小的模型能够在竞争编程任务上实现高准确性和性能指标，在无需训练SRMs的情况下提供了比传统奖励模型更可靠的验证。我们在5个模型和3个数据集上取得显著改进：平均正确性提高26.9%，效率提高42.2%。结果表明，提供结构化的推理空间并给出具体的验证信号对于解决复杂编程任务至关重要。我们已将所有代码和数据开源，访问链接如下：this https URL 

---
# Qwen2.5 Technical Report 

**Title (ZH)**: Qwen2.5 技术报告 

**Authors**: Qwen, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zihan Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2412.15115)  

**Abstract**: In this report, we introduce Qwen2.5, a comprehensive series of large language models (LLMs) designed to meet diverse needs. Compared to previous iterations, Qwen 2.5 has been significantly improved during both the pre-training and post-training stages. In terms of pre-training, we have scaled the high-quality pre-training datasets from the previous 7 trillion tokens to 18 trillion tokens. This provides a strong foundation for common sense, expert knowledge, and reasoning capabilities. In terms of post-training, we implement intricate supervised finetuning with over 1 million samples, as well as multistage reinforcement learning. Post-training techniques enhance human preference, and notably improve long text generation, structural data analysis, and instruction following. To handle diverse and varied use cases effectively, we present Qwen2.5 LLM series in rich sizes. Open-weight offerings include base and instruction-tuned models, with quantized versions available. In addition, for hosted solutions, the proprietary models currently include two mixture-of-experts (MoE) variants: Qwen2.5-Turbo and Qwen2.5-Plus, both available from Alibaba Cloud Model Studio. Qwen2.5 has demonstrated top-tier performance on a wide range of benchmarks evaluating language understanding, reasoning, mathematics, coding, human preference alignment, etc. Specifically, the open-weight flagship Qwen2.5-72B-Instruct outperforms a number of open and proprietary models and demonstrates competitive performance to the state-of-the-art open-weight model, Llama-3-405B-Instruct, which is around 5 times larger. Qwen2.5-Turbo and Qwen2.5-Plus offer superior cost-effectiveness while performing competitively against GPT-4o-mini and GPT-4o respectively. Additionally, as the foundation, Qwen2.5 models have been instrumental in training specialized models such as Qwen2.5-Math, Qwen2.5-Coder, QwQ, and multimodal models. 

**Abstract (ZH)**: 在本报告中，我们介绍了Qwen 2.5，这是一个全面的大型语言模型（LLM）系列，旨在满足各种需求。与之前的版本相比，Qwen 2.5 在预训练和后训练阶段均得到了显著改进。在预训练方面，我们将高质量的预训练数据集从之前的7万亿个 token 扩大到18万亿个 token。这为常识、专家知识和推理能力提供了坚实的基础。在后训练方面，我们实施了复杂的监督微调，涉及超过100万的样本，以及多阶段强化学习。后训练技术增强了对人类偏好的理解，并显著提高了长文本生成、结构化数据分析和指令遵循的能力。为了有效应对各种各样的应用场景，我们推出了丰富规格的Qwen 2.5 LLM系列。开放模型包括基础模型和指令调优模型，并提供量化版本。此外，对于托管解决方案，当前的自研模型包括两种专家混合模型（MoE）变体：Qwen 2.5 Turbo 和 Qwen 2.5 Plus，两者均来自阿里云模型工作室。Qwen 2.5 在广泛的语言理解、推理、数学、编程、人类偏好对齐等基准测试中展现了顶尖的性能。特别是开放模型Qwen 2.5-72B-Instruct，在多项测试中均表现优异，优于多个开放和自研模型，并与更大的开源模型Llama-3-405B-Instruct（约大5倍）性能相当。Qwen 2.5 Turbo 和 Qwen 2.5 Plus 提供了更高的性价比，性能分别与GPT-4o-mini 和 GPT-4o 竞争。此外，作为基础模型，Qwen 2.5 还被用于训练各种专业模型，如Qwen 2.5-Math、Qwen 2.5-Coder、QwQ 和多模态模型。 

---
# Review-Then-Refine: A Dynamic Framework for Multi-Hop Question Answering with Temporal Adaptability 

**Title (ZH)**: 基于回顾与精炼的动态框架：具有时间适应性的多跳问答系统 

**Authors**: Xiangsen Chen, Xuming Hu, Nan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2412.15101)  

**Abstract**: Retrieve-augmented generation (RAG) frameworks have emerged as a promising solution to multi-hop question answering(QA) tasks since it enables large language models (LLMs) to incorporate external knowledge and mitigate their inherent knowledge deficiencies. Despite this progress, existing RAG frameworks, which usually follows the retrieve-then-read paradigm, often struggle with multi-hop QA with temporal information since it has difficulty retrieving and synthesizing accurate time-related information. To address the challenge, this paper proposes a novel framework called review-then-refine, which aims to enhance LLM performance in multi-hop QA scenarios with temporal information. Our approach begins with a review phase, where decomposed sub-queries are dynamically rewritten with temporal information, allowing for subsequent adaptive retrieval and reasoning process. In addition, we implement adaptive retrieval mechanism to minimize unnecessary retrievals, thus reducing the potential for hallucinations. In the subsequent refine phase, the LLM synthesizes the retrieved information from each sub-query along with its internal knowledge to formulate a coherent answer. Extensive experimental results across multiple datasets demonstrate the effectiveness of our proposed framework, highlighting its potential to significantly improve multi-hop QA capabilities in LLMs. 

**Abstract (ZH)**: 检索增强生成（RAG）框架已成为了多跳问答（QA）任务的一种有前景的解决方案，因为它使大型语言模型（LLMs）能够结合外部知识并缓解其固有的知识缺陷。尽管取得了这些进展，现有的RAG框架通常遵循检索-然后-阅读的范式，这在处理包含时间信息的多跳问答时往往表现不佳，因为它们难以检索和合成准确的时间相关信息。为了解决这一挑战，本文提出了一种新的框架，称为审查-然后-精炼（Review-then-Refine），旨在增强LLMs在包含时间信息的多跳问答场景中的性能。我们的方法首先进入一个审查阶段，在这个阶段中，分解后的子查询会根据不同情况动态重写以包含时间信息，从而允许后续的自适应检索和推理过程。此外，我们实现了一种自适应检索机制，以最大限度地减少不必要的检索，从而减少幻觉的可能性。在随后的精炼阶段，LLM结合从每个子查询中检索到的信息及其内部知识来构建一个连贯的答案。在多个数据集上的广泛实验结果表明了我们提出框架的有效性，并突显了它在增强LLMs的多跳问答能力方面具有巨大潜力。 

---
# AceMath: Advancing Frontier Math Reasoning with Post-Training and Reward Modeling 

**Title (ZH)**: AceMath: 通过后训练和奖励建模推进前沿数学推理 

**Authors**: Zihan Liu, Yang Chen, Mohammad Shoeybi, Bryan Catanzaro, Wei Ping  

**Link**: [PDF](https://arxiv.org/pdf/2412.15084)  

**Abstract**: In this paper, we introduce AceMath, a suite of frontier math models that excel in solving complex math problems, along with highly effective reward models capable of evaluating generated solutions and reliably identifying the correct ones. To develop the instruction-tuned math models, we propose a supervised fine-tuning (SFT) process that first achieves competitive performance across general domains, followed by targeted fine-tuning for the math domain using a carefully curated set of prompts and synthetically generated responses. The resulting model, AceMath-72B-Instruct greatly outperforms Qwen2.5-Math-72B-Instruct, GPT-4o and Claude-3.5 Sonnet. To develop math-specialized reward model, we first construct AceMath-RewardBench, a comprehensive and robust benchmark for evaluating math reward models across diverse problems and difficulty levels. After that, we present a systematic approach to build our math reward models. The resulting model, AceMath-72B-RM, consistently outperforms state-of-the-art reward models. Furthermore, when combining AceMath-72B-Instruct with AceMath-72B-RM, we achieve the highest average rm@8 score across the math reasoning benchmarks. We will release model weights, training data, and evaluation benchmarks at: this https URL 

**Abstract (ZH)**: 在本文中，我们介绍了AceMath，这是一个卓越于解决复杂数学问题的一系列前沿数学模型，以及高效的奖励模型，能够评估生成的解决方案并可靠地识别正确的答案。为了开发指令调优的数学模型，我们提出了一个监督微调（SFT）过程，该过程首先在通用领域实现竞争力表现，然后通过精心策划的一系列提示和合成生成的响应，针对数学领域进行有针对性的微调。最终模型AceMath-72B-Instruct在性能上显著优于Qwen2.5-Math-72B-Instruct、GPT-4o和Claude-3.5 Sonnet。为了开发专门的数学奖励模型，我们首先构建了AceMath-RewardBench，这是一个全面且稳健的基准，用于评估数学奖励模型在各种问题和难度级别上的表现。随后，我们提出了一种系统的方法来构建我们的数学奖励模型。最终模型AceMath-72B-RM在性能上持续优于最先进的奖励模型。此外，当结合使用AceMath-72B-Instruct与AceMath-72B-RM时，我们在数学推理基准测试中实现了最高的平均rm@8分数。我们将在以下网址发布模型权重、训练数据和评估基准：this https URL 

---
# ConfliBERT: A Language Model for Political Conflict 

**Title (ZH)**: 冲突BERT：一个用于政治冲突的语言模型 

**Authors**: Patrick T. Brandt, Sultan Alsarra, Vito J. D`Orazio, Dagmar Heintze, Latifur Khan, Shreyas Meher, Javier Osorio, Marcus Sianan  

**Link**: [PDF](https://arxiv.org/pdf/2412.15060)  

**Abstract**: Conflict scholars have used rule-based approaches to extract information about political violence from news reports and texts. Recent Natural Language Processing developments move beyond rigid rule-based approaches. We review our recent ConfliBERT language model (Hu et al. 2022) to process political and violence related texts. The model can be used to extract actor and action classifications from texts about political conflict. When fine-tuned, results show that ConfliBERT has superior performance in accuracy, precision and recall over other large language models (LLM) like Google's Gemma 2 (9B), Meta's Llama 3.1 (7B), and Alibaba's Qwen 2.5 (14B) within its relevant domains. It is also hundreds of times faster than these more generalist LLMs. These results are illustrated using texts from the BBC, re3d, and the Global Terrorism Dataset (GTD). 

**Abstract (ZH)**: 冲突研究学者利用基于规则的方法从新闻报道和文本中提取关于政治暴力的信息。最近的自然语言处理（NLP）发展超越了刚性基于规则的方法。我们回顾了我们最近开发的ConfliBERT语言模型（Hu等，2022），以处理与政治和暴力相关的文本。该模型可以从关于政治冲突的文本中提取行为者和行为分类。当进行微调后，结果显示，与谷歌的Gemma 2（9B）、Meta的Llama 3.1（7B）和阿里巴巴的Qwen 2.5（14B）等其他大型语言模型相比，ConfliBERT在相关领域内的准确率、精确率和召回率方面表现更优。此外，ConfliBERT比这些更为通用的大规模语言模型快上百倍。这些结果通过使用来自BBC、re3d和全球恐怖主义数据库（GTD）的文本数据进行了说明。 

---
# LLMs Lost in Translation: M-ALERT uncovers Cross-Linguistic Safety Gaps 

**Title (ZH)**: LLMs 在跨语言翻译中迷失：M-ALERT 揭示跨语言安全漏洞 

**Authors**: Felix Friedrich, Simone Tedeschi, Patrick Schramowski, Manuel Brack, Roberto Navigli, Huu Nguyen, Bo Li, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2412.15035)  

**Abstract**: Building safe Large Language Models (LLMs) across multiple languages is essential in ensuring both safe access and linguistic diversity. To this end, we introduce M-ALERT, a multilingual benchmark that evaluates the safety of LLMs in five languages: English, French, German, Italian, and Spanish. M-ALERT includes 15k high-quality prompts per language, totaling 75k, following the detailed ALERT taxonomy. Our extensive experiments on 10 state-of-the-art LLMs highlight the importance of language-specific safety analysis, revealing that models often exhibit significant inconsistencies in safety across languages and categories. For instance, Llama3.2 shows high unsafety in the category crime_tax for Italian but remains safe in other languages. Similar differences can be observed across all models. In contrast, certain categories, such as substance_cannabis and crime_propaganda, consistently trigger unsafe responses across models and languages. These findings underscore the need for robust multilingual safety practices in LLMs to ensure safe and responsible usage across diverse user communities. 

**Abstract (ZH)**: 构建跨多种语言的安全大型语言模型（LLMs）对于确保安全访问和语言多样性至关重要。为此，我们提出了M-ALERT，这是一个多语言基准，评估了五种语言（英语、法语、德语、意大利语和西班牙语）中的LLM的安全性。M-ALERT包括每种语言15,000个高质量的提示，共计75,000个，遵循详细的ALERT分类体系。我们在10个最先进的LLM上的广泛实验强调了语言特定安全性分析的重要性，揭示了模型在不同语言和类别中表现出显著的安全性不一致性。例如，对于意大利语，Llama3.2在“犯罪税收”类别中表现出较高的不安全性，但在其他语言中则保持安全。类似的不同之处在所有模型中都可以观察到。相比之下，某些类别，如“毒品大麻”和“犯罪宣传”，在所有模型和语言中都会引发不安全的响应。这些发现强调了在LLM中实施稳健的多语言安全性实践的重要性，以确保在各种用户社区中的安全和负责任的使用。 

---
# Chain-of-MetaWriting: Linguistic and Textual Analysis of How Small Language Models Write Young Students Texts 

**Title (ZH)**: 链式元写作：小型语言模型撰写学生文本的语言学与文本分析 

**Authors**: Ioana Buhnila, Georgeta Cislaru, Amalia Todirascu  

**Link**: [PDF](https://arxiv.org/pdf/2412.14986)  

**Abstract**: Large Language Models (LLMs) have been used to generate texts in response to different writing tasks: reports, essays, story telling. However, language models do not have a meta-representation of the text writing process, nor inherent communication learning needs, comparable to those of young human students. This paper introduces a fine-grained linguistic and textual analysis of multilingual Small Language Models' (SLMs) writing. With our method, Chain-of-MetaWriting, SLMs can imitate some steps of the human writing process, such as planning and evaluation. We mainly focused on short story and essay writing tasks in French for schoolchildren and undergraduate students respectively. Our results show that SLMs encounter difficulties in assisting young students on sensitive topics such as violence in the schoolyard, and they sometimes use words too complex for the target audience. In particular, the output is quite different from the human produced texts in term of text cohesion and coherence regarding temporal connectors, topic progression, reference. 

**Abstract (ZH)**: 下面是这篇论文内容或标题的中文翻译，符合学术规范：

大型语言模型（LLMs）已经被用于完成多种写作任务：报告、议论文以及故事创作。然而，语言模型缺乏对文本写作过程的元表示，也不具备与年轻学生相似的内在沟通学习需求。本文介绍了对多语言小型语言模型（SLMs）写作的细粒度语言和文本分析。通过我们的方法——链条式元写作（Chain-of-MetaWriting），SLMs可以模仿人类写作过程中的某些步骤，如规划和评估。我们主要关注的是分别面向小学和本科学生的法语文本故事和议论文写作任务。实验结果显示，SLMs在处理暴力等敏感话题时存在困难，并且有时使用的词汇过于复杂，不适合目标受众。特别是，在文本连贯性和一致性等时间连词、主题进展和参照方面，输出与人类生成的文本存在较大差异。 

---
# Knowledge Injection via Prompt Distillation 

**Title (ZH)**: 通过提示蒸馏实现知识注入 

**Authors**: Kalle Kujanpää, Harri Valpola, Alexander Ilin  

**Link**: [PDF](https://arxiv.org/pdf/2412.14964)  

**Abstract**: In many practical applications, large language models (LLMs) need to incorporate new knowledge not present in their pre-training data. The primary methods for this are fine-tuning and retrieval-augmented generation (RAG). Although RAG has emerged as the industry standard for knowledge injection, fine-tuning has not yet achieved comparable success. In this paper, we propose a new fine-tuning technique for learning new knowledge and show that it can reach the performance of RAG. The proposed method is based on the self-distillation approach, which we call prompt distillation. First, we generate question-answer pairs about the new knowledge. Then, we fine-tune a student model on the question-answer pairs to imitate the output distributions of a teacher model, which additionally receives the new knowledge in its prompt. The student model is identical to the teacher, except it is equipped with a LoRA adapter. This training procedure facilitates distilling the new knowledge from the teacher's prompt into the student's weights. 

**Abstract (ZH)**: 在许多实际应用中，大规模语言模型（LLMs）需要融入在其预训练数据中不存在的新知识。目前主要的方法是微调和检索增强生成（RAG）。尽管RAG已成为知识注入的工业标准，但微调尚未取得类似的成功。在本文中，我们提出了一种新的微调技术以学习新知识，并证明这种方法可以达到RAG的性能。所提出的方法基于一种称为提示蒸馏的自蒸馏方法。首先，我们生成关于新知识的问题-答案对。然后，我们在问题-答案对上微调学生模型，使其模仿一个额外在提示中接收新知识的教师模型的输出分布。学生模型与教师模型相同，但配备了LoRA适配器。这种训练过程有助于将教师模型提示中的新知识蒸馏到学生模型的权重中。 

---
# Understanding the Dark Side of LLMs' Intrinsic Self-Correction 

**Title (ZH)**: 理解大语言模型内在自我修正的 dark side 

**Authors**: Qingjie Zhang, Han Qiu, Di Wang, Haoting Qian, Yiming Li, Tianwei Zhang, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2412.14959)  

**Abstract**: Intrinsic self-correction was proposed to improve LLMs' responses via feedback prompts solely based on their inherent capability. However, recent works show that LLMs' intrinsic self-correction fails without oracle labels as feedback prompts. In this paper, we aim to interpret LLMs' intrinsic self-correction for different tasks, especially for those failure cases. By including one simple task and three complex tasks with state-of-the-art (SOTA) LLMs like ChatGPT families (o1, 4o, 3.5-turbo) and Llama families (2-7B, 3-8B, and 3.1-8B), we design three interpretation methods to reveal the dark side of LLMs' intrinsic self-correction. We identify intrinsic self-correction can (1) cause LLMs to waver both intermedia and final answers and lead to prompt bias on simple factual questions; (2) introduce human-like cognitive bias on complex tasks. In light of our findings, we also provide two simple yet effective strategies for alleviation: question repeating and supervised fine-tuning with a few samples. We open-source our work at this https URL. 

**Abstract (ZH)**: 自洽性（Intrinsic self-correction）被提出作为一种通过反馈提示来提升大规模语言模型（LLMs）响应能力的方法，这些反馈提示依赖于模型本身的固有能力。然而，最近的研究表明，没有参照标准（oracle labels）作为反馈时，LLMs的自洽性能失效。在本文中，我们旨在对不同任务中LLMs的自洽性进行解读，特别是针对其失败案例。我们通过引入一个简单的任务和三个复杂任务，并使用如ChatGPT系列模型（o1, 4o, 3.5-turbo）和Llama系列模型（2-7B, 3-8B, 和3.1-8B）等最先进的（SOTA）LLMs，设计了三种解释方法来揭示LLMs自洽性的暗面。我们发现自洽性可以（1）导致LLMs在中间答案和最终答案上摇摆不定，并在简单事实性问题上引发提示偏差；（2）在复杂任务中引入类似人类的认知偏差。基于这些发现，我们还提出了两种简单而有效的缓解策略：重复提问和利用少量样本进行监督微调。我们已将此项工作开源，链接为：https://… 

---
# RobustFT: Robust Supervised Fine-tuning for Large Language Models under Noisy Response 

**Title (ZH)**: RobustFT：在嘈杂响应情况下大型语言模型的鲁棒监督微调方法 

**Authors**: Junyu Luo, Xiao Luo, Kaize Ding, Jingyang Yuan, Zhiping Xiao, Ming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.14922)  

**Abstract**: Supervised fine-tuning (SFT) plays a crucial role in adapting large language models (LLMs) to specific domains or tasks. However, as demonstrated by empirical experiments, the collected data inevitably contains noise in practical applications, which poses significant challenges to model performance on downstream tasks. Therefore, there is an urgent need for a noise-robust SFT framework to enhance model capabilities in downstream tasks. To address this challenge, we introduce a robust SFT framework (RobustFT) that performs noise detection and relabeling on downstream task data. For noise identification, our approach employs a multi-expert collaborative system with inference-enhanced models to achieve superior noise detection. In the denoising phase, we utilize a context-enhanced strategy, which incorporates the most relevant and confident knowledge followed by careful assessment to generate reliable annotations. Additionally, we introduce an effective data selection mechanism based on response entropy, ensuring only high-quality samples are retained for fine-tuning. Extensive experiments conducted on multiple LLMs across five datasets demonstrate RobustFT's exceptional performance in noisy scenarios. 

**Abstract (ZH)**: 监督微调（SFT）在将大规模语言模型（LLMs）适应特定领域或任务方面发挥着关键作用。然而，如实验证明的那样，在实际应用中收集的数据不可避免地包含噪声，这对下游任务中的模型性能构成了重大挑战。因此，迫切需要一种噪声鲁棒的SFT框架来提升下游任务中的模型能力。为应对这一挑战，我们提出了一种鲁棒SFT框架（RobustFT），该框架能够在下游任务数据中进行噪声检测和重新标注。在噪声识别方面，我们的方法采用了一个由增强推理模型构成的多专家协作系统，以实现更好的噪声检测。在去噪阶段，我们利用了一种上下文增强策略，该策略结合了最相关和最可信的知识，并经过仔细评估以生成可靠的注释。此外，我们还引入了一种基于响应熵的有效数据选择机制，确保只有高质量样本用于微调。在五个数据集上的多种LLM上进行的广泛实验表明，RobustFT在噪声环境下的性能优异。 

---
# Dehallucinating Parallel Context Extension for Retrieval-Augmented Generation 

**Title (ZH)**: 去幻化并行上下文扩展以实现检索增强生成

在这个翻译中，“Dehallucinating”被翻译为“去幻化”，“Parallel Context Extension”翻译为“并行上下文扩展”，“Retrieval-Augmented Generation”翻译为“检索增强生成”。这样的翻译既符合学术规范，又能够准确传达原文的意思。 

**Authors**: Zexiong Ma, Shengnan An, Zeqi Lin, Yanzhen Zou, Jian-Guang Lou, Bing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2412.14905)  

**Abstract**: Large language models (LLMs) are susceptible to generating hallucinated information, despite the integration of retrieval-augmented generation (RAG). Parallel context extension (PCE) is a line of research attempting to effectively integrating parallel (unordered) contexts, while it still suffers from hallucinations when adapted to RAG scenarios. In this paper, we propose DePaC (Dehallucinating Parallel Context Extension), which alleviates the hallucination problem with context-aware negative training and information-calibrated aggregation. DePaC is designed to alleviate two types of in-context hallucination: fact fabrication (i.e., LLMs present claims that are not supported by the contexts) and fact omission (i.e., LLMs fail to present claims that can be supported by the contexts). Specifically, (1) for fact fabrication, we apply the context-aware negative training that fine-tunes the LLMs with negative supervisions, thus explicitly guiding the LLMs to refuse to answer when contexts are not related to questions; (2) for fact omission, we propose the information-calibrated aggregation which prioritizes context windows with higher information increment from their contexts. The experimental results on nine RAG tasks demonstrate that DePaC significantly alleviates the two types of hallucination and consistently achieves better performances on these tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）在结合检索增强生成（RAG）技术后，仍然容易生成虚构的信息。平行上下文扩展（PCE）是一系列试图有效整合无序并行上下文的研究，但在适应RAG场景时仍然存在虚构的问题。在本文中，我们提出了一种名为DePaC（Dehallucinating Parallel Context Extension）的方法，通过上下文感知的负样本训练和信息校准聚合来缓解虚构问题。DePaC旨在缓解两种类型的上下文内部虚构：事实伪造（即，LLMs呈现没有得到上下文支持的断言）和事实遗漏（即，LLMs未能呈现可以由上下文支持的断言）。具体而言，（1）对于事实伪造，我们采用上下文感知的负样本训练，通过微调LLMs并提供负监督信息，从而明确引导LLMs在上下文与问题不相关时拒绝回答；（2）对于事实遗漏，我们提出了信息校准聚合方法，优先选择那些从上下文中获得较高信息增量的上下文窗口。在九个RAG任务上的实验结果表明，DePaC在显著缓解这两种类型的虚构问题方面表现出色，并在这些任务上实现了一致的更好性能。 

---
# Why language models collapse when trained on recursively generated text 

**Title (ZH)**: 当使用递归生成的文本进行训练时，为什么语言模型会失效 

**Authors**: Lecheng Wang, Xianjie Shi, Ge Li, Jia Li, Yihong Dong, Xuanming Zhang, Wenpin Jiao, Hong Mei  

**Link**: [PDF](https://arxiv.org/pdf/2412.14872)  

**Abstract**: Language models (LMs) have been widely used to generate text on the Internet. The generated text is often collected into the training corpus of the next generations of LMs. Previous work has experimentally found that LMs collapse when trained on recursively generated text. This paper contributes to existing knowledge from two aspects. We present a theoretical proof of LM collapse. Our proof reveals the cause of LM collapse and proves that all auto-regressive LMs will definitely collapse. We present a new finding: the performance of LMs gradually declines when trained on recursively generated text until they perform no better than a randomly initialized LM. The trained LMs produce large amounts of repetitive text and perform poorly across a wide range of natural language tasks. The above proof and new findings deepen our understanding of LM collapse and offer valuable insights that may inspire new training techniques to mitigate this threat. 

**Abstract (ZH)**: 语言模型（LMs）已被广泛用于生成互联网上的文本。生成的文本通常被收集到下一代LMs的训练语料库中。先前的工作通过实验发现，LMs在训练过程中对递归生成的文本会崩溃。本文从两个方面为现有知识做出了贡献。我们给出了LM崩溃的理论证明。我们的证明揭示了LM崩溃的原因，并证明了所有自回归LMs都将不可避免地崩溃。我们还发现了一个新的发现：当LMs在递归生成的文本上进行训练时，其性能会逐渐下降，直到它们的表现不如随机初始化的LM。训练后的LMs会产生大量重复的文本，并在广泛的任务中表现出色。上述证明和新发现加深了我们对LM崩溃的理解，并提供了有价值的见解，可能激发新的训练技术来减轻这一威胁。 

---
# Graph-Convolutional Networks: Named Entity Recognition and Large Language Model Embedding in Document Clustering 

**Title (ZH)**: 图卷积网络：文档聚类中的命名实体识别与大型语言模型嵌入 

**Authors**: Imed Keraghel, Mohamed Nadif  

**Link**: [PDF](https://arxiv.org/pdf/2412.14867)  

**Abstract**: Recent advances in machine learning, particularly Large Language Models (LLMs) such as BERT and GPT, provide rich contextual embeddings that improve text representation. However, current document clustering approaches often ignore the deeper relationships between named entities (NEs) and the potential of LLM embeddings. This paper proposes a novel approach that integrates Named Entity Recognition (NER) and LLM embeddings within a graph-based framework for document clustering. The method builds a graph with nodes representing documents and edges weighted by named entity similarity, optimized using a graph-convolutional network (GCN). This ensures a more effective grouping of semantically related documents. Experimental results indicate that our approach outperforms conventional co-occurrence-based methods in clustering, notably for documents rich in named entities. 

**Abstract (ZH)**: 近年来，特别是大型语言模型（LLMs）如BERT和GPT的发展，为文本表示提供了丰富的上下文嵌入。然而，当前的文档聚类方法往往忽视了命名实体（NEs）之间的深层关系以及LLM嵌入的潜力。本文提出了一种新颖的方法，将命名实体识别（NER）与LLM嵌入结合到基于图的框架中，以用于文档聚类。该方法构建了一个图，其中节点代表文档，边的权重由命名实体相似性决定，并通过图卷积网络（GCN）进行优化。这确保了更有效的语义相关文档的分组。实验结果表明，我们的方法在聚类方面优于基于共现的 conventional 方法，特别是在含有大量命名实体的文档聚类方面表现尤为突出。 

---
# Think&Cite: Improving Attributed Text Generation with Self-Guided Tree Search and Progress Reward Modeling 

**Title (ZH)**: 思与引：通过自引导树搜索和进步奖励建模改进标注文本生成 

**Authors**: Junyi Li, Hwee Tou Ng  

**Link**: [PDF](https://arxiv.org/pdf/2412.14860)  

**Abstract**: Despite their outstanding capabilities, large language models (LLMs) are prone to hallucination and producing factually incorrect information. This challenge has spurred efforts in attributed text generation, which prompts LLMs to generate content with supporting evidence. In this paper, we propose a novel framework, called Think&Cite, and formulate attributed text generation as a multi-step reasoning problem integrated with search. Specifically, we propose Self-Guided Monte Carlo Tree Search (SG-MCTS), which capitalizes on the self-reflection capability of LLMs to reflect on the intermediate states of MCTS for guiding the tree expansion process. To provide reliable and comprehensive feedback, we introduce Progress Reward Models to measure the progress of tree search from the root to the current state from two aspects, i.e., generation and attribution progress. We conduct extensive experiments on three datasets and the results show that our approach significantly outperforms baseline approaches. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）具有出色的能力，但它们容易产生幻觉并生成事实错误的信息。这一挑战激发了属性化文本生成的研究，促使LLMs生成带有支持证据的内容。在本文中，我们提出了一种新型框架，称为Think&Cite，并将属性化文本生成问题表述为一个包含搜索过程的多步骤推理问题。具体而言，我们提出了自引导蒙特卡洛树搜索（Self-Guided Monte Carlo Tree Search，SG-MCTS），利用LLMs的自我反思能力来反思MCTS的中间状态，以指导树的扩展过程。为提供可靠和全面的反馈，我们引入了进步奖励模型（Progress Reward Models），从生成和归因两个方面测量从根节点到当前状态的树搜索进度。我们在三个数据集上进行了广泛的实验，结果表明，我们的方法显著优于基线方法。 

---
# DS$^2$-ABSA: Dual-Stream Data Synthesis with Label Refinement for Few-Shot Aspect-Based Sentiment Analysis 

**Title (ZH)**: DS$^2$-ABSA: 基于双流数据合成与标签 refinement 的少样本方面基于情感分析 

**Authors**: Hongling Xu, Yice Zhang, Qianlong Wang, Ruifeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2412.14849)  

**Abstract**: Recently developed large language models (LLMs) have presented promising new avenues to address data scarcity in low-resource scenarios. In few-shot aspect-based sentiment analysis (ABSA), previous efforts have explored data augmentation techniques, which prompt LLMs to generate new samples by modifying existing ones. However, these methods fail to produce adequately diverse data, impairing their effectiveness. Besides, some studies apply in-context learning for ABSA by using specific instructions and a few selected examples as prompts. Though promising, LLMs often yield labels that deviate from task requirements. To overcome these limitations, we propose DS$^2$-ABSA, a dual-stream data synthesis framework targeted for few-shot ABSA. It leverages LLMs to synthesize data from two complementary perspectives: \textit{key-point-driven} and \textit{instance-driven}, which effectively generate diverse and high-quality ABSA samples in low-resource settings. Furthermore, a \textit{label refinement} module is integrated to improve the synthetic labels. Extensive experiments demonstrate that DS$^2$-ABSA significantly outperforms previous few-shot ABSA solutions and other LLM-oriented data generation methods. 

**Abstract (ZH)**: 最近开发的大规模语言模型（LLMs）为低资源场景下的数据稀缺问题提供了具有前景的新途径。在少量示例方面的基于方面的情感分析（ABSA）中，先前的努力探索了数据增强技术，通过修改现有数据生成新的样本。然而，这些方法未能产生充分多样化的数据，从而影响了它们的效果。此外，一些研究通过使用特定指令和少量选定的示例进行上下文学习来为ABSA服务，尽管前景 promising，但LLMs往往会产生与任务要求相偏离的标签。为克服这些局限性，我们提出了DS$^2$-ABSA，这是一种针对少量示例ABSA的双流数据合成框架。该框架利用LLMs从两个互补的角度进行数据合成：关键点驱动和实例驱动，从而在低资源场景中生成多样化且高质量的ABSA样本。此外，还集成了一个标签精炼模块，以改进合成标签。大量实验表明，DS$^2$-ABSA显著优于先前的少量示例ABSA解决方案以及其他以LLMs为中心的数据生成方法。 

---
# A Survey of RWKV 

**Title (ZH)**: 《RWKV综述》 

**Authors**: Zhiyuan Li, Tingyu Xia, Yi Chang, Yuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2412.14847)  

**Abstract**: The Receptance Weighted Key Value (RWKV) model offers a novel alternative to the Transformer architecture, merging the benefits of recurrent and attention-based systems. Unlike conventional Transformers, which depend heavily on self-attention, RWKV adeptly captures long-range dependencies with minimal computational demands. By utilizing a recurrent framework, RWKV addresses some computational inefficiencies found in Transformers, particularly in tasks with long sequences. RWKV has recently drawn considerable attention for its robust performance across multiple domains. Despite its growing popularity, no systematic review of the RWKV model exists. This paper seeks to fill this gap as the first comprehensive review of the RWKV architecture, its core principles, and its varied applications, such as natural language generation, natural language understanding, and computer vision. We assess how RWKV compares to traditional Transformer models, highlighting its capability to manage long sequences efficiently and lower computational costs. Furthermore, we explore the challenges RWKV encounters and propose potential directions for future research and advancement. We consistently maintain the related open-source materials at: this https URL. 

**Abstract (ZH)**: receptance 加权键值 (RWKV) 模型提供了与Transformer架构的一种新颖替代方案，结合了递归和基于注意力系统的优点。与高度依赖自注意力的传统Transformer不同，RWKV 能够以最小的计算需求高效捕捉长距离依赖关系。通过利用递归框架，RWKV 解决了传统Transformer在处理长序列任务时的一些计算效率问题。RWKV 最近因其在多个领域中的稳健性能而受到广泛关注。尽管它的受欢迎程度日益增长，但尚未有对该模型进行全面系统的回顾。本文旨在填补这一空白，作为首次全面回顾RWKV架构、核心原理及其多领域应用（如自然语言生成、自然语言理解和计算机视觉）的研究。我们评估了RWKV与传统Transformer模型的异同，突显了其高效处理长序列和降低计算成本的能力。此外，我们探讨了RWKV 面临的挑战，并提出了未来研究和发展的潜在方向。我们持续维护相关的开源材料如下：[该网址]。 

---
# Mapping and Influencing the Political Ideology of Large Language Models using Synthetic Personas 

**Title (ZH)**: 使用合成人物映射和影响大型语言模型的政治意识形态 

**Authors**: Pietro Bernardelle, Leon Fröhling, Stefano Civelli, Riccardo Lunardi, Kevin Roiter, Gianluca Demartini  

**Link**: [PDF](https://arxiv.org/pdf/2412.14843)  

**Abstract**: The analysis of political biases in large language models (LLMs) has primarily examined these systems as single entities with fixed viewpoints. While various methods exist for measuring such biases, the impact of persona-based prompting on LLMs' political orientation remains unexplored. In this work we leverage PersonaHub, a collection of synthetic persona descriptions, to map the political distribution of persona-based prompted LLMs using the Political Compass Test (PCT). We then examine whether these initial compass distributions can be manipulated through explicit ideological prompting towards diametrically opposed political orientations: right-authoritarian and left-libertarian. Our experiments reveal that synthetic personas predominantly cluster in the left-libertarian quadrant, with models demonstrating varying degrees of responsiveness when prompted with explicit ideological descriptors. While all models demonstrate significant shifts towards right-authoritarian positions, they exhibit more limited shifts towards left-libertarian positions, suggesting an asymmetric response to ideological manipulation that may reflect inherent biases in model training. 

**Abstract (ZH)**: 大型语言模型（LLMs）中的政治偏见分析主要将这些系统视为具有固定观点的单一实体。尽管存在多种方法来衡量这些偏见，但基于人物角色的提示对LLMs政治倾向的影响尚未被探索。在本研究中，我们利用PersonaHub，一个由合成人物描述组成的集合，通过政治极点测试（PCT）来绘制基于人物角色的提示LLMs的政治分布。然后，我们研究这些初始极点分布是否可以通过明确的意识形态提示，朝完全对立的政治方向（右翼专制和左翼自由主义）进行调整。实验结果显示，合成人物角色主要集中在左翼自由主义象限，且在使用明确的意识形态描述进行提示时，模型表现出不同程度的响应性。所有模型均显示出显著向右翼专制方向的转变，但向左翼自由主义方向的转变却相对有限，这表明对意识形态的调整响应存在不对称性，可能反映了模型训练中固有的偏差。 

---
# DynamicKV: Task-Aware Adaptive KV Cache Compression for Long Context LLMs 

**Title (ZH)**: DynamicKV：面向任务的自适应键值缓存压缩方法用于长上下文语言模型 

**Authors**: Xiabin Zhou, Wenbin Wang, Minyan Zeng, Jiaxian Guo, Xuebo Liu, Li Shen, Min Zhang, Liang Ding  

**Link**: [PDF](https://arxiv.org/pdf/2412.14838)  

**Abstract**: Efficient KV cache management in LLMs is crucial for long-context tasks like RAG and summarization. Existing KV cache compression methods enforce a fixed pattern, neglecting task-specific characteristics and reducing the retention of essential information. However, we observe distinct activation patterns across layers in various tasks, highlighting the need for adaptive strategies tailored to each task's unique demands. Based on this insight, we propose DynamicKV, a method that dynamically optimizes token retention by adjusting the number of tokens retained at each layer to adapt to the specific task. DynamicKV establishes global and per-layer maximum KV cache budgets, temporarily retaining the maximum budget for the current layer, and periodically updating the KV cache sizes of all preceding layers during inference. Our method retains only 1.7% of the KV cache size while achieving ~85% of the Full KV cache performance on LongBench. Notably, even under extreme compression (0.9%), DynamicKV surpasses state-of-the-art (SOTA) methods by 11% in the Needle-in-a-Haystack test using Mistral-7B-Instruct-v0.2. The code will be released. 

**Abstract (ZH)**: 在长上下文任务如检索增强生成（RAG）和总结中，高效的键值对（KV）缓存管理对于LLM（Large Language Model）至关重要。现有的KV缓存压缩方法会强制采用固定的模式，忽视了任务特定的特性，从而减少了重要信息的保留。然而，我们观察到，在各种任务的不同层中具有不同的激活模式，这突显了需要针对每个任务的特定需求进行适应性策略的需求。基于这一洞察，我们提出了一种名为DynamicKV的方法，该方法动态优化了标记保留，通过根据特定任务调整每一层保留的标记数量来适应任务需求。DynamicKV设定了全局和每层的最大KV缓存预算，在推理过程中临时保留当前层的最大预算，并定期更新所有先前层的KV缓存大小。我们的方法在LongBench上的KV缓存性能仅保留1.7%的大小，却能达到全KV缓存性能的约85%。值得注意的是，在极端压缩（90%）的情况下，DynamicKV在使用Mistral-7B-Instruct-v0.2进行Needle-in-a-Haystack测试时，超越了最先进的（SOTA）方法11%。代码将被公开发布。 

---
# Progressive Multimodal Reasoning via Active Retrieval 

**Title (ZH)**: 基于主动检索的渐进多模态推理 

**Authors**: Guanting Dong, Chenghao Zhang, Mengjie Deng, Yutao Zhu, Zhicheng Dou, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2412.14835)  

**Abstract**: Multi-step multimodal reasoning tasks pose significant challenges for multimodal large language models (MLLMs), and finding effective ways to enhance their performance in such scenarios remains an unresolved issue. In this paper, we propose AR-MCTS, a universal framework designed to progressively improve the reasoning capabilities of MLLMs through Active Retrieval (AR) and Monte Carlo Tree Search (MCTS). Our approach begins with the development of a unified retrieval module that retrieves key supporting insights for solving complex reasoning problems from a hybrid-modal retrieval corpus. To bridge the gap in automated multimodal reasoning verification, we employ the MCTS algorithm combined with an active retrieval mechanism, which enables the automatic generation of step-wise annotations. This strategy dynamically retrieves key insights for each reasoning step, moving beyond traditional beam search sampling to improve the diversity and reliability of the reasoning space. Additionally, we introduce a process reward model that aligns progressively to support the automatic verification of multimodal reasoning tasks. Experimental results across three complex multimodal reasoning benchmarks confirm the effectiveness of the AR-MCTS framework in enhancing the performance of various multimodal models. Further analysis demonstrates that AR-MCTS can optimize sampling diversity and accuracy, yielding reliable multimodal reasoning. 

**Abstract (ZH)**: 多步骤多模态推理任务为多模态大型语言模型（MLLMs）带来了重大挑战，如何在这种场景中有效地提升其性能仍是未解问题。本文提出了一种名为AR-MCTS的通用框架，旨在通过主动检索（AR）和蒙特卡洛树搜索（MCTS）逐步提高MLLMs的推理能力。该方法首先开发了一个统一的检索模块，从混合模态检索库中检索解决复杂推理问题的关键支持见解。为了弥合自动多模态推理验证的差距，我们采用了结合主动检索机制的Monte Carlo树搜索算法，这使得能够自动生成逐步标注。这一策略动态检索每个推理步骤的关键见解，超越了传统的束搜索采样，以改善推理空间的多样性和可靠性。此外，我们引入了一种过程奖励模型，以逐步支持多模态推理任务的自动验证。在三个复杂的多模态推理基准测试中，实验结果证实了AR-MCTS框架增强各种多模态模型性能的有效性。进一步分析表明，AR-MCTS可以优化采样多样性和准确性，从而实现可靠的多模态推理。 

---
# Mention Attention for Pronoun Translation 

**Title (ZH)**: 提及注意机制在代词翻译中的应用 

**Authors**: Gongbo Tang, Christian Hardmeier  

**Link**: [PDF](https://arxiv.org/pdf/2412.14829)  

**Abstract**: Most pronouns are referring expressions, computers need to resolve what do the pronouns refer to, and there are divergences on pronoun usage across languages. Thus, dealing with these divergences and translating pronouns is a challenge in machine translation. Mentions are referring candidates of pronouns and have closer relations with pronouns compared to general tokens. We assume that extracting additional mention features can help pronoun translation. Therefore, we introduce an additional mention attention module in the decoder to pay extra attention to source mentions but not non-mention tokens. Our mention attention module not only extracts features from source mentions, but also considers target-side context which benefits pronoun translation. In addition, we also introduce two mention classifiers to train models to recognize mentions, whose outputs guide the mention attention. We conduct experiments on the WMT17 English-German translation task, and evaluate our models on general translation and pronoun translation, using BLEU, APT, and contrastive evaluation metrics. Our proposed model outperforms the baseline Transformer model in terms of APT and BLEU scores, this confirms our hypothesis that we can improve pronoun translation by paying additional attention to source mentions, and shows that our introduced additional modules do not have negative effect on the general translation quality. 

**Abstract (ZH)**: 大多数代词都是指代表达式，计算机需要解决这些代词指的是什么，不同语言中代词的用法存在差异。因此，在机器翻译中处理这些差异并翻译代词是一个挑战。提到的词是代词的候选指代项，与一般词汇相比，提到的词与代词的关系更密切。我们假设提取额外的提到特征有助于代词翻译。因此，我们在解码器中引入了一个补充提到注意力模块，特别关注源提到的词而非非提到词。我们的提到注意力模块不仅从源提到的词中提取特征，还考虑了目标侧的上下文，这对代词翻译有益。此外，我们还引入了两个提到分类器来训练模型识别提到的词，其输出引导提到注意力。我们在WMT17英语-德语翻译任务上进行了实验，并使用BLEU、APT和对比评估指标来评估我们的模型在一般翻译和代词翻译方面的性能。我们所提出的模型在APT和BLEU分数上优于基线变压器模型，这证实了我们可以通过特别关注源提到的词来提高代词翻译的假设，并表明我们引入的额外模块没有对一般翻译质量产生负面影响。 

---
# ResoFilter: Rine-grained Synthetic Data Filtering for Large Language Models through Data-Parameter Resonance Analysis 

**Title (ZH)**: ResoFilter：通过数据-参数共振分析的大规模语言模型细粒度合成数据过滤方法 

**Authors**: Zeao Tu, Xiangdi Meng, Yu He, Zihan Yao, Tianyu Qi, Jun Liu, Ming Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.14809)  

**Abstract**: Large language models (LLMs) have shown remarkable effectiveness across various domains, with data augmentation methods utilizing GPT for synthetic data generation becoming prevalent. However, the quality and utility of augmented data remain questionable, and current methods lack clear metrics for evaluating data characteristics. To address these challenges, we propose ResoFilter, a novel method that integrates models, data, and tasks to refine datasets. ResoFilter leverages the fine-tuning process to obtain Data-Parameter features for data selection, offering improved interpretability by representing data characteristics through model weights. Our experiments demonstrate that ResoFilter achieves comparable results to full-scale fine-tuning using only half the data in mathematical tasks and exhibits strong generalization across different models and domains. This method provides valuable insights for constructing synthetic datasets and evaluating high-quality data, offering a promising solution for enhancing data augmentation techniques and improving training dataset quality for LLMs. For reproducibility, we will release our code and data upon acceptance. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各个领域中展现了显著的有效性，利用GPT进行合成数据生成的数据增强方法变得越来越常见。然而，增强数据的质量和实用性仍然存疑，当前的方法缺乏明确的评价数据特征的指标。为解决这些问题，我们提出了一种名为ResoFilter的新颖方法，该方法将模型、数据和任务相结合来精炼数据集。ResoFilter通过微调过程获取数据-参数特征，通过对模型权重的表示来改进数据特征的解释性。我们的实验表明，ResoFilter在数学任务中仅使用一半的数据就能达到与全量微调相当的结果，并且在不同模型和领域中表现出强大的泛化能力。这种方法为构建合成数据集和评估高质量数据提供了有价值的见解，并为增强数据增强技术以及提高LLM训练数据集质量提供了前景广阔的方法。为了可再现性，我们将在论文被接受后公开我们的代码和数据。 

---
# Disentangling Reasoning Tokens and Boilerplate Tokens For Language Model Fine-tuning 

**Title (ZH)**: 拆分推理令牌和模板令牌以进行语言模型微调 

**Authors**: Ziang Ye, Zhenru Zhang, Yang Zhang, Jianxin Ma, Junyang Lin, Fuli Feng  

**Link**: [PDF](https://arxiv.org/pdf/2412.14780)  

**Abstract**: When using agent-task datasets to enhance agent capabilities for Large Language Models (LLMs), current methodologies often treat all tokens within a sample equally. However, we argue that tokens serving different roles - specifically, reasoning tokens versus boilerplate tokens (e.g., those governing output format) - differ significantly in importance and learning complexity, necessitating their disentanglement and distinct treatment. To address this, we propose a novel Shuffle-Aware Discriminator (SHAD) for adaptive token discrimination. SHAD classifies tokens by exploiting predictability differences observed after shuffling input-output combinations across samples: boilerplate tokens, due to their repetitive nature among samples, maintain predictability, whereas reasoning tokens do not. Using SHAD, we propose the Reasoning-highlighted Fine-Tuning (RFT) method, which adaptively emphasizes reasoning tokens during fine-tuning, yielding notable performance gains over common Supervised Fine-Tuning (SFT). 

**Abstract (ZH)**: 当使用代理任务数据集来增强大型语言模型（LLMs）的代理能力时，当前的方法往往将每个样本内的所有标记视为同等重要。然而，我们认为在不同角色中承担不同功能的标记——尤其是推理标记与模板标记（如控制输出格式的标记）——在重要性和学习复杂性方面存在显著差异，需要进行分离并分别处理。为解决这一问题，我们提出了一种新的Shuffle-Aware Discriminator（SHAD）用于自适应标记区分。SHAD通过在样本间打乱输入和输出组合后观察预测性差异来进行标记分类：模板标记由于在样本间具有重复性，保持预测性，而推理标记则不具有这种特性。利用SHAD，我们提出了推理突出的微调方法（RFT），该方法在微调过程中自适应地强调推理标记，与常见监督微调（SFT）相比，显示出显著的性能提升。 

---
# ALKAFI-LLAMA3: Fine-Tuning LLMs for Precise Legal Understanding in Palestine 

**Title (ZH)**: ALKAFI-LLAMA3：为巴勒斯坦精准法律理解Fine-tuning大语言模型 

**Authors**: Rabee Qasem, Mohannad Hendi, Banan Tantour  

**Link**: [PDF](https://arxiv.org/pdf/2412.14771)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable potential in diverse domains, yet their application in the legal sector, particularly in low-resource contexts, remains limited. This study addresses the challenges of adapting LLMs to the Palestinian legal domain, where political instability, fragmented legal frameworks, and limited AI resources hinder effective machine-learning applications. We present a fine-tuned model based on a quantized version of Llama-3.2-1B-Instruct, trained on a synthetic data set derived from Palestinian legal texts. Using smaller-scale models and strategically generated question-answer pairs, we achieve a cost-effective, locally sustainable solution that provides accurate and contextually relevant legal guidance. Our experiments demonstrate promising performance on various query types, ranging from yes/no questions and narrative explanations to complex legal differentiations, while highlighting areas for improvement, such as handling calculation-based inquiries and structured list formatting. This work provides a pathway for the deployment of AI-driven legal assistance tools tailored to the needs of resource-constrained environments. 

**Abstract (ZH)**: 大型语言模型（Large Language Models, LLMs）在多个领域展现了巨大的潜力，但在法律领域的应用，尤其是在资源匮乏的背景下，仍受到限制。本研究旨在解决将LLMs适应巴勒斯坦法律领域所面临的挑战，其中包括政治不稳定、分散的法律框架以及有限的AI资源，这些因素阻碍了有效的机器学习应用的发展。我们提出了一种基于量化版本Llama-3.2-1B-Instruct模型的微调模型，该模型是通过阿拉伯语建模的巴勒斯坦法律文本合成数据集训练而成。利用规模较小的模型和战略性生成的问题-答案对，我们实现了成本效益高、本地可持续的解决方案，提供了精确且相关性强的法律指导。实验结果显示，在不同类型的查询上，该模型均表现出良好的性能，涵盖了从是/否问题和叙述性解释到复杂法律差异等方面，同时也指出了需要改进的领域，例如处理基于计算的问题和结构化列表格式。本研究为资源受限环境中提供定制化AI驱动的法律辅助工具的部署提供了路径。 

---
# PsyDraw: A Multi-Agent Multimodal System for Mental Health Screening in Left-Behind Children 

**Title (ZH)**: PsyDraw：一种针对留守儿童心理健康筛查的多智能体多模态系统 

**Authors**: Yiqun Zhang, Xiaocui Yang, Xiaobai Li, Siyuan Yu, Yi Luan, Shi Feng, Daling Wang, Yifei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.14769)  

**Abstract**: Left-behind children (LBCs), numbering over 66 million in China, face severe mental health challenges due to parental migration for work. Early screening and identification of at-risk LBCs is crucial, yet challenging due to the severe shortage of mental health professionals, especially in rural areas. While the House-Tree-Person (HTP) test shows higher child participation rates, its requirement for expert interpretation limits its application in resource-scarce regions. To address this challenge, we propose PsyDraw, a multi-agent system based on Multimodal Large Language Models that assists mental health professionals in analyzing HTP drawings. The system employs specialized agents for feature extraction and psychological interpretation, operating in two stages: comprehensive feature analysis and professional report generation. Evaluation of HTP drawings from 290 primary school students reveals that 71.03% of the analyzes achieved High Consistency with professional evaluations, 26.21% Moderate Consistency and only 2.41% Low Consistency. The system identified 31.03% of cases requiring professional attention, demonstrating its effectiveness as a preliminary screening tool. Currently deployed in pilot schools, \method shows promise in supporting mental health professionals, particularly in resource-limited areas, while maintaining high professional standards in psychological assessment. 

**Abstract (ZH)**: 留守儿童（LBCs）在中国的人数超过6600万，由于父母因工作迁出，他们面临着严重的心理健康挑战。早期对风险留守儿童进行筛查和识别至关重要，但由于心理健康专业人士严重短缺，尤其是农村地区，这一任务变得极具挑战性。尽管房屋-树-人（HTP）测试显示出更高的儿童参与率，但它需要专家解读的特点限制了其在资源稀缺地区的应用。为了应对这一挑战，我们提出了一种基于多模态大型语言模型的多代理系统——PsyDraw，该系统可协助心理健康专业人士分析HTP绘画作品。该系统采用专门的代理进行特征提取和心理解读，分为两个阶段：全面特征分析和专业报告生成。对来自290名小学生HTP绘画作品的评估结果显示，71.03%的分析与专业评估达到了高一致性，26.21%达到了中等一致性，只有2.41%达到了低一致性。系统识别出了31.03%需要专业关注的案例，表明其作为初步筛查工具的有效性。目前该系统已在试点学校部署，其方法在支持心理健康专业人士，尤其是在资源有限的地区，同时保持高度的专业标准方面显示出巨大潜力。 

---
# Query pipeline optimization for cancer patient question answering systems 

**Title (ZH)**: 癌症患者问答系统中查询管道的优化方法 

**Authors**: Maolin He, Rena Gao, Mike Conway, Brian E. Chapman  

**Link**: [PDF](https://arxiv.org/pdf/2412.14751)  

**Abstract**: Retrieval-augmented generation (RAG) mitigates hallucination in Large Language Models (LLMs) by using query pipelines to retrieve relevant external information and grounding responses in retrieved knowledge. However, query pipeline optimization for cancer patient question-answering (CPQA) systems requires separately optimizing multiple components with domain-specific considerations. We propose a novel three-aspect optimization approach for the RAG query pipeline in CPQA systems, utilizing public biomedical databases like PubMed and PubMed Central. Our optimization includes: (1) document retrieval, utilizing a comparative analysis of NCBI resources and introducing Hybrid Semantic Real-time Document Retrieval (HSRDR); (2) passage retrieval, identifying optimal pairings of dense retrievers and rerankers; and (3) semantic representation, introducing Semantic Enhanced Overlap Segmentation (SEOS) for improved contextual understanding. On a custom-developed dataset tailored for cancer-related inquiries, our optimized RAG approach improved the answer accuracy of Claude-3-haiku by 5.24% over chain-of-thought prompting and about 3% over a naive RAG setup. This study highlights the importance of domain-specific query optimization in realizing the full potential of RAG and provides a robust framework for building more accurate and reliable CPQA systems, advancing the development of RAG-based biomedical systems. 

**Abstract (ZH)**: 检索增强生成（RAG）通过使用查询管道检索相关外部信息并在检索到的知识基础上确立响应，从而缓解了大型语言模型（LLMs）中的幻觉现象。然而，针对癌症患者问答（CPQA）系统的查询管道优化需要分别优化多个具有领域特定考虑的组件。我们提出了一种针对CPQA系统中RAG查询管道的新型三方面优化方法，利用PubMed和PubMed Central等公共生物医学数据库。我们的优化包括：（1）文档检索，采用NCBI资源的比较分析，并引入了混合语义实时文档检索（HSRDR）；（2）段落检索，确定密集检索器和排序器的最佳配对；（3）语义表示，引入语义增强重叠分割（SEOS），以提高上下文理解能力。在针对癌症相关查询定制开发的数据集上，我们优化的RAG方法在回答精度方面提高了Claude-3-haiku约5.24%，并且相对于链式思考提示（chain-of-thought prompting）高约3%，相对于原始的RAG设置高约3%。本研究强调了在实现RAG潜力方面领域特定查询优化的重要性，并提供了一个强大的框架，用于构建更准确可靠的CPQA系统，推动基于RAG的生物医学系统的发展。 

---
# On Verbalized Confidence Scores for LLMs 

**Title (ZH)**: 关于语音表达的置信分数对于大语言模型的研究 

**Authors**: Daniel Yang, Yao-Hung Hubert Tsai, Makoto Yamada  

**Link**: [PDF](https://arxiv.org/pdf/2412.14737)  

**Abstract**: The rise of large language models (LLMs) and their tight integration into our daily life make it essential to dedicate efforts towards their trustworthiness. Uncertainty quantification for LLMs can establish more human trust into their responses, but also allows LLM agents to make more informed decisions based on each other's uncertainty. To estimate the uncertainty in a response, internal token logits, task-specific proxy models, or sampling of multiple responses are commonly used. This work focuses on asking the LLM itself to verbalize its uncertainty with a confidence score as part of its output tokens, which is a promising way for prompt- and model-agnostic uncertainty quantification with low overhead. Using an extensive benchmark, we assess the reliability of verbalized confidence scores with respect to different datasets, models, and prompt methods. Our results reveal that the reliability of these scores strongly depends on how the model is asked, but also that it is possible to extract well-calibrated confidence scores with certain prompt methods. We argue that verbalized confidence scores can become a simple but effective and versatile uncertainty quantification method in the future. Our code is available at this https URL . 

**Abstract (ZH)**: 大型语言模型（LLMs）的兴起及其与我们日常生活紧密结合，使得致力于提升其可信度变得至关重要。对LLMs进行不确定性量化可以增强人们对它们响应的信任度，同时也使LLM代理能够基于彼此的不确定性做出更加明智的决策。为了估计响应的不确定性，通常使用内部标记概率、任务特定的代理模型或多次响应采样。本研究关注让LLM自己以置信度分数的形式在其输出标记中口头表达其不确定性，这是一种具有低开销的提示和模型无关的不确定性量化方法的有前景方式。通过广泛的标准测试基准，我们评估了口头表达的置信度分数在不同数据集、模型和提示方法方面的可靠性。我们的研究结果表明，这些分数的可靠性高度依赖于模型的提问方式，但也表明通过特定的提示方法可以提取出校准良好的置信度分数。我们认为，口头表达的置信度分数在未来可以成为一种简单但有效且多功能的不确定性量化方法。我们的代码可在以下链接获取：[此处提供链接]。 

---
# How to Synthesize Text Data without Model Collapse? 

**Title (ZH)**: 如何合成文本数据而不发生模型崩塌？ 

**Authors**: Xuekai Zhu, Daixuan Cheng, Hengli Li, Kaiyan Zhang, Ermo Hua, Xingtai Lv, Ning Ding, Zhouhan Lin, Zilong Zheng, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2412.14689)  

**Abstract**: Model collapse in synthetic data indicates that iterative training on self-generated data leads to a gradual decline in performance. With the proliferation of AI models, synthetic data will fundamentally reshape the web data ecosystem. Future GPT-$\{n\}$ models will inevitably be trained on a blend of synthetic and human-produced data. In this paper, we focus on two questions: what is the impact of synthetic data on language model training, and how to synthesize data without model collapse? We first pre-train language models across different proportions of synthetic data, revealing a negative correlation between the proportion of synthetic data and model performance. We further conduct statistical analysis on synthetic data to uncover distributional shift phenomenon and over-concentration of n-gram features. Inspired by the above findings, we propose token editing on human-produced data to obtain semi-synthetic data. As a proof of concept, we theoretically demonstrate that token-level editing can prevent model collapse, as the test error is constrained by a finite upper bound. We conduct extensive experiments on pre-training from scratch, continual pre-training, and supervised fine-tuning. The results validate our theoretical proof that token-level editing improves data quality and enhances model performance. 

**Abstract (ZH)**: 模型在合成数据上的崩溃表明，迭代训练于自我生成的数据会导致性能逐渐下降。随着人工智能模型的普及，合成数据将从根本上重塑网络数据生态系统。未来GPT-$\{n\}$模型不可避免地会采用合成数据和人类生成数据的混合方式进行训练。在本文中，我们重点关注两个问题：合成数据对语言模型训练的影响是什么，以及如何合成数据而不导致模型崩溃？我们首先在不同比例的合成数据下预训练语言模型，发现合成数据的比例与模型性能之间存在负相关关系。进一步地，我们对合成数据进行统计分析，发现了分布偏移现象和n-克元件特征的过度集中。受到上述发现的启发，我们提出对人类生成的数据进行令牌编辑以获取半合成数据。作为一个概念验证，我们理论上证明了令牌级别的编辑可以防止模型崩溃，因为测试误差受到有限上限的约束。我们在从头预训练、持续预训练和监督微调方面进行了广泛的实验。结果验证了我们的理论证明：令牌级别的编辑可以提高数据质量并增强模型性能。 

---
# Each Fake News is Fake in its Own Way: An Attribution Multi-Granularity Benchmark for Multimodal Fake News Detection 

**Title (ZH)**: 每条假新闻都有其独特之处：一个跨模态假新闻检测的属性多粒度基准 

**Authors**: Hao Guo, Zihan Ma, Zhi Zeng, Minnan Luo, Weixin Zeng, Jiuyang Tang, Xiang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2412.14686)  

**Abstract**: Social platforms, while facilitating access to information, have also become saturated with a plethora of fake news, resulting in negative consequences. Automatic multimodal fake news detection is a worthwhile pursuit. Existing multimodal fake news datasets only provide binary labels of real or fake. However, real news is alike, while each fake news is fake in its own way. These datasets fail to reflect the mixed nature of various types of multimodal fake news. To bridge the gap, we construct an attributing multi-granularity multimodal fake news detection dataset \amg, revealing the inherent fake pattern. Furthermore, we propose a multi-granularity clue alignment model \our to achieve multimodal fake news detection and attribution. Experimental results demonstrate that \amg is a challenging dataset, and its attribution setting opens up new avenues for future research. 

**Abstract (ZH)**: 社交媒体平台在促进信息获取的同时，也充斥着大量的虚假新闻，导致了一系列负面影响。自动多模态虚假新闻检测是一个值得追求的目标。现有的多模态虚假新闻数据集仅提供真实或虚假的二元标签。然而，真实新闻大体相似，而每条虚假新闻都有其独特性。这些数据集未能反映各种类型多模态虚假新闻的混合本质。为弥补这一不足，我们构建了一个具有归因多粒度多模态虚假新闻检测数据集 \AMG，揭示了内在的虚假模式。此外，我们提出了一种多粒度线索对齐模型 \Our，以实现多模态虚假新闻的检测和归因。实验结果表明，\AMG 是一个具有挑战性的数据集，并且其归因设置为未来的研究开启了新的途径。 

---
# LLMs as mediators: Can they diagnose conflicts accurately? 

**Title (ZH)**: LLMs作为调解者：它们能否准确诊断冲突？ 

**Authors**: Özgecan Koçak, Phanish Puranam, Afşar Yegin  

**Link**: [PDF](https://arxiv.org/pdf/2412.14675)  

**Abstract**: Prior research indicates that to be able to mediate conflict, observers of disagreements between parties must be able to reliably distinguish the sources of their disagreement as stemming from differences in beliefs about what is true (causality) vs. differences in what they value (morality). In this paper, we test if OpenAI's Large Language Models GPT 3.5 and GPT 4 can perform this task and whether one or other type of disagreement proves particularly challenging for LLM's to diagnose. We replicate study 1 in Koçak et al. (2003), which employes a vignette design, with OpenAI's GPT 3.5 and GPT 4. We find that both LLMs have similar semantic understanding of the distinction between causal and moral codes as humans and can reliably distinguish between them. When asked to diagnose the source of disagreement in a conversation, both LLMs, compared to humans, exhibit a tendency to overestimate the extent of causal disagreement and underestimate the extent of moral disagreement in the moral misalignment condition. This tendency is especially pronounced for GPT 4 when using a proximate scale that relies on concrete language specific to an issue. GPT 3.5 does not perform as well as GPT4 or humans when using either the proximate or the distal scale. The study provides a first test of the potential for using LLMs to mediate conflict by diagnosing the root of disagreements in causal and evaluative codes. 

**Abstract (ZH)**: 先前的研究表明，观察者要在双方争端中充当调解人，必须能够可靠地区分他们分歧的来源是基于对事实的理解差异（因果性）还是基于价值观的差异（道德性）。本研究旨在测试OpenAI的大语言模型GPT 3.5和GPT 4是否能够完成这一任务，并且探究是哪种类型的分歧对大语言模型来说更加具有挑战性。我们复制了Koçak等人（2003）的研究，该研究采用了情景描述的设计，使用了OpenAI的GPT 3.5和GPT 4进行实验。我们发现，这两种大语言模型在区分因果性和道德性方面的语义理解与人类相似，并且能够可靠地区分这两者。

当被要求诊断对话中的分歧来源时，与人类相比，这两种大语言模型在道德不一致的情况下都有倾向于高估因果性分歧并低估道德性分歧的倾向。特别是使用与具体问题相关的具体语言构建的邻近尺度时，这种倾向对于GPT 4尤为显著。GPT 3.5在使用邻近尺度和远端尺度时的表现都不及GPT 4或人类。该研究提供了大语言模型通过诊断因果性和评价性代码的分歧根源来调解冲突的潜在能力的初步测试。 

---
# Analysis and Visualization of Linguistic Structures in Large Language Models: Neural Representations of Verb-Particle Constructions in BERT 

**Title (ZH)**: 大型语言模型中的语言结构分析与可视化：BERT中动词-介词构造的神经表示分析 

**Authors**: Hassane Kissane, Achim Schilling, Patrick Krauss  

**Link**: [PDF](https://arxiv.org/pdf/2412.14670)  

**Abstract**: This study investigates the internal representations of verb-particle combinations within transformer-based large language models (LLMs), specifically examining how these models capture lexical and syntactic nuances at different neural network layers. Employing the BERT architecture, we analyse the representational efficacy of its layers for various verb-particle constructions such as 'agree on', 'come back', and 'give up'. Our methodology includes a detailed dataset preparation from the British National Corpus, followed by extensive model training and output analysis through techniques like multi-dimensional scaling (MDS) and generalized discrimination value (GDV) calculations. Results show that BERT's middle layers most effectively capture syntactic structures, with significant variability in representational accuracy across different verb categories. These findings challenge the conventional uniformity assumed in neural network processing of linguistic elements and suggest a complex interplay between network architecture and linguistic representation. Our research contributes to a better understanding of how deep learning models comprehend and process language, offering insights into the potential and limitations of current neural approaches to linguistic analysis. This study not only advances our knowledge in computational linguistics but also prompts further research into optimizing neural architectures for enhanced linguistic precision. 

**Abstract (ZH)**: 本研究探讨了基于变换器的大型语言模型（LLM）内部对动词-副词组合（verb-particle combinations）的表征，具体研究这些模型在不同神经网络层如何捕捉词汇和句法细微差别。采用BERT架构，我们分析了其各层对诸如“agree on”、“come back”和“give up”等不同动词-副词构型的表征效果。研究方法包括从英国国家语料库（British National Corpus）准备详细的数据集，随后进行广泛的模型训练，并通过多维尺度分析（MDS）和广义鉴别值（GDV）计算等技术进行输出分析。结果显示，BERT的中间层最有效地捕捉句法结构，不同动词类别在表征准确性上存在显著差异。这些发现挑战了神经网络对语言成分处理的常规统一性假设，并暗示了网络架构和语言表征之间的复杂相互作用。我们的研究增进了对深度学习模型如何理解和处理语言的认识，提供了关于当前神经方法在语言分析中潜力和局限性的见解。本研究不仅推进了计算语言学的知识，还激发了优化神经架构以增强语言精确度的进一步研究。 

---
# Length Controlled Generation for Black-box LLMs 

**Title (ZH)**: 长度可控生成用于黑盒大语言模型 

**Authors**: Yuxuan Gu, Wenjie Wang, Xiaocheng Feng, Weihong Zhong, Kun Zhu, Lei Huang, Tat-Seng Chua, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2412.14656)  

**Abstract**: Large language models (LLMs) have demonstrated impressive instruction following capabilities, while still struggling to accurately manage the length of the generated text, which is a fundamental requirement in many real-world applications. Existing length control methods involve fine-tuning the parameters of LLMs, which is inefficient and suboptimal for practical use. In this paper, we propose a novel iterative sampling framework for text length control, integrating the Metropolis-Hastings algorithm with an importance sampling acceleration strategy. This framework efficiently and reliably regulates LLMs to generate length-constrained text without modifying the underlying parameters, thereby preserving the original capabilities of LLMs. Experimental results demonstrate that our framework achieves almost 100\% success rates of length control on Llama3.1 for tasks such as length-controlled abstractive summarization and length-constrained instruction following, with minimal additional computational overhead. This also highlights the significant potential of our method for precise length control across a broader range of applications, without compromising the versatility of LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）在指令遵循方面展现了令人印象深刻的性能，但在生成文本长度控制方面仍然存在挑战，这是许多实际应用中的一个基本要求。现有的长度控制方法通常涉及对LLMs参数的微调，这在实际使用中既无效率又不太理想。本文提出了一种新颖的迭代采样框架，结合了Metropolis-Hastings算法和重要性采样加速策略，以高效且可靠地调节LLMs生成长度受限的文本，且无需修改底层参数，从而保持了LLMs的原始能力。实验结果表明，该框架在Llama3.1上实现了几乎100%的任务长度控制成功率，适用于长度受控的摘要生成和长度受限的指令遵循等任务，且附加的计算开销 minimal。这还突显了该方法在更广泛的应用中实现精确长度控制的巨大潜力，而不牺牲LLMs的灵活性。 

---
# TOMG-Bench: Evaluating LLMs on Text-based Open Molecule Generation 

**Title (ZH)**: TOMG-Bench：基于文本的开放分子生成评测LLM模型 

**Authors**: Jiatong Li, Junxian Li, Yunqing Liu, Dongzhan Zhou, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.14642)  

**Abstract**: In this paper, we propose Text-based Open Molecule Generation Benchmark (TOMG-Bench), the first benchmark to evaluate the open-domain molecule generation capability of LLMs. TOMG-Bench encompasses a dataset of three major tasks: molecule editing (MolEdit), molecule optimization (MolOpt), and customized molecule generation (MolCustom). Each task further contains three subtasks, with each subtask comprising 5,000 test samples. Given the inherent complexity of open molecule generation, we have also developed an automated evaluation system that helps measure both the quality and the accuracy of the generated molecules. Our comprehensive benchmarking of 25 LLMs reveals the current limitations and potential areas for improvement in text-guided molecule discovery. Furthermore, with the assistance of OpenMolIns, a specialized instruction tuning dataset proposed for solving challenges raised by TOMG-Bench, Llama3.1-8B could outperform all the open-source general LLMs, even surpassing GPT-3.5-turbo by 46.5\% on TOMG-Bench. Our codes and datasets are available through this https URL. 

**Abstract (ZH)**: 在本文中，我们提出了基于文本的开放分子生成基准（TOMG-Bench），这是第一个用于评估大规模语言模型（LLMs）在开放域分子生成能力的基准。TOMG-Bench 包含三个主要任务的数据集：分子编辑（MolEdit）、分子优化（MolOpt）和定制分子生成（MolCustom）。每个任务进一步细分为三个子任务，每个子任务包含5,000个测试样本。鉴于开放分子生成的固有复杂性，我们还开发了一个自动化评估系统，用于测量生成分子的质量和准确性。我们对25个LLM的全面基准测试揭示了当前文本引导分子发现中的局限性以及改进的潜在领域。此外，在OpenMolIns（一个专门为此提出的指令调整数据集）的帮助下，Llama3.1-8B 能够超越所有开源通用LLM，甚至在TOMG-Bench 上优于GPT-3.5-turbo 46.5%。我们的代码和数据集可以通过以下链接获取：[提供链接]。 

---
# Learning to Generate Research Idea with Dynamic Control 

**Title (ZH)**: 学习使用动态控制生成研究idea 

**Authors**: Ruochen Li, Liqiang Jing, Chi Han, Jiawei Zhou, Xinya Du  

**Link**: [PDF](https://arxiv.org/pdf/2412.14626)  

**Abstract**: The rapid advancements in large language models (LLMs) have demonstrated their potential to accelerate scientific discovery, particularly in automating the process of research ideation. LLM-based systems have shown promise in generating hypotheses and research ideas. However, current approaches predominantly rely on prompting-based pre-trained models, limiting their ability to optimize generated content effectively. Moreover, they also lack the capability to deal with the complex interdependence and inherent restrictions among novelty, feasibility, and effectiveness, which remains challenging due to the inherent trade-offs among these dimensions, such as the innovation-feasibility conflict. To address these limitations, we for the first time propose fine-tuning LLMs to be better idea proposers and introduce a novel framework that employs a two-stage approach combining Supervised Fine-Tuning (SFT) and controllable Reinforcement Learning (RL). In the SFT stage, the model learns foundational patterns from pairs of research papers and follow-up ideas. In the RL stage, multi-dimensional reward modeling, guided by fine-grained feedback, evaluates and optimizes the generated ideas across key metrics. Dimensional controllers enable dynamic adjustment of generation, while a sentence-level decoder ensures context-aware emphasis during inference. Our framework provides a balanced approach to research ideation, achieving high-quality outcomes by dynamically navigating the trade-offs among novelty, feasibility, and effectiveness. 

**Abstract (ZH)**: 大型语言模型（LLMs）的快速进步已经展示了它们加速科学研究的潜力，特别是在自动化研究构思过程中。基于LLM的系统在生成假设和研究构思方面显示出一定的前景。然而，当前的方法主要依赖于基于提示的预训练模型，这限制了它们优化生成内容的能力。此外，它们在处理新颖性、可行性与有效性之间的复杂相互依赖关系和固有限制方面还缺乏能力，这些限制主要源于这些维度之间固有的权衡，例如创新与可行性之间的冲突。为了解决这些局限性，我们首次提出对LLM进行微调，使其更好地提出想法，并引入了一种新颖的框架，该框架结合了监督微调（SFT）和可控强化学习（RL）的两阶段方法。在SFT阶段，模型从研究论文及其后续想法的配对中学习基础模式。在RL阶段，基于细粒度反馈的多维度奖励建模评估和优化生成的想法，这些想法在关键指标上得到改进。维度控制器允许动态调整生成过程，而句子级解码器则确保在推理过程中对上下文有所关注。我们的框架提供了一种平衡的科研构思方法，通过动态导航新颖性、可行性与有效性之间的权衡，实现了高质量的结果。 

---
# How good is GPT at writing political speeches for the White House? 

**Title (ZH)**: 《GPT撰写白宫政治演讲的能力如何？》

这个标题翻译成中文符合学术规范，保留了原文的主要信息和疑问点。 

**Authors**: Jacques Savoy  

**Link**: [PDF](https://arxiv.org/pdf/2412.14617)  

**Abstract**: Using large language models (LLMs), computers are able to generate a written text in response to a us er request. As this pervasive technology can be applied in numerous contexts, this study analyses the written style of one LLM called GPT by comparing its generated speeches with those of the recent US presidents. To achieve this objective, the State of the Union (SOTU) addresses written by Reagan to Biden are contrasted to those produced by both GPT-3.5 and GPT-4.o versions. Compared to US presidents, GPT tends to overuse the lemma "we" and produce shorter messages with, on average, longer sentences. Moreover, GPT opts for an optimistic tone, opting more often for political (e.g., president, Congress), symbolic (e.g., freedom), and abstract terms (e.g., freedom). Even when imposing an author's style to GPT, the resulting speech remains distinct from addresses written by the target author. Finally, the two GPT versions present distinct characteristics, but both appear overall dissimilar to true presidential messages. 

**Abstract (ZH)**: 使用大规模语言模型（LLMs），计算机能够根据用户请求生成书面文本。由于这项普遍的技术可以在多种场景中应用，本研究通过将一种名为GPT的语言模型生成的演讲与其最近的美国总统的演讲进行比较，分析了GPT的书面风格。具体来说，本研究对比了里根总统至拜登总统的国情咨文（SOTU），并将这些演讲与GPT-3.5和GPT-4版本生成的演讲进行对照。与美国总统的演讲相比，GPT更倾向于过度使用“我们”这一词素，并生成较短的信息，平均每句话更长。此外，GPT倾向于采用更为乐观的语气，更频繁地使用政治性（如，总统、国会）、象征性（如，自由）和抽象性（如，自由）的术语。即使尝试为GPT设定制作者的风格，生成的演讲仍与目标作者的演讲存在明显差异。最后，两个GPT版本都展现出独特的特点，但总体上与真实的总统演讲相比显得不同。 

---
# HarmonicEval: Multi-modal, Multi-task, Multi-criteria Automatic Evaluation Using a Vision Language Model 

**Title (ZH)**: 谐波评估：基于视觉语言模型的多模态、多任务、多指标自动评估方法 

**Authors**: Masanari Ohi, Masahiro Kaneko, Naoaki Okazaki, Nakamasa Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2412.14613)  

**Abstract**: Vision-language models (VLMs) have shown impressive abilities in text and image understanding. However, existing metrics for evaluating the text generated by VLMs focus exclusively on overall quality, leading to two limitations: 1) it is challenging to identify which aspects of the text need improvement from the overall score; 2) metrics may overlook specific evaluation criteria when predicting an overall score. To address these limitations, we propose HarmonicEval, a reference-free evaluation metric that aggregates criterion-wise scores to produce the overall score in a bottom-up manner. Furthermore, we construct the Multi-task Multi-criteria Human Evaluation (MMHE) dataset, which comprises 18,000 expert human judgments across four vision-language tasks. Our experiments demonstrate that HarmonicEval achieves higher correlations with human judgments than conventional metrics while providing numerical scores for each criterion. 

**Abstract (ZH)**: 视觉语言模型（VLMs）在文本和图像理解方面展现出了令人印象深刻的性能。然而，现有的用于评估VLMs生成文本的指标仅专注于整体质量，导致了两个局限性：1) 从整体评分中难以识别哪方面文本需要改进；2) 指标在预测整体评分时可能忽略某些特定的评估标准。为解决这些局限性，我们提出了一个参考自由的评估指标——HarmonicEval，该指标通过自底向上的方式聚合各个指标的评分以产生整体评分。此外，我们构建了多任务多指标人工评估（MMHE）数据集，该数据集包含了四个视觉语言任务的18,000个专家人工判断。我们的实验结果表明，HarmonicEval与人工判断的相关性高于传统指标，并且能够为每个指标提供数值评分。 

---
# KARRIEREWEGE: A Large Scale Career Path Prediction Dataset 

**Title (ZH)**: 职业生涯路径：一个大规模的职业路径预测数据集 

**Authors**: Elena Senger, Yuri Campbell, Rob van der Goot, Barbara Plank  

**Link**: [PDF](https://arxiv.org/pdf/2412.14612)  

**Abstract**: Accurate career path prediction can support many stakeholders, like job seekers, recruiters, HR, and project managers. However, publicly available data and tools for career path prediction are scarce. In this work, we introduce KARRIEREWEGE, a comprehensive, publicly available dataset containing over 500k career paths, significantly surpassing the size of previously available datasets. We link the dataset to the ESCO taxonomy to offer a valuable resource for predicting career trajectories. To tackle the problem of free-text inputs typically found in resumes, we enhance it by synthesizing job titles and descriptions resulting in KARRIEREWEGE+. This allows for accurate predictions from unstructured data, closely aligning with real-world application challenges. We benchmark existing state-of-the-art (SOTA) models on our dataset and a prior benchmark and observe improved performance and robustness, particularly for free-text use cases, due to the synthesized data. 

**Abstract (ZH)**: 准确的职业路径预测可以为许多相关方提供支持，例如求职者、招聘人员、人力资源管理人员和项目经理。然而，可用于职业路径预测的公开数据和工具较为稀缺。在这项工作中，我们介绍了KARRIEREWEGE，这是一个全面且公开可用的数据集，包含超过50万条职业路径，远远超过了之前可用的数据集规模。我们将该数据集链接到ESCOnomics分类体系，从而提供了一个宝贵的职业轨道预测资源。为了解决简历中常见的自由文本输入问题，我们通过合成职位名称和描述，将其增强为KARRIEREWEGE+，从而能够从非结构化数据中进行准确预测，并且更贴近实际应用中的挑战。我们在KARRIEREWEGE和先前的基准数据集上对现有的最先进的（SOTA）模型进行评估，并观察到性能和稳健性提升，尤其是在自由文本应用场景中，这是由于使用了合成数据。 

---
# Beyond Guilt: Legal Judgment Prediction with Trichotomous Reasoning 

**Title (ZH)**: 超越罪责：基于三元推理的法律判决预测 

**Authors**: Kepu Zhang, Haoyue Yang, Xu Tang, Weijie Yu, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2412.14588)  

**Abstract**: In legal practice, judges apply the trichotomous dogmatics of criminal law, sequentially assessing the elements of the offense, unlawfulness, and culpability to determine whether an individual's conduct constitutes a crime. Although current legal large language models (LLMs) show promising accuracy in judgment prediction, they lack trichotomous reasoning capabilities due to the absence of an appropriate benchmark dataset, preventing them from predicting innocent outcomes. As a result, every input is automatically assigned a charge, limiting their practical utility in legal contexts. To bridge this gap, we introduce LJPIV, the first benchmark dataset for Legal Judgment Prediction with Innocent Verdicts. Adhering to the trichotomous dogmatics, we extend three widely-used legal datasets through LLM-based augmentation and manual verification. Our experiments with state-of-the-art legal LLMs and novel strategies that integrate trichotomous reasoning into zero-shot prompting and fine-tuning reveal: (1) current legal LLMs have significant room for improvement, with even the best models achieving an F1 score of less than 0.3 on LJPIV; and (2) our strategies notably enhance both in-domain and cross-domain judgment prediction accuracy, especially for cases resulting in an innocent verdict. 

**Abstract (ZH)**: 在法律实践中，法官根据刑事法律的三元理论，依次评估犯罪行为的构成要素、非法性和故意性，以确定某人的行为是否构成犯罪。尽管现有的法律大型语言模型（LLMs）在判决预测方面展现出令人鼓舞的准确性，但由于缺乏适当的基准数据集，它们缺乏三元推理能力，因此无法预测无罪判决。因此，每一个输入都被自动分配一个罪名，限制了它们在法律环境中的实际应用价值。为了解决这一问题，我们提出了LJPIV，即首个包含无罪判决的法律判决预测基准数据集。根据三元理论，我们通过基于LLM的数据扩增和人工验证，扩展了三个广泛使用的法律数据集。我们的实验使用最新的法律LLMs和新的策略，这些策略将三元推理集成到零样本提示和微调中，结果显示：（1）当前的法律LLMs有着显著的改进空间，即使是最佳模型在LJPIV上的F1分数也低于0.3；（2）我们的策略显著提高了领域内和跨领域的判决预测准确性，特别是在预测无罪判决的案件中效果尤为明显。 

---
# Simulation-Free Hierarchical Latent Policy Planning for Proactive Dialogues 

**Title (ZH)**: 无模拟层次潜在策略规划以实现主动对话 

**Authors**: Tao He, Lizi Liao, Yixin Cao, Yuanxing Liu, Yiheng Sun, Zerui Chen, Ming Liu, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2412.14584)  

**Abstract**: Recent advancements in proactive dialogues have garnered significant attention, particularly for more complex objectives (e.g. emotion support and persuasion). Unlike traditional task-oriented dialogues, proactive dialogues demand advanced policy planning and adaptability, requiring rich scenarios and comprehensive policy repositories to develop such systems. However, existing approaches tend to rely on Large Language Models (LLMs) for user simulation and online learning, leading to biases that diverge from realistic scenarios and result in suboptimal efficiency. Moreover, these methods depend on manually defined, context-independent, coarse-grained policies, which not only incur high expert costs but also raise concerns regarding their completeness. In our work, we highlight the potential for automatically discovering policies directly from raw, real-world dialogue records. To this end, we introduce a novel dialogue policy planning framework, LDPP. It fully automates the process from mining policies in dialogue records to learning policy planning. Specifically, we employ a variant of the Variational Autoencoder to discover fine-grained policies represented as latent vectors. After automatically annotating the data with these latent policy labels, we propose an Offline Hierarchical Reinforcement Learning (RL) algorithm in the latent space to develop effective policy planning capabilities. Our experiments demonstrate that LDPP outperforms existing methods on two proactive scenarios, even surpassing ChatGPT with only a 1.8-billion-parameter LLM. 

**Abstract (ZH)**: 近年来，主动对话领域取得了显著进展，特别适用于更复杂的任务（如情绪支持和说服）。与传统的任务导向对话不同，主动对话要求进行高级策略规划和适应性，需要丰富的场景和全面的策略仓库来开发此类系统。然而，现有的方法往往依赖大型语言模型（LLMs）进行用户模拟和在线学习，这导致了与现实场景不符的偏差，从而降低了效率。此外，这些方法依赖于手动定义的、与上下文无关的、粗粒度的策略，不仅增加了专家成本，还引发了其完整性方面的担忧。在我们的工作中，我们强调了从原始的对话记录中自动发现策略的潜力。为此，我们提出了一个新的对话策略规划框架——LDPP。它完全自动化了从挖掘对话记录中的策略到学习策略规划的整个过程。具体而言，我们采用了一种变分自动编码器的变体来发现作为潜在向量表示的细粒度策略。在自动为数据添加这些潜在策略标签后，我们提出了一种在潜在空间中的脱机分层强化学习（RL）算法，以开发有效的策略规划能力。实验结果表明，LDPP在两个主动对话场景中均优于现有方法，甚至仅凭一个参数量为1.8亿的LLM便超越了ChatGPT。 

---
# CORD: Balancing COnsistency and Rank Distillation for Robust Retrieval-Augmented Generation 

**Title (ZH)**: CORD: 平衡一致性与排名蒸馏以实现稳健的检索增强生成 

**Authors**: Youngwon Lee, Seung-won Hwang, Daniel Campos, Filip Graliński, Zhewei Yao, Yuxiong He  

**Link**: [PDF](https://arxiv.org/pdf/2412.14581)  

**Abstract**: With the adoption of retrieval-augmented generation (RAG), large language models (LLMs) are expected to ground their generation to the retrieved contexts. Yet, this is hindered by position bias of LLMs, failing to evenly attend to all contexts. Previous work has addressed this by synthesizing contexts with perturbed positions of gold segment, creating a position-diversified train set. We extend this intuition to propose consistency regularization with augmentation and distillation. First, we augment each training instance with its position perturbation to encourage consistent predictions, regardless of ordering. We also distill behaviors of this pair, although it can be counterproductive in certain RAG scenarios where the given order from the retriever is crucial for generation quality. We thus propose CORD, balancing COnsistency and Rank Distillation. CORD adaptively samples noise-controlled perturbations from an interpolation space, ensuring both consistency and respect for the rank prior. Empirical results show this balance enables CORD to outperform consistently in diverse RAG benchmarks. 

**Abstract (ZH)**: 随着检索增强生成（RAG）技术的应用，大规模语言模型（LLMs）被期望将其生成内容与检索到的上下文联系起来。然而，这一目标受到LLMs位置偏见的阻碍，使其难以平等地关注所有上下文。此前的研究通过合成具有扰动位置的黄金片段，创建一个位置多样化的数据集，来解决这一问题。我们借鉴这一思路，提出了包含增强和蒸馏的一致性正则化方法。首先是增强每个训练实例的每种位置扰动，以促进一致预测，而不考虑顺序问题。同时，我们对这一对的行为进行蒸馏，尽管在某些RAG场景中，检索器给出的顺序对于生成质量至关重要，这可能会适得其反。因此，我们提出了CORD（Consistency and Rank Distillation），该方法在保持一致性和尊重排名先验的同时，实现了二者的平衡。实验证明，这种平衡使CORD在各种RAG基准测试中表现出色。 

---
# CitaLaw: Enhancing LLM with Citations in Legal Domain 

**Title (ZH)**: CitaLaw：在法律领域增强语言模型的引注技术 

**Authors**: Kepu Zhang, Weijie Yu, Sunhao Dai, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2412.14556)  

**Abstract**: In this paper, we propose CitaLaw, the first benchmark designed to evaluate LLMs' ability to produce legally sound responses with appropriate citations. CitaLaw features a diverse set of legal questions for both laypersons and practitioners, paired with a comprehensive corpus of law articles and precedent cases as a reference pool. This framework enables LLM-based systems to retrieve supporting citations from the reference corpus and align these citations with the corresponding sentences in their responses. Moreover, we introduce syllogism-inspired evaluation methods to assess the legal alignment between retrieved references and LLM-generated responses, as well as their consistency with user questions. Extensive experiments on 2 open-domain and 7 legal-specific LLMs demonstrate that integrating legal references substantially enhances response quality. Furthermore, our proposed syllogism-based evaluation method exhibits strong agreement with human judgments. 

**Abstract (ZH)**: 在本文中，我们提出了CitaLaw，这是第一个用于评估大规模语言模型（LLM）生成合法且具适当引用的响应能力的标准基准。CitaLaw包含了一系列适用于普通民众和专业人士的多样化法律问题，并配有完整的法律文章和先例案例的参考库。该框架使基于LLM的系统能够从参考库中检索支持引文，并将这些引文与响应中的相应句子对齐。此外，我们引入了灵感源自三段论的评估方法，用于评估检索到的参考信息与LLM生成的响应之间的法律匹配度以及其与用户问题的一致性。在2个开放领域和7个法律特定的LLM上的广泛实验表明，整合法律参考极大地提高了响应质量。此外，我们提出的方法基于三段论的评估标准与人类判断表现出很强的一致性。 

---
# ClusterTalk: Corpus Exploration Framework using Multi-Dimensional Exploratory Search 

**Title (ZH)**: ClusterTalk：基于多维探索性搜索的语料库探索框架 

**Authors**: Ashish Chouhan, Saifeldin Mandour, Michael Gertz  

**Link**: [PDF](https://arxiv.org/pdf/2412.14533)  

**Abstract**: Exploratory search of large text corpora is essential in domains like biomedical research, where large amounts of research literature are continuously generated. This paper presents ClusterTalk (The demo video and source code are available at: this https URL), a framework for corpus exploration using multi-dimensional exploratory search. Our system integrates document clustering with faceted search, allowing users to interactively refine their exploration and ask corpus and document-level queries. Compared to traditional one-dimensional search approaches like keyword search or clustering, this system improves the discoverability of information by encouraging a deeper interaction with the corpus. We demonstrate the functionality of the ClusterTalk framework based on four million PubMed abstracts for the four-year time frame. 

**Abstract (ZH)**: 在生物医学研究等领域，大规模文本语料库的探索性搜索是至关重要的，因为大量研究文献不断产生。本文介绍了一种名为ClusterTalk的框架（演示视频和源代码可访问：此链接），该框架使用多维探索性搜索进行语料库探索。我们的系统将文档聚类与细化搜索集成在一起，允许用户互动地细化其探索并提出语料库级和文档级查询。与传统的基于关键词搜索或聚类的一维搜索方法相比，该系统通过促进对语料库的更深入互动，提高了信息的可发现性。基于四年内包含四百万PubMed摘要的数据，本文演示了ClusterTalk框架的功能。 

---
# Multi-Level Optimal Transport for Universal Cross-Tokenizer Knowledge Distillation on Language Models 

**Title (ZH)**: 多层级最优 transport 在语言模型上通过跨分词器知识蒸馏的通用方法 

**Authors**: Xiao Cui, Mo Zhu, Yulei Qin, Liang Xie, Wengang Zhou, Houqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.14528)  

**Abstract**: Knowledge distillation (KD) has become a prevalent technique for compressing large language models (LLMs). Existing KD methods are constrained by the need for identical tokenizers (i.e., vocabularies) between teacher and student models, limiting their versatility in handling LLMs of different architecture families. In this paper, we introduce the Multi-Level Optimal Transport (MultiLevelOT), a novel approach that advances the optimal transport for universal cross-tokenizer knowledge distillation. Our method aligns the logit distributions of the teacher and the student at both token and sequence levels using diverse cost matrices, eliminating the need for dimensional or token-by-token correspondence. At the token level, MultiLevelOT integrates both global and local information by jointly optimizing all tokens within a sequence to enhance robustness. At the sequence level, we efficiently capture complex distribution structures of logits via the Sinkhorn distance, which approximates the Wasserstein distance for divergence measures. Extensive experiments on tasks such as extractive QA, generative QA, and summarization demonstrate that the MultiLevelOT outperforms state-of-the-art cross-tokenizer KD methods under various settings. Our approach is robust to different student and teacher models across model families, architectures, and parameter sizes. 

**Abstract (ZH)**: 知识蒸馏（KD）已成为压缩大型语言模型（LLMs）的一种普遍技术。现有的KD方法受限制于教师模型和学生模型之间需要相同的分词器（即词表），这限制了它们在处理不同架构家族的LLMs时的灵活性。在本文中，我们引入了多层次最优运输（MultiLevelOT），这是一种新颖的方法，它推进了最优运输在通用跨分词器知识蒸馏中的应用。我们的方法通过使用不同的成本矩阵，在token级别和序列级别对教师和学生的logit分布进行对齐，从而消除了维度或逐token对应的需求。在token级别，MultiLevelOT通过联合优化序列中的所有token来整合全局和局部信息，以增强鲁棒性。在序列级别，我们利用Sinkhorn距离高效地捕获logit的复杂分布结构，Sinkhorn距离近似 Wasserstein距离，用作离散化度量。在诸如提取式问答、生成式问答和摘要等任务中的广泛实验表明，在各种设置下，MultiLevelOT均优于最先进的跨分词器KD方法。我们的方法在不同架构家族、模型结构和参数规模的学生模型和教师模型上具有鲁棒性。 

---
# PA-RAG: RAG Alignment via Multi-Perspective Preference Optimization 

**Title (ZH)**: PA-RAG：多视角偏好优化下的RAG对齐 

**Authors**: Jiayi Wu, Hengyi Cai, Lingyong Yan, Hao Sun, Xiang Li, Shuaiqiang Wang, Dawei Yin, Ming Gao  

**Link**: [PDF](https://arxiv.org/pdf/2412.14510)  

**Abstract**: The emergence of Retrieval-augmented generation (RAG) has alleviated the issues of outdated and hallucinatory content in the generation of large language models (LLMs), yet it still reveals numerous limitations. When a general-purpose LLM serves as the RAG generator, it often suffers from inadequate response informativeness, response robustness, and citation quality. Past approaches to tackle these limitations, either by incorporating additional steps beyond generating responses or optimizing the generator through supervised fine-tuning (SFT), still failed to align with the RAG requirement thoroughly. Consequently, optimizing the RAG generator from multiple preference perspectives while maintaining its end-to-end LLM form remains a challenge. To bridge this gap, we propose Multiple Perspective Preference Alignment for Retrieval-Augmented Generation (PA-RAG), a method for optimizing the generator of RAG systems to align with RAG requirements comprehensively. Specifically, we construct high-quality instruction fine-tuning data and multi-perspective preference data by sampling varied quality responses from the generator across different prompt documents quality scenarios. Subsequently, we optimize the generator using SFT and Direct Preference Optimization (DPO). Extensive experiments conducted on four question-answer datasets across three LLMs demonstrate that PA-RAG can significantly enhance the performance of RAG generators. Our code and datasets are available at this https URL. 

**Abstract (ZH)**: 检索增强生成（RAG）的出现缓解了大型语言模型（LLMs）生成过程中出现内容过时和幻觉的问题，但仍暴露了诸多局限性。当通用的LLM用作RAG生成器时，它通常会遭受响应信息不足、响应稳健性差和引文质量低的问题。过去为解决这些局限性的方法，要么通过生成响应之外的额外步骤，要么通过监督微调（SFT）来优化生成器，都没有完全满足RAG的要求。因此，从多方面优化RAG生成器同时保持其端到端的大语言模型形式仍是一项挑战。为了填补这一空白，我们提出了一种多视角偏好对齐方法——PA-RAG（偏好对齐的RAG），该方法旨在全面优化RAG系统的生成器以满足RAG的要求。具体来说，我们通过在不同提示文档质量情景下采样生成器生成的多种质量的指令，构建高质量的指令微调数据和多视角偏好数据。随后，我们使用监督微调（SFT）和直接偏好优化（DPO）来优化生成器。在三个LLM上进行的四个问答数据集的广泛实验表明，PA-RAG能够显著提高RAG生成器的性能。我们的代码和数据集可在此处访问：[此链接]。 

---
# Do Large Language Models Defend Inferentialist Semantics?: On the Logical Expressivism and Anti-Representationalism of LLMs 

**Title (ZH)**: 大规模语言模型是否捍卫了推理主义者语义学？论LLMs的逻辑表达主义和反表征主义 

**Authors**: Yuzuki Arai, Sho Tsugawa  

**Link**: [PDF](https://arxiv.org/pdf/2412.14501)  

**Abstract**: The philosophy of language, which has historically been developed through an anthropocentric lens, is now being forced to move towards post-anthropocentrism due to the advent of large language models (LLMs) like ChatGPT (OpenAI), Claude (Anthropic), which are considered to possess linguistic abilities comparable to those of humans. Traditionally, LLMs have been explained through distributional semantics as their foundational semantics. However, recent research is exploring alternative foundational semantics beyond distributional semantics. This paper proposes Robert Brandom's inferentialist semantics as an suitable foundational semantics for LLMs, specifically focusing on the issue of linguistic representationalism within this post-anthropocentric trend. Here, we show that the anti-representationalism and logical expressivism of inferential semantics, as well as quasi-compositionality, are useful in interpreting the characteristics and behaviors of LLMs. Further, we propose a \emph{consensus theory of truths} for LLMs. This paper argues that the characteristics of LLMs challenge mainstream assumptions in philosophy of language, such as semantic externalism and compositionality. We believe the argument in this paper leads to a re-evaluation of anti\hyphen{}representationalist views of language, potentially leading to new developments in the philosophy of language. 

**Abstract (ZH)**: 语言哲学自历史上的anthropocentric（人类中心主义）视角发展而来，现在由于大型语言模型（LLMs）如ChatGPT（OpenAI）和Claude（Anthropic）的出现，被迫向后anthropocentric（去人类中心主义）方向转变。这些LLMs被认为具有与人类相当的语言能力。传统上，LLMs主要通过分布语义学作为其基础语义进行解释。然而，近期的研究正在探索超越分布语义学的替代性基础语义。本文提议罗伯特·布朗姆诺（Robert Brandom）的推理语义学作为LLMs基础语义的合适选择，特别是在这一去人类中心主义趋势下关注语言的表征问题。本文展示了推理语义学的反表征论和逻辑表达主义以及准组合性在解释LLMs的特性和行为中的有用性。此外，本文提出了用于LLMs的一种“共识真理理论”。本文认为，LLMs的特性挑战了语言哲学中的主流假设，如语义外部性和组合性。我们相信本文的观点将促进对语言的反表征论的重新评估，可能导致语言哲学中的新发展。 

---
# Why We Build Local Large Language Models: An Observational Analysis from 35 Japanese and Multilingual LLMs 

**Title (ZH)**: 我们构建本地大型语言模型的原因：来自35个日语和多语言LLM的观察性分析 

**Authors**: Koshiro Saito, Sakae Mizuki, Masanari Ohi, Taishi Nakamura, Taihei Shiotani, Koki Maeda, Youmi Ma, Kakeru Hattori, Kazuki Fujii, Takumi Okamoto, Shigeki Ishida, Hiroya Takamura, Rio Yokota, Naoaki Okazaki  

**Link**: [PDF](https://arxiv.org/pdf/2412.14471)  

**Abstract**: Why do we build local large language models (LLMs)? What should a local LLM learn from the target language? Which abilities can be transferred from other languages? Do language-specific scaling laws exist? To explore these research questions, we evaluated 35 Japanese, English, and multilingual LLMs on 19 evaluation benchmarks for Japanese and English, taking Japanese as a local language. Adopting an observational approach, we analyzed correlations of benchmark scores, and conducted principal component analysis (PCA) on the scores to derive \textit{ability factors} of local LLMs. We found that training on English text can improve the scores of academic subjects in Japanese (JMMLU). In addition, it is unnecessary to specifically train on Japanese text to enhance abilities for solving Japanese code generation, arithmetic reasoning, commonsense, and reading comprehension tasks. In contrast, training on Japanese text could improve question-answering tasks about Japanese knowledge and English-Japanese translation, which indicates that abilities for solving these two tasks can be regarded as \textit{Japanese abilities} for LLMs. Furthermore, we confirmed that the Japanese abilities scale with the computational budget for Japanese text. 

**Abstract (ZH)**: 我们为什么要构建本地大型语言模型（LLMs）？本地LLMs应该从目标语言中学到哪些能力？其他语言的能力能被转移到目标语言中吗？特定语言是否存在规模扩展定律？为了探索这些问题，我们对35个日语、英语和多语言LLMs在19个日语和英语评估基准测试中进行了评估，将日语作为本地语言进行研究。通过观测方法，我们分析了基准测试分数的相关性，并通过对分数进行主成分分析（PCA）来推导出本地LLMs的能力因子。我们发现，使用英语文本进行训练可以提高日语学术科目（JMMLU）的分数。此外，增强解决日语代码生成、算术推理、常识和阅读理解任务的能力，不需要专门使用日语文本进行训练。相比之下，使用日语文本进行训练可以在日语知识问答任务和英语-日语翻译任务上提高性能，这表明解决这两种任务的能力可以被视为LLMs的“日语能力”。此外，我们确认了日语能力与日语文本的计算预算之间存在规模扩展关系。 

---
# Agent-SafetyBench: Evaluating the Safety of LLM Agents 

**Title (ZH)**: Agent-SafetyBench: 评估大规模语言模型代理的安全性 

**Authors**: Zhexin Zhang, Shiyao Cui, Yida Lu, Jingzhuo Zhou, Junxiao Yang, Hongning Wang, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2412.14470)  

**Abstract**: As large language models (LLMs) are increasingly deployed as agents, their integration into interactive environments and tool use introduce new safety challenges beyond those associated with the models themselves. However, the absence of comprehensive benchmarks for evaluating agent safety presents a significant barrier to effective assessment and further improvement. In this paper, we introduce Agent-SafetyBench, a comprehensive benchmark designed to evaluate the safety of LLM agents. Agent-SafetyBench encompasses 349 interaction environments and 2,000 test cases, evaluating 8 categories of safety risks and covering 10 common failure modes frequently encountered in unsafe interactions. Our evaluation of 16 popular LLM agents reveals a concerning result: none of the agents achieves a safety score above 60%. This highlights significant safety challenges in LLM agents and underscores the considerable need for improvement. Through quantitative analysis, we identify critical failure modes and summarize two fundamental safety detects in current LLM agents: lack of robustness and lack of risk awareness. Furthermore, our findings suggest that reliance on defense prompts alone is insufficient to address these safety issues, emphasizing the need for more advanced and robust strategies. We release Agent-SafetyBench at \url{this https URL} to facilitate further research and innovation in agent safety evaluation and improvement. 

**Abstract (ZH)**: 随着大规模语言模型（LLMs）被越来越多地用作代理角色，它们在交互环境中的整合和工具使用引入了与模型本身相关的新安全挑战。然而，缺乏全面的评估标准来衡量代理安全性，这成为有效评估和进一步改进的重要障碍。本文介绍了一种名为Agent-SafetyBench的综合性基准，旨在评估LLM代理的安全性。Agent-SafetyBench涵盖了349种交互环境和2000个测试案例，评估了8类安全风险，并涵盖了10种常见的不安全交互中频繁出现的失败模式。我们的评估显示了16种流行LLM代理的结果：没有一个代理的安全评分超过60%。这一结果突出了LLM代理在安全性方面的重要挑战，并强调了改进的迫切需求。通过定量分析，我们识别了关键的失败模式，并总结了当前LLM代理中的两种基本安全检测：缺乏鲁棒性和缺乏风险意识。此外，我们的研究结果表明，仅仅依赖防御提示是不足以解决这些安全问题的，强调了需要采取更先进和可靠的策略。我们在此处发布Agent-SafetyBench（\url{this https URL}），以便进一步促进代理安全性评估和改进的研究与创新。 

---
# From Human Annotation to LLMs: SILICON Annotation Workflow for Management Research 

**Title (ZH)**: 从人工标注到LLM：SILICON标注工作流程在管理研究中的应用 

**Authors**: Xiang Cheng, Raveesh Mayya, João Sedoc  

**Link**: [PDF](https://arxiv.org/pdf/2412.14461)  

**Abstract**: Unstructured text data annotation and analysis are fundamental to management research, often relying on human annotators through crowdsourcing platforms. While Large Language Models (LLMs) promise to provide a cost-effective and efficient alternative to human annotation, there lacks a systematic workflow that evaluate when LLMs are suitable or how to proceed with LLM-based text annotation in a reproducible manner. This paper addresses this methodological gap by introducing the ``SILICON" (\textbf{S}ystematic \textbf{I}nference with \textbf{L}LMs for \textbf{I}nformation \textbf{C}lassificati\textbf{o}n and \textbf{N}otation) workflow. The workflow integrates established principles of human annotation with systematic prompt optimization and model selection, addressing challenges such as developing robust annotation guidelines, establishing high-quality human baselines, optimizing prompts, and ensuring reproducibility across LLMs. We validate the SILICON workflow through seven case studies covering common management research tasks, including business proposal evaluation, dialog intent and breakdown analysis, review attribute detection. Our findings highlight the importance of validating annotation guideline agreement, the superiority of expert-developed human baselines over crowdsourced ones, the iterative nature of prompt optimization, and the necessity of testing multiple LLMs. Notably, we propose a regression-based methodology to empirically compare LLM outputs across prompts and models. Our workflow advances management research by establishing reproducible processes for LLM-based annotation that maintain scientific rigor. We provide practical guidance for researchers to effectively navigate the evolving landscape of generative AI tools effectively while maintaining transparency and reproducibility. 

**Abstract (ZH)**: 无结构文本数据的标注和分析是管理研究的基础，通常依赖于通过众包平台的人工标注者。虽然大型语言模型（LLMs）提供了成本效益更高且更高效的替代方案，但缺乏系统的工作流程来评估LLMs的适用性或在具有可重复性的情况下如何进行基于LLM的文本标注。本文通过引入“SILICON”（系统化的LLMs在信息分类与标注中的推断）工作流程解决了这一方法论上的缺口。该工作流程将人类标注的基本原则与系统化的提示优化和模型选择相结合，解决了诸如制定稳健的标注指南、建立高质量的人类基准、优化提示以及在不同LLM之间确保可重复性等挑战。

我们通过七个案例研究验证了SILICON工作流程，涵盖常见的管理研究任务，包括商业提案评估、对话意图和分析、评论属性检测等。我们的研究结果强调了验证标注指南一致性的重要性、专家开发的人类基准优于众包基准、提示优化的迭代性质以及测试多种LLM的必要性。值得注意的是，我们提出了基于回归的方法来实证比较不同提示和模型下的LLM输出效果。我们的工作流程推动了管理研究的进步，通过建立基于LLM的标注的可重复过程，保持科学严谨性。

我们为研究人员提供实用的指导，帮助他们有效地利用不断发展中的生成性AI工具，同时保持透明性和可重复性。 

---
# ORBIT: Cost-Effective Dataset Curation for Large Language Model Domain Adaptation with an Astronomy Case Study 

**Title (ZH)**: ORBIT：用于大型语言模型领域适配的成本效益数据集整理——以天文学案例研究为例

这个翻译符合学术规范，保留了原文的核心信息和专业术语。 

**Authors**: Eric Modesitt, Ke Yang, Spencer Hulsey, Chengxiang Zhai, Volodymyr Kindratenko  

**Link**: [PDF](https://arxiv.org/pdf/2412.14436)  

**Abstract**: Recent advances in language modeling demonstrate the need for high-quality domain-specific training data, especially for tasks that require specialized knowledge. General-purpose models, while versatile, often lack the depth needed for expert-level tasks because of limited domain-specific information. Domain adaptation training can enhance these models, but it demands substantial, high-quality data. To address this, we propose ORBIT, a cost-efficient methodology for curating massive, high-quality domain-specific datasets from noisy web sources, tailored for training specialist large language models. Using astronomy as a primary case study, we refined the 1.3T-token FineWeb-Edu dataset into a high-quality, 10B-token subset focused on astronomy. Fine-tuning \textsc{LLaMA-3-8B} on a 1B-token astronomy subset improved performance on the MMLU astronomy benchmark from 69\% to 76\% and achieved top results on AstroBench, an astronomy-specific benchmark. Moreover, our model (Orbit-LLaMA) outperformed \textsc{LLaMA-3-8B-base}, with GPT-4o evaluations preferring it in 73\% of cases across 1000 astronomy-specific questions. Additionally, we validated ORBIT's generalizability by applying it to law and medicine, achieving a significant improvement of data quality compared to an unfiltered baseline. We open-source the ORBIT methodology, including the curated datasets, the codebase, and the resulting model at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 近年来，语言模型的发展表明，高质量的专业领域训练数据对于完成需要专门知识的任务至关重要。通用模型虽然用途广泛，但往往因缺乏特定领域的信息而无法胜任专家级任务。领域适应训练可以提升这些模型，但需要大量的高质量数据。为了解决这一问题，我们提出了一种名为ORBIT的成本效益方法，用于从嘈杂的网络源中整理出大规模的高质量专业领域数据集，专门用于训练专业大型语言模型。以天文学为主要案例研究，我们将1.3万亿令牌的FineWeb-Edu数据集精炼成一个关注天文学的、高质量的100亿令牌子集。通过在10亿令牌的天文学子集上对\textsc{LLaMA-3-8B}进行微调，模型在MMLU天文学基准测试中的性能从69%提升到76%，并在专为天文学设计的AstroBench基准测试中取得了最佳成绩。此外，我们的模型Orbit-LLaMA在天文学特定问题上的表现优于\textsc{LLaMA-3-8B-base}，根据GPT-4o的评估，在1000个特定于天文学的问题中，有73%的情况偏好我们的模型。我们还验证了ORBIT方法的通用性，将其应用于法律和医学领域，相比未经过滤的基线数据，显著提高了数据质量。我们公开了ORBIT方法，包括整理好的数据集、代码库以及生成的模型，链接如下：\href{this https URL}{this https URL}。 

---
# All-in-One Tuning and Structural Pruning for Domain-Specific LLMs 

**Title (ZH)**: 面向特定领域的大型语言模型的一体化调优与结构剪枝 

**Authors**: Lei Lu, Zhepeng Wang, Ruexue Bao, Mengbing Wang, Fangyi Li, Yawen Wu, Weiwen Jiang, Jie Xu, Yanzhi Wang, Shangqian Gao  

**Link**: [PDF](https://arxiv.org/pdf/2412.14426)  

**Abstract**: Existing pruning techniques for large language models (LLMs) targeting domain-specific applications typically follow a two-stage process: pruning the pretrained general-purpose LLMs and then fine-tuning the pruned LLMs on specific domains. However, the pruning decisions, derived from the pretrained weights, remain unchanged during fine-tuning, even if the weights have been updated. Therefore, such a combination of the pruning decisions and the finetuned weights may be suboptimal, leading to non-negligible performance degradation. To address these limitations, we propose ATP: All-in-One Tuning and Structural Pruning, a unified one-stage structural pruning and fine-tuning approach that dynamically identifies the current optimal substructure throughout the fine-tuning phase via a trainable pruning decision generator. Moreover, given the limited available data for domain-specific applications, Low-Rank Adaptation (LoRA) becomes a common technique to fine-tune the LLMs. In ATP, we introduce LoRA-aware forward and sparsity regularization to ensure that the substructures corresponding to the learned pruning decisions can be directly removed after the ATP process. ATP outperforms the state-of-the-art two-stage pruning methods on tasks in the legal and healthcare domains. More specifically, ATP recovers up to 88% and 91% performance of the dense model when pruning 40% parameters of LLaMA2-7B and LLaMA3-8B models, respectively. 

**Abstract (ZH)**: 现有的针对特定领域应用的大语言模型（LLMs）的裁剪技术通常遵循一个两阶段过程：首先对预训练的一般用途LLMs进行裁剪，然后再对裁剪后的模型进行特定领域的微调。然而，从预训练权重中得出的裁剪决策在微调过程中保持不变，即使权重已被更新。因此，这种裁剪决策与微调权重相结合的方式可能是次优的，导致性能下降。为了解决这些限制，我们提出了一种名为ATP（All-in-One Tuning and Structural Pruning）的统一方法，该方法结合了一次性的结构裁剪和微调步骤，通过可训练的裁剪决策生成器，在微调过程中动态识别当前最优的子结构。此外，由于特定领域应用的数据有限，低秩适应（LoRA）成为了一种常用的对LLMs进行微调的技术。在ATP中，我们引入了LoRA意识的前向传播和稀疏正则化，以确保能够在ATP处理后直接移除与学习到的裁剪决策对应的子结构。ATP在法律和医疗领域任务中优于最先进的两阶段裁剪方法。具体而言，当对LLaMA2-7B和LLaMA3-8B模型裁剪40%的参数时，ATP分别恢复了密集模型88%和91%的性能。 

---
# ECG-Byte: A Tokenizer for End-to-End Generative Electrocardiogram Language Modeling 

**Title (ZH)**: ECG-Byte: 一种用于端到端心电图语言建模的分词器 

**Authors**: William Han, Chaojing Duan, Michael A. Rosenberg, Emerson Liu, Ding Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2412.14373)  

**Abstract**: Large Language Models (LLMs) have shown remarkable adaptability across domains beyond text, specifically electrocardiograms (ECGs). More specifically, there is a growing body of work exploring the task of generating text from a multi-channeled ECG and corresponding textual prompt. Current approaches typically involve pretraining an ECG-specific encoder with a self-supervised learning (SSL) objective and using the features output by the pretrained encoder to finetune a LLM for natural language generation (NLG). However, these methods are limited by 1) inefficiency from two-stage training and 2) interpretability challenges with encoder-generated features. To address these limitations, we introduce ECG-Byte, an adapted byte pair encoding (BPE) tokenizer pipeline for autoregressive language modeling of ECGs. This approach compresses and encodes ECG signals into tokens, enabling end-to-end LLM training by combining ECG and text tokens directly, while being much more interpretable since the ECG tokens can be directly mapped back to the original signal. Using ECG-Byte, we achieve competitive performance in NLG tasks in only half the time and ~48% of the data required by two-stage approaches. 

**Abstract (ZH)**: 大语言模型（LLMs）在文本之外的领域也表现出了惊人的适应能力，特别是在心电图（ECGs）方面。具体来说，越来越多的研究致力于从多通道心电图和相应的文本提示中生成文本的任务。当前的方法通常涉及使用自我监督学习（SSL）目标预先训练一个特定于心电图的编码器，并使用该预训练编码器输出的特征来微调一个语言模型以进行自然语言生成（NLG）。然而，这些方法受到了 1）双阶段训练的低效性以及 2）编码器生成特征的可解释性挑战的限制。为了应对这些限制，我们提出了ECG-Byte，这是一种适应性的字节对编码（BPE）分词流水线，专门用于心电图的自回归语言建模。这种方法将心电图信号压缩并编码为分词，从而可以在直接结合心电图和文本分词的同时实现端到端的LLM训练，同时由于心电图分词可以直接映射回原始信号，因此更具可解释性。使用ECG-Byte方法，我们仅用双阶段方法所需时间的一半及大约48%的数据实现了与双阶段方法相当的NLG任务性能。 

---
# Memorization Over Reasoning? Exposing and Mitigating Verbatim Memorization in Large Language Models' Character Understanding Evaluation 

**Title (ZH)**: 记忆胜过推理？揭示并减轻大型语言模型在角色理解评估中逐字记忆的问题 

**Authors**: Yuxuan Jiang, Francis Ferraro  

**Link**: [PDF](https://arxiv.org/pdf/2412.14368)  

**Abstract**: Recently, Large Language Models (LLMs) have shown impressive performance in character understanding tasks, such as analyzing the roles, personalities, and relationships of fictional characters. However, the extensive pre-training corpora used by LLMs raise concerns that they may rely on memorizing popular fictional works rather than genuinely understanding and reasoning about them. In this work, we argue that 'gist memory'-capturing essential meaning - should be the primary mechanism for character understanding tasks, as opposed to 'verbatim memory' - exact match of a string. We introduce a simple yet effective method to mitigate mechanized memorization in character understanding evaluations while preserving the essential implicit cues needed for comprehension and reasoning. Our approach reduces memorization-driven performance on popular fictional works from 96% accuracy to 72% and results in up to an 18% drop in accuracy across various character understanding tasks. These findings underscore the issue of data contamination in existing benchmarks, which often measure memorization rather than true character understanding. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）在人物理解任务中展示了令人印象深刻的性能，例如分析虚构人物的角色、个性及其关系。然而，LLMs所使用的大量预训练语料库引发了担忧，它们可能依赖于记忆流行的小说作品，而非真正理解和推理。在本文中，我们主张“纲要记忆”——即捕捉核心意义——应成为人物理解任务中的主要机制，而非“逐字记忆”——即字符串的精确匹配。我们提出了一种简单而有效的解决方案，以减轻在人物理解评估中对机械记忆的依赖，同时保留用于理解和推理所必需的核心暗示信息。我们的方法将对流行虚构作品的机械记忆驱动性能从96%的准确率降低到72%，并在各种人物理解任务中导致高达18%的准确率下降。这些发现凸显了现有基准数据污染的问题，这些基准往往测量的是记忆而非真正的角色理解。 

---
# State Space Models are Strong Text Rerankers 

**Title (ZH)**: 状态空间模型是强大的文本重排器 

**Authors**: Zhichao Xu, Jinghua Yan, Ashim Gupta, Vivek Srikumar  

**Link**: [PDF](https://arxiv.org/pdf/2412.14354)  

**Abstract**: Transformers dominate NLP and IR; but their inference inefficiencies and challenges in extrapolating to longer contexts have sparked interest in alternative model architectures. Among these, state space models (SSMs) like Mamba offer promising advantages, particularly $O(1)$ time complexity in inference. Despite their potential, SSMs' effectiveness at text reranking -- a task requiring fine-grained query-document interaction and long-context understanding -- remains underexplored.
This study benchmarks SSM-based architectures (specifically, Mamba-1 and Mamba-2) against transformer-based models across various scales, architectures, and pre-training objectives, focusing on performance and efficiency in text reranking tasks. We find that (1) Mamba architectures achieve competitive text ranking performance, comparable to transformer-based models of similar size; (2) they are less efficient in training and inference compared to transformers with flash attention; and (3) Mamba-2 outperforms Mamba-1 in both performance and efficiency. These results underscore the potential of state space models as a transformer alternative and highlight areas for improvement in future IR applications. 

**Abstract (ZH)**: 变压器在自然语言处理（NLP）和信息检索（IR）领域占据主导地位，但它们在推断效率上的不足以及在长上下文推理中的挑战激发了对替代模型架构的兴趣。在这些替代模型中，如Mamba这样的状态空间模型（SSMs）展现出诱人的优势，尤其是在推理中提供恒定时间复杂度的$O(1)$表现。尽管如此，状态空间模型在文本重排序任务中的有效性——这一任务需要精细的查询-文档交互和对长上下文的理解——仍鲜有探索。

本研究评估了基于状态空间模型的架构（具体而言是Mamba-1和Mamba-2）与基于变压器的模型在不同规模、架构和预训练目标下的表现，重点关注文本重排序任务中的性能和效率。我们发现：（1）Mamba架构在文本排名性能上达到了与相似规模的变压器模型相当的水平；（2）它们在训练和推断效率上不如使用快速注意机制的变压器模型；（3）Mamba-2在性能和效率上优于Mamba-1。这些结果强调了状态空间模型作为变压器替代方案的潜在价值，并指出了未来在信息检索应用中需要改进的领域。 

---
# A Survey on LLM Inference-Time Self-Improvement 

**Title (ZH)**: LLM 推理时自我改进综述 

**Authors**: Xiangjue Dong, Maria Teleki, James Caverlee  

**Link**: [PDF](https://arxiv.org/pdf/2412.14352)  

**Abstract**: Techniques that enhance inference through increased computation at test-time have recently gained attention. In this survey, we investigate the current state of LLM Inference-Time Self-Improvement from three different perspectives: Independent Self-improvement, focusing on enhancements via decoding or sampling methods; Context-Aware Self-Improvement, leveraging additional context or datastore; and Model-Aided Self-Improvement, achieving improvement through model collaboration. We provide a comprehensive review of recent relevant studies, contribute an in-depth taxonomy, and discuss challenges and limitations, offering insights for future research. 

**Abstract (ZH)**: 在测试时通过增加计算来增强推理的技术最近受到了广泛关注。本文从三个不同的视角对该领域的现状进行了调查：独立自我改进，重点关注通过解码或采样方法的增强；上下文感知自我改进，利用额外的上下文或数据存储；以及模型辅助自我改进，通过模型协作实现改进。我们对最近的相关研究进行了全面回顾，贡献了一套深入的分类体系，并讨论了面临的挑战和局限性，为未来的研究提供了见解。 

---
# Is Peer-Reviewing Worth the Effort? 

**Title (ZH)**: 评审工作值得付出努力吗？ 

**Authors**: Kenneth Church, Raman Chandrasekar, John E. Ortega, Ibrahim Said Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2412.14351)  

**Abstract**: How effective is peer-reviewing in identifying important papers? We treat this question as a forecasting task. Can we predict which papers will be highly cited in the future based on venue and "early returns" (citations soon after publication)? We show early returns are more predictive than venue. Finally, we end with constructive suggestions to address scaling challenges: (a) too many submissions and (b) too few qualified reviewers. 

**Abstract (ZH)**: 同行评审在识别重要论文方面有多有效？我们将这个问题视为一项预测任务。我们能否根据会议venue和“早期反馈”（即发表后不久的引用情况）来预测哪些论文将会在未来被高引用？我们表明，早期反馈比venue更能预测引用情况。最后，我们提出了一些建设性的建议以应对规模挑战：(a) 评审投稿过多，(b) 评审人资源不足。 

---
# Semantic Role Labeling of NomBank Partitives 

**Title (ZH)**: NomBank 部分词项的语义角色标注 

**Authors**: Adam Meyers, Advait Pravin Savant, John E. Ortega  

**Link**: [PDF](https://arxiv.org/pdf/2412.14328)  

**Abstract**: This article is about Semantic Role Labeling for English partitive nouns (5%/REL of the price/ARG1; The price/ARG1 rose 5 percent/REL) in the NomBank annotated corpus. Several systems are described using traditional and transformer-based machine learning, as well as ensembling. Our highest scoring system achieves an F1 of 91.74% using "gold" parses from the Penn Treebank and 91.12% when using the Berkeley Neural parser. This research includes both classroom and experimental settings for system development. 

**Abstract (ZH)**: 本文探讨了英语部分名词（如“5%/REL of the price/ARG1；价格/ARG1上涨了5个百分点/REL”）的语义角色标注问题，研究基于NomBank标注语料库进行。文中描述了几种使用传统机器学习和变换器模型的方法，并采用了集成学习。我们的最优系统在使用宾夕法尼亚树库的“金标准”解析时获得了91.74%的F1值，在使用伯克利神经句法解析器时获得了91.12%的F1值。本研究涵盖了系统开发的课堂和实验设置。 

---
# The Role of Handling Attributive Nouns in Improving Chinese-To-English Machine Translation 

**Title (ZH)**: 改进中文到英文机器翻译中属性名词处理的作用 

**Authors**: Haohao, Wang, Adam Meyers, John E. Ortega, Rodolfo Zevallos  

**Link**: [PDF](https://arxiv.org/pdf/2412.14323)  

**Abstract**: Translating between languages with drastically different grammatical conventions poses challenges, not just for human interpreters but also for machine translation systems. In this work, we specifically target the translation challenges posed by attributive nouns in Chinese, which frequently cause ambiguities in English translation. By manually inserting the omitted particle X ('DE'). In news article titles from the Penn Chinese Discourse Treebank, we developed a targeted dataset to fine-tune Hugging Face Chinese to English translation models, specifically improving how this critical function word is handled. This focused approach not only complements the broader strategies suggested by previous studies but also offers a practical enhancement by specifically addressing a common error type in Chinese-English translation. 

**Abstract (ZH)**: 将具有巨大语法差异的语言进行互译既对人工译者构成挑战，也对机器翻译系统构成挑战。本研究特别关注汉语中属性名词带来的翻译难题，这些名词在英译时经常导致模糊性。通过人工插入缺失的助词“的”（X），我们基于宾夕法尼亚中文语篇树库中的新闻文章标题，构建了一个专门的数据集，以改进Hugging Face的中文到英文翻译模型。这一改善特别是通过更好地处理这一关键功能词来实现。这种聚焦的方法不仅补充了先前研究中提出的更广泛策略，还通过特别解决中英文翻译中的常见错误类型，提供了一种实用的改进方案。 

---
# Multi-OphthaLingua: A Multilingual Benchmark for Assessing and Debiasing LLM Ophthalmological QA in LMICs 

**Title (ZH)**: 多语眼科语言：用于评估和去偏见低收入和中等收入国家（LMICs）的LLM眼科问答的多语言基准 

**Authors**: David Restrepo, Chenwei Wu, Zhengxu Tang, Zitao Shuai, Thao Nguyen Minh Phan, Jun-En Ding, Cong-Tinh Dao, Jack Gallifant, Robyn Gayle Dychiao, Jose Carlo Artiaga, André Hiroshi Bando, Carolina Pelegrini Barbosa Gracitelli, Vincenz Ferrer, Leo Anthony Celi, Danielle Bitterman, Michael G Morley, Luis Filipe Nakayama  

**Link**: [PDF](https://arxiv.org/pdf/2412.14304)  

**Abstract**: Current ophthalmology clinical workflows are plagued by over-referrals, long waits, and complex and heterogeneous medical records. Large language models (LLMs) present a promising solution to automate various procedures such as triaging, preliminary tests like visual acuity assessment, and report summaries. However, LLMs have demonstrated significantly varied performance across different languages in natural language question-answering tasks, potentially exacerbating healthcare disparities in Low and Middle-Income Countries (LMICs). This study introduces the first multilingual ophthalmological question-answering benchmark with manually curated questions parallel across languages, allowing for direct cross-lingual comparisons. Our evaluation of 6 popular LLMs across 7 different languages reveals substantial bias across different languages, highlighting risks for clinical deployment of LLMs in LMICs. Existing debiasing methods such as Translation Chain-of-Thought or Retrieval-augmented generation (RAG) by themselves fall short of closing this performance gap, often failing to improve performance across all languages and lacking specificity for the medical domain. To address this issue, We propose CLARA (Cross-Lingual Reflective Agentic system), a novel inference time de-biasing method leveraging retrieval augmented generation and self-verification. Our approach not only improves performance across all languages but also significantly reduces the multilingual bias gap, facilitating equitable LLM application across the globe. 

**Abstract (ZH)**: 当前的眼科临床工作流程受到过度转诊、等待时间长以及复杂多样的医疗记录的困扰。大规模语言模型（LLMs）提供了自动处理各种程序的潜在解决方案，如初步分类、视力评估等，以及报告总结等。然而，在自然语言问答任务中，LLMs在不同语言上的表现显示出显著的差异，这可能加剧了低收入和中等收入国家（LMICs）的医疗不平等现象。本研究引入了首个手工构建的多语言眼科问答基准数据集，允许在不同语言之间进行直接的跨语言比较。我们对7种不同语言下的6种流行LLMs进行的评估显示存在显著的语言偏差，突出了在LMICs中临床应用LLMs的风险。现有的去偏方法，如翻译链式思维或检索增强生成（RAG），单独使用时难以弥合这种性能差距，往往无法在所有语言中提升性能，且缺乏医学领域的特异性。为了解决这一问题，我们提出了CLARA（跨语言反思性代理系统），这是一种利用检索增强生成和自我验证的新颖推理时去偏方法。我们的方法不仅在所有语言中提高了性能，还显著减少了多语言偏差差距，促进了全球范围内LLMs的公平应用。 

---
# Fake News Detection: Comparative Evaluation of BERT-like Models and Large Language Models with Generative AI-Annotated Data 

**Title (ZH)**: 虚假新闻检测：基于BERT类模型和生成AI标注数据的大语言模型比较评估 

**Authors**: haina Raza, Drai Paulen-Patterson, Chen Ding  

**Link**: [PDF](https://arxiv.org/pdf/2412.14276)  

**Abstract**: Fake news poses a significant threat to public opinion and social stability in modern society. This study presents a comparative evaluation of BERT-like encoder-only models and autoregressive decoder-only large language models (LLMs) for fake news detection. We introduce a dataset of news articles labeled with GPT-4 assistance (an AI-labeling method) and verified by human experts to ensure reliability. Both BERT-like encoder-only models and LLMs were fine-tuned on this dataset. Additionally, we developed an instruction-tuned LLM approach with majority voting during inference for label generation. Our analysis reveals that BERT-like models generally outperform LLMs in classification tasks, while LLMs demonstrate superior robustness against text perturbations. Compared to weak labels (distant supervision) data, the results show that AI labels with human supervision achieve better classification results. This study highlights the effectiveness of combining AI-based annotation with human oversight and demonstrates the performance of different families of machine learning models for fake news detection 

**Abstract (ZH)**: 虚假信息对现代社会的公众意见和社会稳定构成了重大威胁。本研究旨在比较基于BERT的编码器-only模型和自回归解码器-only大型语言模型（LLMs）在虚假信息检测中的性能。我们引入了一个由GPT-4辅助（一种AI标注方法）并经人类专家验证的数据集，以确保数据的可靠性。两种模型（BERT-like编码器-only模型和LLMs）都在这 datasets 上进行了微调。此外，我们还开发了一种指令微调的LLM方法，在推理时采用多数投票生成标签。我们的分析表明，基于BERT的模型在分类任务中通常优于LLMs，而LLMs在对抗文本扰动方面表现出更强的鲁棒性。与弱标签（远程监督）数据相比，结果表明在人类监督下生成的AI标签能取得更好的分类效果。本研究强调了将基于AI的标注与人类监督相结合的有效性，并展示了不同家族的机器学习模型在虚假信息检测中的性能。 

---
# Tokenisation is NP-Complete 

**Title (ZH)**: 词元化是NP完全问题 

**Authors**: Philip Whittington, Gregor Bachmann, Tiago Pimentel  

**Link**: [PDF](https://arxiv.org/pdf/2412.15210)  

**Abstract**: In this work, we prove the NP-completeness of two variants of tokenisation, defined as the problem of compressing a dataset to at most $\delta$ symbols by either finding a vocabulary directly (direct tokenisation), or selecting a sequence of merge operations (bottom-up tokenisation). 

**Abstract (ZH)**: 在本文中，我们证明了两种变体的标记化问题的NP完全性，这两种变体分别是将数据集压缩至最多 $\delta$ 个符号的问题：一种是直接找到词汇表（直接标记化），另一种是选择一系列合并操作（自底向上的标记化）。 

---
# Critical-Questions-of-Thought: Steering LLM reasoning with Argumentative Querying 

**Title (ZH)**: 批判性思考问题：通过论辩性查询引导大语言模型推理 

**Authors**: Federico Castagna, Isabel Sassoon, Simon Parsons  

**Link**: [PDF](https://arxiv.org/pdf/2412.15177)  

**Abstract**: Studies have underscored how, regardless of the recent breakthrough and swift advances in AI research, even state-of-the-art Large Language models (LLMs) continue to struggle when performing logical and mathematical reasoning. The results seem to suggest that LLMs still work as (highly advanced) data pattern identifiers, scoring poorly when attempting to generalise and solve reasoning problems the models have never previously seen or that are not close to samples presented in their training data. To address this compelling concern, this paper makes use of the notion of critical questions from the literature on argumentation theory, focusing in particular on Toulmin's model of argumentation. We show that employing these critical questions can improve the reasoning capabilities of LLMs. By probing the rationale behind the models' reasoning process, the LLM can assess whether some logical mistake is occurring and correct it before providing the final reply to the user prompt. The underlying idea is drawn from the gold standard of any valid argumentative procedure: the conclusion is valid if it is entailed by accepted premises. Or, to paraphrase such Aristotelian principle in a real-world approximation, characterised by incomplete information and presumptive logic, the conclusion is valid if not proved otherwise. This approach successfully steers the models' output through a reasoning pipeline, resulting in better performance against the baseline and its Chain-of-Thought (CoT) implementation. To this end, an extensive evaluation of the proposed approach on the MT-Bench Reasoning and Math tasks across a range of LLMs is provided. 

**Abstract (ZH)**: 研究表明，尽管人工智能（AI）研究领域最近取得了突破并迅速发展，最先进的大型语言模型（LLMs）在进行逻辑和数学推理时仍然面临着困难。研究结果表明，尽管LLMs表现出高度先进的数据模式识别能力，但在尝试概括和解决未见过的新颖推理问题时（尤其是那些与训练数据样本差异较大的问题），它们的性能仍然较差。为应对这一重要挑战，本文借鉴论证理论中的批判性问题概念，特别是Toulmin的论证模型，旨在提升LLMs的推理能力。通过探究模型推理过程中的逻辑依据，LLMs可以在最终回应用户提示之前评估并纠正潜在的逻辑错误。这一理念基于任何有效论证程序的标准原则：如果结论是由被接受的前提所蕴含，则结论是有效的。换句话说，在现实世界的近似情境中，如果结论未被证明为无效，则结论即是有效的，前提是存在部分信息不完整和归纳性逻辑的情况。这种方法可以引导模型的输出顺利通过推理管道，从而在基线及其链式思维（CoT）实现的基础上提升性能。为此，本文在MT-Bench推理与数学任务上对多种LLMs进行了广泛的评估，展示并验证了该方法的有效性。 

---
# Prompt-A-Video: Prompt Your Video Diffusion Model via Preference-Aligned LLM 

**Title (ZH)**: 当然，以下是标题的中文翻译，符合学术规范：

Prompt-A-Video: 通过偏好对齐的大语言模型优化视频扩散模型 

**Authors**: Yatai Ji, Jiacheng Zhang, Jie Wu, Shilong Zhang, Shoufa Chen, Chongjian GE, Peize Sun, Weifeng Chen, Wenqi Shao, Xuefeng Xiao, Weilin Huang, Ping Luo  

**Link**: [PDF](https://arxiv.org/pdf/2412.15156)  

**Abstract**: Text-to-video models have made remarkable advancements through optimization on high-quality text-video pairs, where the textual prompts play a pivotal role in determining quality of output videos. However, achieving the desired output often entails multiple revisions and iterative inference to refine user-provided prompts. Current automatic methods for refining prompts encounter challenges such as Modality-Inconsistency, Cost-Discrepancy, and Model-Unaware when applied to text-to-video diffusion models. To address these problem, we introduce an LLM-based prompt adaptation framework, termed as Prompt-A-Video, which excels in crafting Video-Centric, Labor-Free and Preference-Aligned prompts tailored to specific video diffusion model. Our approach involves a meticulously crafted two-stage optimization and alignment system. Initially, we conduct a reward-guided prompt evolution pipeline to automatically create optimal prompts pool and leverage them for supervised fine-tuning (SFT) of the LLM. Then multi-dimensional rewards are employed to generate pairwise data for the SFT model, followed by the direct preference optimization (DPO) algorithm to further facilitate preference alignment. Through extensive experimentation and comparative analyses, we validate the effectiveness of Prompt-A-Video across diverse generation models, highlighting its potential to push the boundaries of video generation. 

**Abstract (ZH)**: 文本到视频模型通过优化高质量的文本-视频对取得了显著进展，其中文本提示在决定输出视频质量方面发挥着关键作用。然而，实现所需输出往往需要多次修订和迭代推理来细化用户提供的提示。当将现有的提示优化方法应用于文本到视频扩散模型时，它们遇到了模态不一致性（Modality-Inconsistency）、成本偏差（Cost-Discrepancy）和模型不感知（Model-Unaware）等问题。为了解决这些问题，我们提出了一种基于大语言模型（LLM）的提示适应框架，称为Prompt-A-Video，该框架能够生成面向视频、无需劳力且符合用户偏好的提示，适用于特定的视频扩散模型。我们的方法涉及一个精心构建的两阶段优化和对齐系统。首先，我们通过奖励引导的提示演化管道自动创建最优提示池，并将其用于对LLM进行监督微调（SFT）。然后，我们使用多维奖励生成监督微调模型的数据对，并采用直接偏好优化（DPO）算法进一步促进偏好对齐。通过广泛的实验和对比分析，我们验证了Prompt-A-Video在各种生成模型中的有效性，突显了其推动视频生成边界的可能性。 

---
# Associative memory inspires improvements for in-context learning using a novel attention residual stream architecture 

**Title (ZH)**: 关联记忆启发基于新型注意力残差流架构的场景内学习改进 

**Authors**: Thomas F Burns, Tomoki Fukai, Christopher J Earls  

**Link**: [PDF](https://arxiv.org/pdf/2412.15113)  

**Abstract**: Large language models (LLMs) demonstrate an impressive ability to utilise information within the context of their input sequences to appropriately respond to data unseen by the LLM during its training procedure. This ability is known as in-context learning (ICL). Humans and non-human animals demonstrate similar abilities, however their neural architectures differ substantially from LLMs. Despite this, a critical component within LLMs, the attention mechanism, resembles modern associative memory models, widely used in and influenced by the computational neuroscience community to model biological memory systems. Using this connection, we introduce an associative memory model capable of performing ICL. We use this as inspiration for a novel residual stream architecture which allows information to directly flow between attention heads. We test this architecture during training within a two-layer Transformer and show its ICL abilities manifest more quickly than without this modification. We then apply our architecture in small language models with 8 million parameters, focusing on attention head values, with results also indicating improved ICL performance at this larger and more naturalistic scale. 

**Abstract (ZH)**: 大型语言模型（LLMs）展示了在输入序列上下文中利用信息以适当响应训练过程中未见过的数据的能力。这种能力称为上下文内学习（ICL）。人类和非人类动物也表现出类似的能力，但它们的神经架构与LLMs有显著差异。尽管如此，LLMs 中的一个关键组成部分——注意机制——与现代联想记忆模型相似，这些模型在计算神经科学领域广泛使用并受其影响，用于建模生物记忆系统。利用这种联系，我们提出了一种能够执行ICL的联想记忆模型。我们利用这一灵感设计了一种新型残差流架构，允许信息直接在注意头之间流动。我们在这项研究中将该架构应用于两层Transformer的训练过程中，并表明在没有这种修改的情况下，ICL能力的表现要慢一些。然后，我们将在具有800万个参数的小型语言模型中应用该架构，重点研究注意头值，实验结果也表明，在较大的、更自然的规模下，这种架构的ICL性能有所提高。 

---
# A Cross-Domain Study of the Use of Persuasion Techniques in Online Disinformation 

**Title (ZH)**: 跨域研究：在线 misinformation 中说服技术的运用探讨 

**Authors**: João A. Leite, Olesya Razuvayevskaya, Carolina Scarton, Kalina Bontcheva  

**Link**: [PDF](https://arxiv.org/pdf/2412.15098)  

**Abstract**: Disinformation, irrespective of domain or language, aims to deceive or manipulate public opinion, typically through employing advanced persuasion techniques. Qualitative and quantitative research on the weaponisation of persuasion techniques in disinformation has been mostly topic-specific (e.g., COVID-19) with limited cross-domain studies, resulting in a lack of comprehensive understanding of these strategies. This study employs a state-of-the-art persuasion technique classifier to conduct a large-scale, multi-domain analysis of the role of 16 persuasion techniques in disinformation narratives. It shows how different persuasion techniques are employed disproportionately in different disinformation domains. We also include a detailed case study on climate change disinformation, highlighting how linguistic, psychological, and cultural factors shape the adaptation of persuasion strategies to fit unique thematic contexts. 

**Abstract (ZH)**: 无论是哪个领域或使用哪种语言，虚假信息都旨在欺骗或操控公众舆论，通常通过运用先进的说服技术实现。有关说服技术在虚假信息中的武器化使用的研究，主要是针对特定主题（例如，新冠肺炎）进行的，跨领域的研究相对有限，导致对这些策略的理解不够全面。本研究采用最先进的说服技术分类器，对16种说服技术在虚假信息故事中的角色进行了大规模的多领域分析。研究表明，在不同类型的虚假信息领域中，不同说服技术的应用存在显著差异。此外，我们还对气候变化领域的虚假信息进行了详细的案例研究，揭示了语言、心理、文化等因素如何影响说服策略的适应性，以适应特定主题背景下的独特需求。 

---
# Till the Layers Collapse: Compressing a Deep Neural Network through the Lenses of Batch Normalization Layers 

**Title (ZH)**: 直到层失效：通过批归一化层的视角压缩深度神经网络 

**Authors**: Zhu Liao, Nour Hezbri, Victor Quétu, Van-Tam Nguyen, Enzo Tartaglione  

**Link**: [PDF](https://arxiv.org/pdf/2412.15077)  

**Abstract**: Today, deep neural networks are widely used since they can handle a variety of complex tasks. Their generality makes them very powerful tools in modern technology. However, deep neural networks are often overparameterized. The usage of these large models consumes a lot of computation resources. In this paper, we introduce a method called \textbf{T}ill the \textbf{L}ayers \textbf{C}ollapse (TLC), which compresses deep neural networks through the lenses of batch normalization layers. By reducing the depth of these networks, our method decreases deep neural networks' computational requirements and overall latency. We validate our method on popular models such as Swin-T, MobileNet-V2, and RoBERTa, across both image classification and natural language processing (NLP) tasks. 

**Abstract (ZH)**: 如今，由于深度神经网络能够处理各种复杂的任务，它们在现代技术中被广泛使用。它们的通用性使它们成为非常强大的工具。然而，深度神经网络通常会被过度参数化，这使得这些大模型消耗大量的计算资源。在本文中，我们提出了一种名为**Till the Layers Collapse (TLC)** 的方法，该方法通过批量归一化层的视角来压缩深度神经网络。通过减少这些网络的深度，我们的方法降低了深度神经网络的计算需求和整体延迟。我们在流行的模型如Swin-T、MobileNet-V2和RoBERTa上验证了该方法，涵盖了图像分类和自然语言处理（NLP）任务。 

---
# Large Language Models and Code Security: A Systematic Literature Review 

**Title (ZH)**: 大规模语言模型与代码安全：系统综述文献研究 

**Authors**: Enna Basic, Alberto Giaretta  

**Link**: [PDF](https://arxiv.org/pdf/2412.15004)  

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools for automating various programming tasks, including security-related ones, such as detecting and fixing vulnerabilities. Despite their promising capabilities, when required to produce or modify pre-existing code, LLMs could introduce vulnerabilities unbeknown to the programmer. When analyzing code, they could miss clear vulnerabilities or signal nonexistent ones. In this Systematic Literature Review (SLR), we aim to investigate both the security benefits and potential drawbacks of using LLMs for a variety of code-related tasks. In particular, first we focus on the types of vulnerabilities that could be introduced by LLMs, when used for producing code. Second, we analyze the capabilities of LLMs to detect and fix vulnerabilities, in any given code, and how the prompting strategy of choice impacts their performance in these two tasks. Last, we provide an in-depth analysis on how data poisoning attacks on LLMs can impact performance in the aforementioned tasks. 

**Abstract (ZH)**: 大规模语言模型（LLMs）已演化为自动化各种编程任务的强大工具，包括安全相关的任务，如检测和修复漏洞。尽管它们具有巨大的潜力，但在生成或修改现有代码时，LLMs仍然可能引入未知的漏洞。在分析代码时，它们可能会忽略明显的漏洞或误报不存在的漏洞。在这项系统文献综述（SLR）中，我们旨在探讨在各种代码相关任务中使用LLMs的安全益处和潜在缺点。首先，我们将重点研究当LLMs用于生成代码时可能引入的漏洞类型。其次，我们将分析LLMs在任何给定代码中检测和修复漏洞的能力，以及选定的提示策略如何影响其在这两项任务中的表现。最后，我们将深入分析数据注入攻击对LLMs性能的影响，特别是在上述任务中。 

---
# Movie2Story: A framework for understanding videos and telling stories in the form of novel text 

**Title (ZH)**: Movie2Story：一种理解视频并以新颖文本形式讲述故事的框架 

**Authors**: Kangning Li, Zheyang Jia, Anyu Ying  

**Link**: [PDF](https://arxiv.org/pdf/2412.14965)  

**Abstract**: Multimodal video-to-text models have made considerable progress, primarily in generating brief descriptions of video content. However, there is still a deficiency in generating rich long-form text descriptions that integrate both video and audio. In this paper, we introduce a framework called M2S, designed to generate novel-length text by combining audio, video, and character recognition. M2S includes modules for video long-form text description and comprehension, audio-based analysis of emotion, speech rate, and character alignment, and visual-based character recognition alignment. By integrating multimodal information using the large language model GPT4o, M2S stands out in the field of multimodal text generation. We demonstrate the effectiveness and accuracy of M2S through comparative experiments and human evaluation. Additionally, the model framework has good scalability and significant potential for future research. 

**Abstract (ZH)**: 多模态视频到文本模型在生成简短视频内容描述方面取得了显著进步。然而，在生成结合视频和音频的丰富长篇文本描述方面仍然存在不足。本文提出了一种名为M2S的框架，旨在通过结合音频、视频和字符识别来生成新颖长度的文本。M2S包括用于生成视频长篇文本描述和理解、基于音频的情感分析、语速分析以及基于视觉的字符识别对齐的模块。借助大型语言模型GPT4o整合多模态信息，M2S在多模态文本生成领域脱颖而出。通过对比实验和人工评估，我们展示了M2S的有效性和准确性。此外，该模型框架具有良好的可扩展性，并且具有巨大的未来研究潜力。 

---
# Unveiling Uncertainty: A Deep Dive into Calibration and Performance of Multimodal Large Language Models 

**Title (ZH)**: 揭开不确定性面纱：对多模态大型语言模型校准与性能的深入探讨 

**Authors**: Zijun Chen, Wenbo Hu, Guande He, Zhijie Deng, Zheng Zhang, Richang Hong  

**Link**: [PDF](https://arxiv.org/pdf/2412.14660)  

**Abstract**: Multimodal large language models (MLLMs) combine visual and textual data for tasks such as image captioning and visual question answering. Proper uncertainty calibration is crucial, yet challenging, for reliable use in areas like healthcare and autonomous driving. This paper investigates representative MLLMs, focusing on their calibration across various scenarios, including before and after visual fine-tuning, as well as before and after multimodal training of the base LLMs. We observed miscalibration in their performance, and at the same time, no significant differences in calibration across these scenarios. We also highlight how uncertainty differs between text and images and how their integration affects overall uncertainty. To better understand MLLMs' miscalibration and their ability to self-assess uncertainty, we construct the IDK (I don't know) dataset, which is key to evaluating how they handle unknowns. Our findings reveal that MLLMs tend to give answers rather than admit uncertainty, but this self-assessment improves with proper prompt adjustments. Finally, to calibrate MLLMs and enhance model reliability, we propose techniques such as temperature scaling and iterative prompt optimization. Our results provide insights into improving MLLMs for effective and responsible deployment in multimodal applications. Code and IDK dataset: \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）结合了视觉和文本数据，用于图像说明、视觉问答等任务。在诸如医疗保健和自主驾驶等关键领域中，适当的不确定性校准至关重要且颇具挑战性。本文研究了代表性MLLMs在不同场景下的校准情况，包括视觉微调前后以及基础大语言模型的多模态训练前后。我们观察到了性能的偏差校准问题，但在这些场景中的校准差异并不显著。此外，我们还探讨了文本与图像间不确定性的差异以及这种整合如何影响整体不确定性。为了更好地理解MLLMs的偏差校准及其自我评估不确定性的能力，我们构建了IDK（我不知道）数据集，这对于评估它们处理未知情况的能力至关重要。我们的研究发现，MLLMs倾向于给出答案而回避不确定性，但通过适当的提示调整，这种自我评估可以得到改善。最后，为了校准MLLMs并提高模型可靠性，我们提出了一些技术，如温度缩放和迭代提示优化。我们的研究结果为有效和负责任地部署多模态应用程序中的MLLMs提供了见解。相关代码和IDK数据集：\href{这个链接}{这个链接}。 

---
# LDP: Generalizing to Multilingual Visual Information Extraction by Language Decoupled Pretraining 

**Title (ZH)**: LDP：通过语言解耦预训练实现跨多语言视觉信息提取的泛化能力 

**Authors**: Huawen Shen, Gengluo Li, Jinwen Zhong, Yu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2412.14596)  

**Abstract**: Visual Information Extraction (VIE) plays a crucial role in the comprehension of semi-structured documents, and several pre-trained models have been developed to enhance performance. However, most of these works are monolingual (usually English). Due to the extremely unbalanced quantity and quality of pre-training corpora between English and other languages, few works can extend to non-English scenarios. In this paper, we conduct systematic experiments to show that vision and layout modality hold invariance among images with different languages. If decoupling language bias from document images, a vision-layout-based model can achieve impressive cross-lingual generalization. Accordingly, we present a simple but effective multilingual training paradigm LDP (Language Decoupled Pre-training) for better utilization of monolingual pre-training data. Our proposed model LDM (Language Decoupled Model) is first pre-trained on the language-independent data, where the language knowledge is decoupled by a diffusion model, and then the LDM is fine-tuned on the downstream languages. Extensive experiments show that the LDM outperformed all SOTA multilingual pre-trained models, and also maintains competitiveness on downstream monolingual/English benchmarks. 

**Abstract (ZH)**: 视觉信息提取（VIE）在半结构化文档的理解中扮演着至关重要的角色，已有多种预训练模型被开发出来以提高性能。然而，大多数这些工作都是单语言的（通常是英语）。由于英语和其他语言之间预训练语料库的数量和质量存在极其不平衡的情况，极少有工作能够扩展到非英语场景。在本文中，我们进行了系统的实验，以展示视觉和布局模态在不同语言图像之间的不变性。如果从文档图像中分离出语言偏见，基于视觉和布局的模型可以实现跨语言的出色泛化能力。因此，我们提出了一种简单的但有效的多语言训练范式，称为LDP（语言解耦预训练），以更好地利用单语言预训练数据。我们提出的模型LDM（语言解耦模型）首先在语言无关的数据上进行预训练，通过扩散模型解耦语言知识，然后在下游语言上进行微调。大量实验表明，LDM 在所有最先进的多语言预训练模型中表现最佳，并且在下游单语言/英语基准上也保持了竞争力。 

---
# Sliding Windows Are Not the End: Exploring Full Ranking with Long-Context Large Language Models 

**Title (ZH)**: 滑动窗口并非终点：探索长上下文大型语言模型的全面排名 

**Authors**: Wenhan Liu, Xinyu Ma, Yutao Zhu, Ziliang Zhao, Shuaiqiang Wang, Dawei Yin, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2412.14574)  

**Abstract**: Large Language Models (LLMs) have shown exciting performance in listwise passage ranking. Due to the limited input length, existing methods often adopt the sliding window strategy. Such a strategy, though effective, is inefficient as it involves repetitive and serialized processing, which usually re-evaluates relevant passages multiple times. As a result, it incurs redundant API costs, which are proportional to the number of inference tokens. The development of long-context LLMs enables the full ranking of all passages within a single inference, avoiding redundant API costs. In this paper, we conduct a comprehensive study of long-context LLMs for ranking tasks in terms of efficiency and effectiveness. Surprisingly, our experiments reveal that full ranking with long-context LLMs can deliver superior performance in the supervised fine-tuning setting with a huge efficiency improvement. Furthermore, we identify two limitations of fine-tuning the full ranking model based on existing methods: (1) sliding window strategy fails to produce a full ranking list as a training label, and (2) the language modeling loss cannot emphasize top-ranked passage IDs in the label. To alleviate these issues, we propose a new complete listwise label construction approach and a novel importance-aware learning objective for full ranking. Experiments show the superior performance of our method over baselines. Our codes are available at \url{this https URL}. 

**Abstract (ZH)**: 大型语言模型（LLMs）在列表级段落排序任务中展现了令人兴奋的性能。由于输入长度有限，现有方法常常采用滑动窗口策略。这种策略虽然有效，但由于涉及重复且串行的处理，通常需要多次重新评估相关段落，导致计算冗余，计算成本与推理令牌的数量成正比。随着具有长上下文能力的LLMs的发展，现在可以在单次推理中对所有段落进行全面排序，从而避免了冗余的API成本。在本文中，我们对具有长上下文的LLMs在排序任务中的效率和效果进行了全面研究。令人惊讶的是，我们的实验表明，在监督微调设置中，使用长上下文的LLMs进行全面排序可以显著提高性能，且具有巨大的效率改进。此外，我们还发现基于现有方法微调全排序模型的两个限制：（1）滑动窗口策略不能生成一个完整的排序列表作为训练标签，（2）语言模型损失无法强调标签中排名靠前的段落ID。为了缓解这些问题，我们提出了一种新的完整列表标签构建方法和一种新颖的重要性感知学习目标，以促进全排序。实验结果表明，我们的方法优于基线方法。我们已在<https://this-url> 公开了我们的代码。 

---
# Cal-DPO: Calibrated Direct Preference Optimization for Language Model Alignment 

**Title (ZH)**: Cal-DPO：校准的直接偏好优化方法用于语言模型对齐 

**Authors**: Teng Xiao, Yige Yuan, Huaisheng Zhu, Mingxiao Li, Vasant G Honavar  

**Link**: [PDF](https://arxiv.org/pdf/2412.14516)  

**Abstract**: We study the problem of aligning large language models (LLMs) with human preference data. Contrastive preference optimization has shown promising results in aligning LLMs with available preference data by optimizing the implicit reward associated with the policy. However, the contrastive objective focuses mainly on the relative values of implicit rewards associated with two responses while ignoring their actual values, resulting in suboptimal alignment with human preferences. To address this limitation, we propose calibrated direct preference optimization (Cal-DPO), a simple yet effective algorithm. We show that substantial improvement in alignment with the given preferences can be achieved simply by calibrating the implicit reward to ensure that the learned implicit rewards are comparable in scale to the ground-truth rewards. We demonstrate the theoretical advantages of Cal-DPO over existing approaches. The results of our experiments on a variety of standard benchmarks show that Cal-DPO remarkably improves off-the-shelf methods. 

**Abstract (ZH)**: 我们研究了将大规模语言模型（LLM）与人类偏好数据对齐的问题。对比偏好优化已经在通过优化与策略相关的隐含奖励来对齐LLM时显示出有前途的结果。然而，对比目标主要关注两个响应相关隐含奖励的相对值，而忽略了它们的实际值，从而导致与人类偏好对齐的次优结果。为了解决这一局限，我们提出了一种校准直接偏好优化（Cal-DPO）方法，该方法简单而有效。我们表明，通过校准隐含奖励以确保学习到的隐含奖励与真实奖励在规模上可比，可以仅此实现对给定偏好对齐的显著改善。我们证明了Cal-DPO相比现有方法具有理论优势。我们实验在多种标准基准上的结果表明，Cal-DPO显著改善了现成方法的性能。 

---
# GraphEQA: Using 3D Semantic Scene Graphs for Real-time Embodied Question Answering 

**Title (ZH)**: GraphEQA：使用3D语义场景图进行实时嵌入式问答 

**Authors**: Saumya Saxena, Blake Buchanan, Chris Paxton, Bingqing Chen, Narunas Vaskevicius, Luigi Palmieri, Jonathan Francis, Oliver Kroemer  

**Link**: [PDF](https://arxiv.org/pdf/2412.14480)  

**Abstract**: In Embodied Question Answering (EQA), agents must explore and develop a semantic understanding of an unseen environment in order to answer a situated question with confidence. This remains a challenging problem in robotics, due to the difficulties in obtaining useful semantic representations, updating these representations online, and leveraging prior world knowledge for efficient exploration and planning. Aiming to address these limitations, we propose GraphEQA, a novel approach that utilizes real-time 3D metric-semantic scene graphs (3DSGs) and task relevant images as multi-modal memory for grounding Vision-Language Models (VLMs) to perform EQA tasks in unseen environments. We employ a hierarchical planning approach that exploits the hierarchical nature of 3DSGs for structured planning and semantic-guided exploration. Through experiments in simulation on the HM-EQA dataset and in the real world in home and office environments, we demonstrate that our method outperforms key baselines by completing EQA tasks with higher success rates and fewer planning steps. 

**Abstract (ZH)**: 在具身问答（EQA）中，代理需要探索并培养对不可见环境的语义理解，以便自信地回答定位问题。由于获得有用语义表示、在线更新这些表示以及利用先验世界知识进行高效探索和规划的困难，这仍然是机器人领域的一个具有挑战性的问题。为了解决这些限制，我们提出了一种名为GraphEQA的新颖方法，该方法利用实时3D度量语义场景图（3DSGs）和与任务相关的图像作为多模态记忆，将视觉-语言模型（VLMs）接地以在不可见环境中执行EQA任务。我们采用了一种基于层次规划的方法，利用3DSGs的层次结构特性进行结构化规划和语义引导的探索。通过在HM-EQA数据集上进行模拟实验以及在家和办公室环境中的真实世界实验，我们展示了我们的方法通过更高的完成EQA任务的成功率和更少的规划步骤优于关键基准方法。 

---
# MegaPairs: Massive Data Synthesis For Universal Multimodal Retrieval 

**Title (ZH)**: MegaPairs：面向通用多模态检索的海量数据合成 

**Authors**: Junjie Zhou, Zheng Liu, Ze Liu, Shitao Xiao, Yueze Wang, Bo Zhao, Chen Jason Zhang, Defu Lian, Yongping Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2412.14475)  

**Abstract**: Despite the rapidly growing demand for multimodal retrieval, progress in this field remains severely constrained by a lack of training data. In this paper, we introduce MegaPairs, a novel data synthesis method that leverages vision language models (VLMs) and open-domain images, together with a massive synthetic dataset generated from this method. Our empirical analysis shows that MegaPairs generates high-quality data, enabling the multimodal retriever to significantly outperform the baseline model trained on 70$\times$ more data from existing datasets. Moreover, since MegaPairs solely relies on general image corpora and open-source VLMs, it can be easily scaled up, enabling continuous improvements in retrieval performance. In this stage, we produced more than 26 million training instances and trained several models of varying sizes using this data. These new models achieve state-of-the-art zero-shot performance across 4 popular composed image retrieval (CIR) benchmarks and the highest overall performance on the 36 datasets provided by MMEB. They also demonstrate notable performance improvements with additional downstream fine-tuning. Our produced dataset, well-trained models, and data synthesis pipeline will be made publicly available to facilitate the future development of this field. 

**Abstract (ZH)**: 尽管对多模态检索的需求迅速增长，但该领域的发展仍严重受限于训练数据的缺乏。本文介绍了一种名为MegaPairs的新型数据合成方法，该方法利用了视觉语言模型（VLMs）和开放域图像，以及由此方法生成的巨大合成数据集。我们的实证分析表明，MegaPairs生成了高质量的数据，使多模态检索模型显著优于现有数据集70倍的训练数据为基础训练的基本模型。此外，由于MegaPairs仅依赖于通用图像集合和开源VLMs，它可以很容易地扩展，从而不断改进检索性能。目前，我们已经生成了超过2600万训练实例，并使用此数据训练了多个不同规模的模型。这些新模型在4个流行的图像检索（CIR）基准测试中取得了最先进的零样本性能，并且在MMEB提供的36个数据集中取得了最高总体性能。此外，这些模型在额外的下游微调后也显示出显著的性能改进。我们生成的数据集、训练良好的模型以及数据合成管道将公开提供，以促进该领域未来的开发。 

---
# Are Longer Prompts Always Better? Prompt Selection in Large Language Models for Recommendation Systems 

**Title (ZH)**: 较长的提示是否Always更好？大型语言模型在推荐系统中的提示选择研究 

**Authors**: Genki Kusano, Kosuke Akimoto, Kunihiro Takeoka  

**Link**: [PDF](https://arxiv.org/pdf/2412.14454)  

**Abstract**: In large language models (LLM)-based recommendation systems (LLM-RSs), accurately predicting user preferences by leveraging the general knowledge of LLMs is possible without requiring extensive training data. By converting recommendation tasks into natural language inputs called prompts, LLM-RSs can efficiently solve issues that have been difficult to address due to data scarcity but are crucial in applications such as cold-start and cross-domain problems. However, when applying this in practice, selecting the prompt that matches tasks and data is essential. Although numerous prompts have been proposed in LLM-RSs and representing the target user in prompts significantly impacts recommendation accuracy, there are still no clear guidelines for selecting specific prompts.
In this paper, we categorize and analyze prompts from previous research to establish practical prompt selection guidelines. Through 450 experiments with 90 prompts and five real-world datasets, we examined the relationship between prompts and dataset characteristics in recommendation accuracy. We found that no single prompt consistently outperforms others; thus, selecting prompts on the basis of dataset characteristics is crucial. Here, we propose a prompt selection method that achieves higher accuracy with minimal validation data. Because increasing the number of prompts to explore raises costs, we also introduce a cost-efficient strategy using high-performance and cost-efficient LLMs, significantly reducing exploration costs while maintaining high prediction accuracy. Our work offers valuable insights into the prompt selection, advancing accurate and efficient LLM-RSs. 

**Abstract (ZH)**: 在基于大规模语言模型（LLM）的推荐系统（LLM-RSs）中，通过利用LLM的一般知识，可以在不依赖大量训练数据的情况下准确预测用户偏好。通过将推荐任务转化为称为提示的自然语言输入，LLM-RSs可以高效地解决由于数据稀缺而难以解决但在冷启动和跨域问题等应用中至关重要的问题。然而，在实际应用中，选择与任务和数据相匹配的提示至关重要。尽管在LLM-RS中提出了许多提示，并且提示中表示目标用户对推荐准确性的显著影响，但仍然没有明确的指导原则来选择具体的提示。

在本文中，我们对先前研究中的提示进行分类和分析，以建立实用的提示选择指南。通过在90个提示和五个真实世界数据集上的450次实验，我们研究了提示与数据集特征之间的关系对推荐准确性的影响。我们发现没有哪种提示能够始终优于其他提示，因此，根据数据集特征选择提示至关重要。在此基础上，我们提出了一种提示选择方法，该方法能够在最少的验证数据下实现更高的准确率。由于增加提示的数量以进行探索会增加成本，我们还提出了一个成本效益策略，使用高性能且成本效益高的LLM，显著降低了探索成本，同时保持了高预测准确率。我们的工作为提示选择提供了有价值的观点，推动了准确高效的LLM-RS的发展。 

---
# In-Group Love, Out-Group Hate: A Framework to Measure Affective Polarization via Contentious Online Discussions 

**Title (ZH)**: 拥护内群体，排斥外群体：一种通过争议性在线讨论衡量情感极化的框架 

**Authors**: Buddhika Nettasinghe, Ashwin Rao, Bohan Jiang, Allon Percus, Kristina Lerman  

**Link**: [PDF](https://arxiv.org/pdf/2412.14414)  

**Abstract**: Affective polarization, the emotional divide between ideological groups marked by in-group love and out-group hate, has intensified in the United States, driving contentious issues like masking and lockdowns during the COVID-19 pandemic. Despite its societal impact, existing models of opinion change fail to account for emotional dynamics nor offer methods to quantify affective polarization robustly and in real-time. In this paper, we introduce a discrete choice model that captures decision-making within affectively polarized social networks and propose a statistical inference method estimate key parameters -- in-group love and out-group hate -- from social media data. Through empirical validation from online discussions about the COVID-19 pandemic, we demonstrate that our approach accurately captures real-world polarization dynamics and explains the rapid emergence of a partisan gap in attitudes towards masking and lockdowns. This framework allows for tracking affective polarization across contentious issues has broad implications for fostering constructive online dialogues in digital spaces. 

**Abstract (ZH)**: 情感极化是指意识形态群体之间的情感分裂，表现为对群体内成员的爱和对群体外成员的恨。在美国，这一现象加剧了，特别是在新冠疫情期间，导致了诸如口罩佩戴和封锁等争议性问题的激烈争论。尽管情感极化对社会产生了深远影响，现有的意见变化模型并未考虑情感动态，也无法提供有效的方法来实时、稳健地量化情感极化。本文中，我们提出了一种离散选择模型，该模型能够捕捉情感极化社会网络内的决策过程，并提出了一种统计推断方法，以估计关键参数——群体内爱与群体外恨——从社交媒体数据中。通过对关于新冠疫情期间在线讨论的实证验证，我们证明了该方法能够准确捕捉现实世界的情感极化动态，并解释了党派意见在口罩佩戴和封锁态度上的快速分化。该框架允许跨争议性问题追踪情感极化，对促进数字空间中的建设性对话具有广泛的意义。 

---
# ResQ: Mixed-Precision Quantization of Large Language Models with Low-Rank Residuals 

**Title (ZH)**: ResQ：低秩残差辅助的大语言模型混合精度量化 

**Authors**: Utkarsh Saxena, Sayeh Sharify, Kaushik Roy, Xin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.14363)  

**Abstract**: Post-training quantization (PTQ) of large language models (LLMs) holds the promise in reducing the prohibitive computational cost at inference time. Quantization of all weight, activation and key-value (KV) cache tensors to 4-bit without significantly degrading generalizability is challenging, due to the high quantization error caused by extreme outliers in activations. To tackle this problem, we propose ResQ, a PTQ method that pushes further the state-of-the-art. By means of principal component analysis (PCA), it identifies a low-rank subspace (in practice 1/8 of the hidden dimension) in which activation variances are highest, and keep the coefficients within this subspace in high precision, e.g. 8-bit, while quantizing the rest to 4-bit. Within each subspace, invariant random rotation is applied to further suppress outliers. We show that this is a provably optimal mixed precision quantization scheme that minimizes error. With the Llama families of models, we demonstrate that ResQ outperforms recent uniform and mixed precision PTQ methods on a variety of benchmarks, achieving up to 33% lower perplexity on Wikitext than the next best method SpinQuant, and a 2.4x speedup over 16-bit baseline. Code is available at this https URL. 

**Abstract (ZH)**: 大语言模型（LLMs）的后训练量化（PTQ）有望在推理时减少巨大的计算成本。将所有权重、激活和关键值（KV）缓存张量量化到4位，同时不显著牺牲泛化能力，是一项具有挑战性的任务，这主要是由于激活中极端异常值引起的高量化误差。为解决这一问题，我们提出了ResQ，这是一种推进当前最先进的PTQ方法。通过主成分分析（PCA），ResQ识别出一个低秩子空间（实际上占隐藏维度的1/8），在这个子空间中激活方差最高，并保持该子空间内的系数以高精度表示，例如8位，而将其余部分量化为4位。在每个子空间内，应用不变随机旋转进一步抑制异常值。我们证明这是一种可证明的最优混合精度量化方案，能够最小化误差。在Llama模型家族中，我们证明ResQ在多种基准测试中优于最近的均匀精度和混合精度的PTQ方法，相较于最佳方法SpinQuant，在Wikitext上的困惑度降低了33%，并且比16位基线快2.4倍。代码可在以下链接找到：\[插入链接\] 

---
# Towards AI-$45^{\circ}$ Law: A Roadmap to Trustworthy AGI 

**Title (ZH)**: 朝着AI-45°定律：通向可信AGI的道路 

**Authors**: Yang Chao, Lu Chaochao, Wang Yingchun, Zhou Bowen  

**Link**: [PDF](https://arxiv.org/pdf/2412.14186)  

**Abstract**: Ensuring Artificial General Intelligence (AGI) reliably avoids harmful behaviors is a critical challenge, especially for systems with high autonomy or in safety-critical domains. Despite various safety assurance proposals and extreme risk warnings, comprehensive guidelines balancing AI safety and capability remain lacking. In this position paper, we propose the \textit{AI-\textbf{$45^{\circ}$} Law} as a guiding principle for a balanced roadmap toward trustworthy AGI, and introduce the \textit{Causal Ladder of Trustworthy AGI} as a practical framework. This framework provides a systematic taxonomy and hierarchical structure for current AI capability and safety research, inspired by Judea Pearl's ``Ladder of Causation''. The Causal Ladder comprises three core layers: the Approximate Alignment Layer, the Intervenable Layer, and the Reflectable Layer. These layers address the key challenges of safety and trustworthiness in AGI and contemporary AI systems. Building upon this framework, we define five levels of trustworthy AGI: perception, reasoning, decision-making, autonomy, and collaboration trustworthiness. These levels represent distinct yet progressive aspects of trustworthy AGI. Finally, we present a series of potential governance measures to support the development of trustworthy AGI.\footnote{In this paper, trustworthiness is generally considered a broad form of safety, and no explicit distinction is made between the two. However, in some contexts, safety and trustworthiness are treated as distinct: safety involves assurance of correct behavior, while trustworthiness refers to user confidence in the system's decision-making. In such cases, different terms or both may be used depending on the context. 

**Abstract (ZH)**: 确保通用人工智能（AGI）可靠地避免有害行为是一个关键挑战，特别是在具有高度自主权的系统或安全关键领域中。尽管有各种各样的安全保障提案和极端风险警告，但平衡AI安全与能力的全面指导原则仍然缺乏。在本文中，我们提出 \textit{AI-45° 原则} 作为走向可信赖AGI的平衡路线图的指导原则，并介绍了 \textit{可信赖AGI的因果梯阶} 作为实用框架。该框架提供了一个系统性分类和层级结构，对当前的AI能力和安全研究给予了启发，灵感来源于Judea Pearl的“因果梯阶”。因果梯阶包括三个核心层级：近似对齐层、干预层和反思层。这些层级解决了AGI和现代AI系统中安全与信任的关键挑战。在此框架的基础上，我们定义了可信赖AGI的五个层级：感知可信度、推理可信度、决策可信度、自主性和合作信任度。这些层级代表了可信AGI的不同但渐进的方面。最后，我们提出了一系列潜在的治理措施，以支持可信赖AGI的发展。\footnote{在本文中，可信赖性通常被认为是一种广泛的保障形式，对两者之间没有进行明确区分。但在某些情况下，安全保障和可信赖性被认为是不同的：安全保障涉及正确行为的保障，而可信赖性则涉及用户对系统决策的信心。在这种情况下，根据不同的情境，可以使用不同的术语或两者兼用。} 

---
# Whisper-GPT: A Hybrid Representation Audio Large Language Model 

**Title (ZH)**: whispers-GPT：一种混合表示音频大型语言模型 

**Authors**: Prateek Verma  

**Link**: [PDF](https://arxiv.org/pdf/2412.11449)  

**Abstract**: We propose WHISPER-GPT: A generative large language model (LLM) for speech and music that allows us to work with continuous audio representations and discrete tokens simultaneously as part of a single architecture. There has been a huge surge in generative audio, speech, and music models that utilize discrete audio tokens derived from neural compression algorithms, e.g. ENCODEC. However, one of the major drawbacks of this approach is handling the context length. It blows up for high-fidelity generative architecture if one has to account for all the audio contents at various frequencies for the next token prediction. By combining continuous audio representation like the spectrogram and discrete acoustic tokens, we retain the best of both worlds: Have all the information needed from the audio at a specific time instance in a single token, yet allow LLM to predict the future token to allow for sampling and other benefits discrete space provides. We show how our architecture improves the perplexity and negative log-likelihood scores for the next token prediction compared to a token-based LLM for speech and music. 

**Abstract (ZH)**: 我们提出WHISPER-GPT：一种既能处理连续声频表示又能同时处理离散声学标记的大规模语言模型（LLM），使其能够在一个单一架构中协同工作。在利用神经压缩算法（例如ENCODEC）衍生的离散声频标记的生成声频、语音和音乐模型方面，存在着巨大的发展。然而，这种方法的一个主要缺点是对上下文长度的处理。如果需要考虑到各种频率下的所有音频内容以预测下一个标记，则对于高保真生成架构而言，上下文长度会急剧增加。通过结合连续声频表示（如频谱图）和离散声学标记，我们保留了两者的优点：在单一标记中获得特定时间点所需的全部音频信息，同时允许LLM预测未来的标记，从而使离散空间提供的采样和其他好处得以实现。我们展示了与基于标记的语言模型相比，我们的架构在下一个标记预测方面如何改进语文困惑度和负对数似然度得分。 

---
# LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks 

**Title (ZH)**: LongBench v2：朝着对现实长上下文多任务有更深的理解和推理发展 

**Authors**: Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xiaozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu, Lei Hou, Yuxiao Dong, Jie Tang, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.15204)  

**Abstract**: This paper introduces LongBench v2, a benchmark designed to assess the ability of LLMs to handle long-context problems requiring deep understanding and reasoning across real-world multitasks. LongBench v2 consists of 503 challenging multiple-choice questions, with contexts ranging from 8k to 2M words, across six major task categories: single-document QA, multi-document QA, long in-context learning, long-dialogue history understanding, code repository understanding, and long structured data understanding. To ensure the breadth and the practicality, we collect data from nearly 100 highly educated individuals with diverse professional backgrounds. We employ both automated and manual review processes to maintain high quality and difficulty, resulting in human experts achieving only 53.7% accuracy under a 15-minute time constraint. Our evaluation reveals that the best-performing model, when directly answers the questions, achieves only 50.1% accuracy. In contrast, the o1-preview model, which includes longer reasoning, achieves 57.7%, surpassing the human baseline by 4%. These results highlight the importance of enhanced reasoning ability and scaling inference-time compute to tackle the long-context challenges in LongBench v2. The project is available at this https URL. 

**Abstract (ZH)**: 本文介绍了LongBench v2，这是一个基准测试，旨在评估大型语言模型（LLMs）处理需要深刻理解和推理的长文脉问题的能力，涵盖现实世界多任务。LongBench v2 包含503个具有挑战性的选择题，文脉长度从8千到2百万字不等，涵盖了六大主要任务类别：单文档问答、多文档问答、长文脉学习、长对话历史理解、代码仓库理解以及长结构化数据理解。为了确保涵盖范围和实用性，我们从近100名具有不同专业背景的高学历个人中收集了数据。我们采用了自动和人工审核两种流程，以保持高质量和难度，最终在15分钟的时间限制下，人类专家的准确率为53.7%。评估结果显示，最佳模型直接回答问题的准确率为50.1%。相比之下，包含更长推理过程的o1-preview模型的准确率为57.7%，超过了人类基线4%。这些结果突显了增强推理能力和扩展推理时计算能力对于应对LongBench v2中的长文脉挑战的重要性。该项目可在以下链接获取：[该项目链接]。 

---
# MMLU-CF: A Contamination-free Multi-task Language Understanding Benchmark 

**Title (ZH)**: MMLU-CF：一种无污染的多任务语言理解基准测试 

**Authors**: Qihao Zhao, Yangyu Huang, Tengchao Lv, Lei Cui, Qinzheng Sun, Shaoguang Mao, Xin Zhang, Ying Xin, Qiufeng Yin, Scarlett Li, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2412.15194)  

**Abstract**: Multiple-choice question (MCQ) datasets like Massive Multitask Language Understanding (MMLU) are widely used to evaluate the commonsense, understanding, and problem-solving abilities of large language models (LLMs). However, the open-source nature of these benchmarks and the broad sources of training data for LLMs have inevitably led to benchmark contamination, resulting in unreliable evaluation results. To alleviate this issue, we propose a contamination-free and more challenging MCQ benchmark called MMLU-CF. This benchmark reassesses LLMs' understanding of world knowledge by averting both unintentional and malicious data leakage. To avoid unintentional data leakage, we source data from a broader domain and design three decontamination rules. To prevent malicious data leakage, we divide the benchmark into validation and test sets with similar difficulty and subject distributions. The test set remains closed-source to ensure reliable results, while the validation set is publicly available to promote transparency and facilitate independent verification. Our evaluation of mainstream LLMs reveals that the powerful GPT-4o achieves merely a 5-shot score of 73.4% and a 0-shot score of 71.9% on the test set, which indicates the effectiveness of our approach in creating a more rigorous and contamination-free evaluation standard. The GitHub repository is available at this https URL and the dataset refers to this https URL. 

**Abstract (ZH)**: 下面是经过学术规范翻译的中文版本：

多选题数据集（MCQ datasets）如大规模多任务语言理解（MMLU）广泛用于评估大型语言模型（LLMs）的常识、理解和问题解决能力。然而，这些基准的开源性质以及LLMs训练数据的广泛来源不可避免地导致了基准污染，从而产生了不可靠的评估结果。为了解决这一问题，我们提出了一种无污染且更具挑战性的MCQ基准，称为MMLU-CF。该基准通过避免无意和恶意的数据泄露，重新评估LLMs对世界知识的理解能力。为避免无意的数据泄露，我们从更广泛的领域中获取数据，并设计了三条去污染规则。为了防止恶意数据泄露，我们将基准划分为具有相似难度和主题分布的验证集和测试集。测试集保持封闭源代码状态，以确保评估结果的可靠性，而验证集则可供公众访问，以促进透明度和独立验证。我们的评估结果显示，强大的GPT-4o在测试集上的5-shot得分为73.4%，0-shot得分为71.9%，这表明了我们方法在创建更严格和无污染的评估标准方面的有效性。该项目的GitHub仓库地址为：[此 https URL](此 https URL)，数据集的链接为：[此 https URL](此 https URL)。 

---
# Face the Facts! Evaluating RAG-based Fact-checking Pipelines in Realistic Settings 

**Title (ZH)**: 面对事实！在现实情境中评估基于RAG的事实核查流水线 

**Authors**: Daniel Russo, Stefano Menini, Jacopo Staiano, Marco Guerini  

**Link**: [PDF](https://arxiv.org/pdf/2412.15189)  

**Abstract**: Natural Language Processing and Generation systems have recently shown the potential to complement and streamline the costly and time-consuming job of professional fact-checkers. In this work, we lift several constraints of current state-of-the-art pipelines for automated fact-checking based on the Retrieval-Augmented Generation (RAG) paradigm. Our goal is to benchmark, under more realistic scenarios, RAG-based methods for the generation of verdicts - i.e., short texts discussing the veracity of a claim - evaluating them on stylistically complex claims and heterogeneous, yet reliable, knowledge bases. Our findings show a complex landscape, where, for example, LLM-based retrievers outperform other retrieval techniques, though they still struggle with heterogeneous knowledge bases; larger models excel in verdict faithfulness, while smaller models provide better context adherence, with human evaluations favouring zero-shot and one-shot approaches for informativeness, and fine-tuned models for emotional alignment. 

**Abstract (ZH)**: 自然语言处理与生成系统近年来显示出补充和简化专业事实核查者耗时且成本高昂工作的潜力。本工作中，我们放宽了基于检索增强生成（RAG）范式的当前最佳自动事实核查管道的若干限制。我们的目标是在更现实的情景下对基于RAG的方法进行基准测试，特别是在生成裁定——即讨论声明真实性的小段文本——方面，这些裁定评估的内容包括风格复杂的声明和多样但可靠的知识库。我们的研究发现了一个复杂的变化环境：例如，基于大型语言模型的检索器在性能上超过了其他检索技术，尽管它们在处理异构知识库时仍然面临挑战；更大的模型在裁定的忠实性方面表现优异，而较小的模型在上下文一致性方面表现更佳；人工评估显示，在信息量方面，零样本和单样本方法更受欢迎，而在情感对齐方面，则是微调后的模型表现更佳。 

---
# LlamaFusion: Adapting Pretrained Language Models for Multimodal Generation 

**Title (ZH)**: LlamaFusion：适应预训练语言模型的多模态生成 

**Authors**: Weijia Shi, Xiaochuang Han, Chunting Zhou, Weixin Liang, Xi Victoria Lin, Luke Zettlemoyer, Lili Yu  

**Link**: [PDF](https://arxiv.org/pdf/2412.15188)  

**Abstract**: We present LlamaFusion, a framework for empowering pretrained text-only large language models (LLMs) with multimodal generative capabilities, enabling them to understand and generate both text and images in arbitrary sequences. LlamaFusion leverages existing Llama-3's weights for processing texts autoregressively while introducing additional and parallel transformer modules for processing images with diffusion. During training, the data from each modality is routed to its dedicated modules: modality-specific feedforward layers, query-key-value projections, and normalization layers process each modality independently, while the shared self-attention layers allow interactions across text and image features. By freezing the text-specific modules and only training the image-specific modules, LlamaFusion preserves the language capabilities of text-only LLMs while developing strong visual understanding and generation abilities. Compared to methods that pretrain multimodal generative models from scratch, our experiments demonstrate that, LlamaFusion improves image understanding by 20% and image generation by 3.6% using only 50% of the FLOPs while maintaining Llama-3's language capabilities. We also demonstrate that this framework can adapt existing vision-language models with multimodal generation ability. Overall, this framework not only leverages existing computational investments in text-only LLMs but also enables the parallel development of language and vision capabilities, presenting a promising direction for efficient multimodal model development. 

**Abstract (ZH)**: 我们提出了LlamaFusion，这是一种框架，旨在为预训练的文本型大型语言模型（LLMs）赋予多模态生成能力，使其能够理解和生成任意序列的文字与图像。LlamaFusion 利用 Llama-3 的现有权重进行文本自回归处理，并引入并行的变压器模块以通过扩散处理图像。在训练过程中，来自每种模态的数据都被导向其专用模块：特定模态的前馈层、查询-键-值投影和规范化层独立处理每种模态，而共享的自注意力层则允许跨文本和图像特征的交互。通过冻结特定于文本的模块，仅训练特定于图像的模块，LlamaFusion 保留了纯文本型LLMs 的语言能力，同时发展出了强大的视觉理解和生成能力。与从零开始预训练多模态生成模型的方法相比，我们的实验结果表明，LlamaFusion 仅使用Llama-3计算量的50%，就能提高图像理解20% 和图像生成3.6%，同时保持Llama-3的语言能力。我们还展示了该框架能够适应现有的具备多模态生成能力的视觉语言模型。总体而言，该框架不仅利用了纯文本型LLMs 的现有计算投资，还促进了语言和视觉能力的并行发展，为高效多模态模型开发提供了有希望的方向。 

---
# Language Models as Continuous Self-Evolving Data Engineers 

**Title (ZH)**: 语言模型作为连续自进化的数据工程师 

**Authors**: Peidong Wang, Ming Wang, Zhiming Ma, Xiaocui Yang, Shi Feng, Daling Wang, Yifei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.15151)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities on various tasks, while the further evolvement is limited to the lack of high-quality training data. In addition, traditional training approaches rely too much on expert-labeled data, setting an upper limit on the performance of LLMs. To address this issue, we propose a novel paradigm that enables LLMs to train itself by autonomously generating, cleaning, reviewing, and annotating data with preference information, named LANCE. Our approach demonstrates that LLMs can serve as continuous self-evolving data engineers, significantly reducing the time and cost of the post-training data construction process. Through iterative fine-tuning on different variants of the Qwen2, we validate the effectiveness of LANCE across various tasks, showing that it can continuously improve model performance and maintain high-quality data generation. Across eight benchmark dimensions, LANCE resulted in an average score enhancement of 3.36 for Qwen2-7B and 2.70 for Qwen2-7B-Instruct. This training paradigm with autonomous data construction not only reduces the reliance on human experts or external models but also ensures that the data aligns with human values and preferences, paving the way for the development of future superintelligent systems that can exceed human capabilities. 

**Abstract (ZH)**: 大规模语言模型（Large Language Models, LLMs）在各种任务上展现出了卓越的能力，但进一步的发展受限于高质量训练数据的缺乏。此外，传统的训练方法过于依赖专家标注的数据，从而限制了LLMs的性能。为了解决这一问题，我们提出了一种新的范式，使LLMs能够自主生成、清理、审查和注释数据，并根据偏好信息进行训练，我们将其命名为LANCE。我们的方法表明，LLMs能够作为持续自我演化的数据工程师，显著减少了训练后数据构建过程的时间和成本。通过在不同版本的Qwen2上进行迭代微调，我们验证了LANCE在多种任务上的有效性，展示了它可以持续提升模型性能并保持高质量的数据生成。LANCE在八个基准维度上，分别使Qwen2-7B和Qwen2-7B-Instruct的平均评分提高了3.36和2.70。这种具有自主数据构建的训练范式不仅减少了对人类专家或外部模型的依赖，还确保数据与人类的价值和偏好相一致，为未来可以超越人类能力的超级智能系统的发展铺平了道路。 

---
# Adaptive Pruning for Large Language Models with Structural Importance Awareness 

**Title (ZH)**: 带有结构重要性awareness的大型语言模型自适应剪枝 

**Authors**: Haotian Zheng, Jinke Ren, Yushan Sun, Ruichen Zhang, Wenbo Zhang, Zhen Li, Dusit Niyato, Shuguang Cui, Yatong Han  

**Link**: [PDF](https://arxiv.org/pdf/2412.15127)  

**Abstract**: The recent advancements in large language models (LLMs) have significantly improved language understanding and generation capabilities. However, it is difficult to deploy LLMs on resource-constrained edge devices due to their high computational and storage resource demands. To address this issue, we propose a novel LLM model pruning method, namely structurally-aware adaptive pruning (SAAP), to significantly reduce the computational and memory costs while maintaining model performance. We first define an adaptive importance fusion metric to evaluate the importance of all coupled structures in LLMs by considering their homoscedastic uncertainty. Then, we rank the importance of all modules to determine the specific layers that should be pruned to meet particular performance requirements. Furthermore, we develop a new group fine-tuning strategy to improve the inference efficiency of LLMs. Finally, we evaluate the proposed SAAP method on multiple LLMs across two common tasks, i.e., zero-shot classification and text generation. Experimental results show that our SAAP method outperforms several state-of-the-art baseline methods, achieving 2.17%, 2.37%, and 2.39% accuracy gains on LLaMA-7B, Vicuna-7B, and LLaMA-13B. Additionally, SAAP improves the token generation speed by 5%, showcasing its practical advantages in resource-constrained scenarios. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）在语言理解和生成能力方面取得了显著进步。然而，由于其对计算和存储资源的高度需求，部署到资源受限的边缘设备上仍然极具挑战性。为了解决这一问题，我们提出了一种新颖的LLM模型剪枝方法，即结构感知自适应剪枝（SAAP），能够在大幅降低计算和内存成本的同时保持模型性能。首先，我们定义了一种自适应重要性融合度量，通过考虑耦合结构的同方差不确定性来评估LLMs中所有耦合结构的重要性。然后，我们对所有模块的重要性进行排序，以确定应修剪的具体层以满足特定的性能要求。此外，我们开发了一种新的组微调策略，以提高LLMs的推理效率。最后，我们在两个常见的任务即零样本分类和文本生成上，对多种LLM进行了SAAP方法的评估。实验结果表明，我们的SAAP方法在LLaMA-7B、Vicuna-7B和LLaMA-13B上分别实现了2.17%、2.37%和2.39%的准确率提升，同时SAAP方法提高了5%的令牌生成速度，展示了其在资源受限场景下的实用优势。 

---
# Outcome-Refining Process Supervision for Code Generation 

**Title (ZH)**: 代码生成中的目标优化过程监督 

**Authors**: Zhuohao Yu, Weizheng Gu, Yidong Wang, Zhengran Zeng, Jindong Wang, Wei Ye, Shikun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.15118)  

**Abstract**: Large Language Models have demonstrated remarkable capabilities in code generation, yet they often struggle with complex programming tasks that require deep algorithmic reasoning. While process supervision through learned reward models shows promise in guiding reasoning steps, it requires expensive training data and suffers from unreliable evaluation. We propose Outcome-Refining Process Supervision, a novel paradigm that treats outcome refinement itself as the process to be supervised. Our framework leverages concrete execution signals to ground the supervision of reasoning steps, while using tree-structured exploration to maintain multiple solution trajectories simultaneously. Experiments demonstrate that our approach enables even smaller models to achieve high success accuracy and performance metrics on competitive programming tasks, creates more reliable verification than traditional reward models without requiring training PRMs. Our approach achieves significant improvements across 5 models and 3 datasets: an average of 26.9% increase in correctness and 42.2% in efficiency. The results suggest that providing structured reasoning space with concrete verification signals is crucial for solving complex programming tasks. We open-source all our code and data at: this https URL 

**Abstract (ZH)**: 大型语言模型在代码生成方面展现了出色的性能，但在需要深入算法推理的复杂编程任务中却常常表现不佳。虽然通过学习奖励模型进行过程监督显示出一定的潜力，但这种方法需要昂贵的训练数据，并且在评估过程中存在不稳定性。我们提出了一种新颖的方法——结果精炼过程监督（Outcome-Refining Process Supervision），将其看作要监督的过程本身。我们的框架利用具体的执行信号将监督过程与推理步骤紧密结合，同时采用树状结构的探索机制以同时维护多个解决方案路径。实验结果表明，我们的方法能够使更小的模型在竞争编程任务上实现高成功率和高性能指标，而无需训练特定奖励模型就能提供更可靠的验证。我们的方法在5个模型和3个数据集上实现了显著改进：平均正确率提高了26.9%，效率提高了42.2%。结果表明，为解决复杂编程任务提供结构化的推理空间和具体的验证信号至关重要。我们已开源了所有代码和数据：请访问以下链接：[这里提供链接] 

---
# Qwen2.5 Technical Report 

**Title (ZH)**: Qwen2.5技术报告 

**Authors**: Qwen, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zihan Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2412.15115)  

**Abstract**: In this report, we introduce Qwen2.5, a comprehensive series of large language models (LLMs) designed to meet diverse needs. Compared to previous iterations, Qwen 2.5 has been significantly improved during both the pre-training and post-training stages. In terms of pre-training, we have scaled the high-quality pre-training datasets from the previous 7 trillion tokens to 18 trillion tokens. This provides a strong foundation for common sense, expert knowledge, and reasoning capabilities. In terms of post-training, we implement intricate supervised finetuning with over 1 million samples, as well as multistage reinforcement learning. Post-training techniques enhance human preference, and notably improve long text generation, structural data analysis, and instruction following. To handle diverse and varied use cases effectively, we present Qwen2.5 LLM series in rich sizes. Open-weight offerings include base and instruction-tuned models, with quantized versions available. In addition, for hosted solutions, the proprietary models currently include two mixture-of-experts (MoE) variants: Qwen2.5-Turbo and Qwen2.5-Plus, both available from Alibaba Cloud Model Studio. Qwen2.5 has demonstrated top-tier performance on a wide range of benchmarks evaluating language understanding, reasoning, mathematics, coding, human preference alignment, etc. Specifically, the open-weight flagship Qwen2.5-72B-Instruct outperforms a number of open and proprietary models and demonstrates competitive performance to the state-of-the-art open-weight model, Llama-3-405B-Instruct, which is around 5 times larger. Qwen2.5-Turbo and Qwen2.5-Plus offer superior cost-effectiveness while performing competitively against GPT-4o-mini and GPT-4o respectively. Additionally, as the foundation, Qwen2.5 models have been instrumental in training specialized models such as Qwen2.5-Math, Qwen2.5-Coder, QwQ, and multimodal models. 

**Abstract (ZH)**: 以下是经过学术规范翻译后的中文内容：

本报告介绍了Qwen 2.5，这是一个全面的大型语言模型（LLM）系列，旨在满足多样化的应用需求。与之前的版本相比，Qwen 2.5 在预训练和后训练阶段均得到了显著改进。在预训练阶段，我们将高质量的预训练数据集规模从之前的 7 万亿个令牌扩展至 18 万亿个令牌，这为常识、专家知识和推理能力奠定了坚实的基础。在后训练阶段，我们采用超过 100 万个样本的复杂监督微调，并实施多阶段强化学习。后训练技术提高了对人类偏好的模拟，并显著改进了长文生成、结构化数据分析和指令遵循等方面的能力。为了有效处理多样化和复杂的应用场景，我们提供了多档规模的 Qwen 2.5 LLM 系列。开源版本包括基础模型和指令调优模型，并且提供了量化版本。此外，对于托管解决方案，我们的专有模型包括两个专家混合模型（MoE）变体：Qwen 2.5-Turbo 和 Qwen 2.5-Plus，均可通过阿里云模型工作室获得。Qwen 2.5 在一系列评测基准测试中表现出顶级性能，涵盖语言理解、推理、数学、编程、人类偏好对齐等多个领域。特别是开源版本的旗舰模型 Qwen 2.5-72B-Instruct，在多项评测基准测试中表现出色，优于多种开源和专有模型，并且其性能与开源重量级模型 Llama-3-405B-Instruct 相当，而后者规模大约是 Qwen 2.5-72B-Instruct 的 5 倍。Qwen 2.5-Turbo 和 Qwen 2.5-Plus 在各自的对比中展现出较高的成本效益，同时在性能上与 GPT-4o-mini 和 GPT-4o 分别竞争。此外，Qwen 2.5 模型作为基础模型在训练专门化模型如 Qwen 2.5-Math、Qwen 2.5-Coder、QwQ 以及多模态模型方面发挥了关键作用。 

---
# Review-Then-Refine: A Dynamic Framework for Multi-Hop Question Answering with Temporal Adaptability 

**Title (ZH)**: Review-Then-Refine：一种具有时间适应性的多跳问答动态框架 

**Authors**: Xiangsen Chen, Xuming Hu, Nan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2412.15101)  

**Abstract**: Retrieve-augmented generation (RAG) frameworks have emerged as a promising solution to multi-hop question answering(QA) tasks since it enables large language models (LLMs) to incorporate external knowledge and mitigate their inherent knowledge deficiencies. Despite this progress, existing RAG frameworks, which usually follows the retrieve-then-read paradigm, often struggle with multi-hop QA with temporal information since it has difficulty retrieving and synthesizing accurate time-related information. To address the challenge, this paper proposes a novel framework called review-then-refine, which aims to enhance LLM performance in multi-hop QA scenarios with temporal information. Our approach begins with a review phase, where decomposed sub-queries are dynamically rewritten with temporal information, allowing for subsequent adaptive retrieval and reasoning process. In addition, we implement adaptive retrieval mechanism to minimize unnecessary retrievals, thus reducing the potential for hallucinations. In the subsequent refine phase, the LLM synthesizes the retrieved information from each sub-query along with its internal knowledge to formulate a coherent answer. Extensive experimental results across multiple datasets demonstrate the effectiveness of our proposed framework, highlighting its potential to significantly improve multi-hop QA capabilities in LLMs. 

**Abstract (ZH)**: 检索增强生成（RAG）框架自问世以来被视为解决多跳问答（QA）任务的有前途的解决方案，因为它使大型语言模型（LLMs）能够融入外部知识并缓解其固有的知识缺陷。尽管取得了一定进展，但现有的RAG框架通常遵循“先检索后阅读”的模式，在涉及时间信息的多跳问答中往往表现不佳，因为它难以检索和整合准确的时间相关信息。为应对这一挑战，本文提出了一种名为“先审查后优化”的新颖框架，旨在增强LLMs在包含时间信息的多跳问答场景中的性能。我们的方法首先进入一个审查阶段，在此阶段，分解后的子查询会动态地加入时间信息，使得随后的适应性检索与推理过程更具针对性。此外，我们实施了适应性检索机制，以减少不必要的检索次数，从而降低幻觉的可能性。在随后的优化阶段，LLMs会整合从每个子查询中检索到的信息以及其内部知识，以形成一个连贯的答案。在多个数据集上的广泛实验结果表明，我们提出的框架具有显著效果，突显了其在提升LLMs的多跳问答能力方面的潜力。 

---
# AceMath: Advancing Frontier Math Reasoning with Post-Training and Reward Modeling 

**Title (ZH)**: AceMath：通过后训练和奖励建模推动数学推理前沿发展 

**Authors**: Zihan Liu, Yang Chen, Mohammad Shoeybi, Bryan Catanzaro, Wei Ping  

**Link**: [PDF](https://arxiv.org/pdf/2412.15084)  

**Abstract**: In this paper, we introduce AceMath, a suite of frontier math models that excel in solving complex math problems, along with highly effective reward models capable of evaluating generated solutions and reliably identifying the correct ones. To develop the instruction-tuned math models, we propose a supervised fine-tuning (SFT) process that first achieves competitive performance across general domains, followed by targeted fine-tuning for the math domain using a carefully curated set of prompts and synthetically generated responses. The resulting model, AceMath-72B-Instruct greatly outperforms Qwen2.5-Math-72B-Instruct, GPT-4o and Claude-3.5 Sonnet. To develop math-specialized reward model, we first construct AceMath-RewardBench, a comprehensive and robust benchmark for evaluating math reward models across diverse problems and difficulty levels. After that, we present a systematic approach to build our math reward models. The resulting model, AceMath-72B-RM, consistently outperforms state-of-the-art reward models. Furthermore, when combining AceMath-72B-Instruct with AceMath-72B-RM, we achieve the highest average rm@8 score across the math reasoning benchmarks. We will release model weights, training data, and evaluation benchmarks at: this https URL 

**Abstract (ZH)**: 在本文中，我们介绍了AceMath，这是一个卓越于解决复杂数学问题的前沿数学模型系列，同时配备了一套高效的目标奖励模型，能够评估生成的解决方案并可靠地识别正确的答案。为了开发这些指令调优的数学模型，我们提出了一种监督微调（SFT）过程，该过程首先在通用领域内实现了竞争力的表现，然后通过精心准备的提示集和合成生成的响应，在数学领域进行目标化的微调。最终生成的模型AceMath-72B-Instruct在性能上大幅超过了Qwen2.5-Math-72B-Instruct、GPT-4o和Claude-3.5 Sonnet。为了开发专用于数学的奖励模型，我们首先构建了AceMath-RewardBench，这是一个全面且稳健的基准，用于评估不同种类和难度级别的数学奖励模型。之后，我们提出了一个系统的方法来构建我们的数学奖励模型。生成的模型AceMath-72B-RM在多个基准上持续超过最先进的奖励模型。此外，当我们结合AceMath-72B-Instruct与AceMath-72B-RM时，我们在数学推理基准上的平均rm@8得分达到了最高。我们将在以下链接发布模型权重、训练数据和评估基准：[this https URL] 

---
# ConfliBERT: A Language Model for Political Conflict 

**Title (ZH)**: ConfliBERT：一种用于政治冲突的语言模型 

**Authors**: Patrick T. Brandt, Sultan Alsarra, Vito J. D`Orazio, Dagmar Heintze, Latifur Khan, Shreyas Meher, Javier Osorio, Marcus Sianan  

**Link**: [PDF](https://arxiv.org/pdf/2412.15060)  

**Abstract**: Conflict scholars have used rule-based approaches to extract information about political violence from news reports and texts. Recent Natural Language Processing developments move beyond rigid rule-based approaches. We review our recent ConfliBERT language model (Hu et al. 2022) to process political and violence related texts. The model can be used to extract actor and action classifications from texts about political conflict. When fine-tuned, results show that ConfliBERT has superior performance in accuracy, precision and recall over other large language models (LLM) like Google's Gemma 2 (9B), Meta's Llama 3.1 (7B), and Alibaba's Qwen 2.5 (14B) within its relevant domains. It is also hundreds of times faster than these more generalist LLMs. These results are illustrated using texts from the BBC, re3d, and the Global Terrorism Dataset (GTD). 

**Abstract (ZH)**: 冲突研究学者使用基于规则的方法从新闻报道和文本中提取政治暴力相关信息。最近的自然语言处理发展超越了僵化的基于规则的方法。我们回顾了我们最近提出的ConfliBERT语言模型（Hu et al. 2022），以处理与政治和暴力相关的文本。该模型可以用于从关于政治冲突的文本中提取行为者和行为分类。当进行微调时，结果显示，在相关领域内，ConfliBERT在准确度、精确度和召回率方面优于其他大型语言模型（LLM），如Google的Gemma 2（9B）、Meta的Llama 3.1（7B）和阿里巴巴的Qwen 2.5（14B）。而且，它比这些更通用的LLM快数百倍。这些结果通过使用BBC、re3d和全球恐怖主义数据库（GTD）中的文本进行了说明。 

---
# LLMs Lost in Translation: M-ALERT uncovers Cross-Linguistic Safety Gaps 

**Title (ZH)**: LLMs 在跨语言翻译中迷失：M-ALERT 暴露了跨语言安全差距

解释：
- LLMs (Large Language Models) 超大语言模型
- Lost in Translation 跨语言翻译中迷失
- M-ALERT M-ALERT（假设这是一个特定的研究或检测工具的名称）
- uncovers 暴露了
- Cross-Linguistic 定义域跨语言
- Safety Gaps 安全差距

这个翻译符合学术论文标题和内容翻译的要求，简洁且准确地传达了原文的意思。 

**Authors**: Felix Friedrich, Simone Tedeschi, Patrick Schramowski, Manuel Brack, Roberto Navigli, Huu Nguyen, Bo Li, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2412.15035)  

**Abstract**: Building safe Large Language Models (LLMs) across multiple languages is essential in ensuring both safe access and linguistic diversity. To this end, we introduce M-ALERT, a multilingual benchmark that evaluates the safety of LLMs in five languages: English, French, German, Italian, and Spanish. M-ALERT includes 15k high-quality prompts per language, totaling 75k, following the detailed ALERT taxonomy. Our extensive experiments on 10 state-of-the-art LLMs highlight the importance of language-specific safety analysis, revealing that models often exhibit significant inconsistencies in safety across languages and categories. For instance, Llama3.2 shows high unsafety in the category crime_tax for Italian but remains safe in other languages. Similar differences can be observed across all models. In contrast, certain categories, such as substance_cannabis and crime_propaganda, consistently trigger unsafe responses across models and languages. These findings underscore the need for robust multilingual safety practices in LLMs to ensure safe and responsible usage across diverse user communities. 

**Abstract (ZH)**: 跨多种语言构建安全的大语言模型（LLMs）对于确保安全访问和语言多样性至关重要。为此，我们引入了M-ALERT，这是一个多语言基准，评估了五种语言（英语、法语、德语、意大利语和西班牙语）中LLMs的安全性。M-ALERT 每种语言包含 15,000 个高质量的提示，总计 75,000 个，遵循详细的 ALERT 分类法。我们在 10 个最新的LLMs上的广泛实验强调了语言特定安全分析的重要性，揭示出模型在不同语言和类别中的安全性存在显著差异。例如，Llama3.2 在意大利语中的犯罪税类别中表现出高水平的不安全性，但在其他语言中则保持安全。类似的现象在所有模型中普遍存在。相比之下，某些类别，如物质大麻和犯罪宣传，会在所有模型和语言中触发不安全的响应。这些发现强调了在LLMs中实施稳健的多语言安全性实践的重要性，以确保跨多样用户社区的安全和负责任使用。 

---
# Chain-of-MetaWriting: Linguistic and Textual Analysis of How Small Language Models Write Young Students Texts 

**Title (ZH)**: 链式元写作：小语言模型为年轻学生撰写文本的语言与文本分析 

**Authors**: Ioana Buhnila, Georgeta Cislaru, Amalia Todirascu  

**Link**: [PDF](https://arxiv.org/pdf/2412.14986)  

**Abstract**: Large Language Models (LLMs) have been used to generate texts in response to different writing tasks: reports, essays, story telling. However, language models do not have a meta-representation of the text writing process, nor inherent communication learning needs, comparable to those of young human students. This paper introduces a fine-grained linguistic and textual analysis of multilingual Small Language Models' (SLMs) writing. With our method, Chain-of-MetaWriting, SLMs can imitate some steps of the human writing process, such as planning and evaluation. We mainly focused on short story and essay writing tasks in French for schoolchildren and undergraduate students respectively. Our results show that SLMs encounter difficulties in assisting young students on sensitive topics such as violence in the schoolyard, and they sometimes use words too complex for the target audience. In particular, the output is quite different from the human produced texts in term of text cohesion and coherence regarding temporal connectors, topic progression, reference. 

**Abstract (ZH)**: 大语言模型（LLMs）被用于完成各种写作任务，如生成报告、文章和故事。然而，语言模型缺乏对文本写作过程的元表征，也没有与年轻人类学生相提并论的固有沟通学习需求。本文通过对多语言小型语言模型（SLMs）写作进行细粒度的语言和文本分析，介绍了Chain-of-MetaWriting方法，使SLMs能够模仿人类写作过程中的某些步骤，如计划和评估。我们主要针对学校儿童和本科生分别进行短篇故事和文章的写作任务进行了研究。结果显示，SLMs在协助年轻学生处理敏感话题（如校园暴力）方面遇到困难，并且有时使用过于复杂的词汇，不适合目标受众。尤其是在文本连贯性和一致性、时间连接词、话题进展和参照方面，模型生成的内容与人类生成的文本存在显著差异。 

---
# Knowledge Injection via Prompt Distillation 

**Title (ZH)**: 通过提示蒸馏实现知识注入 

**Authors**: Kalle Kujanpää, Harri Valpola, Alexander Ilin  

**Link**: [PDF](https://arxiv.org/pdf/2412.14964)  

**Abstract**: In many practical applications, large language models (LLMs) need to incorporate new knowledge not present in their pre-training data. The primary methods for this are fine-tuning and retrieval-augmented generation (RAG). Although RAG has emerged as the industry standard for knowledge injection, fine-tuning has not yet achieved comparable success. In this paper, we propose a new fine-tuning technique for learning new knowledge and show that it can reach the performance of RAG. The proposed method is based on the self-distillation approach, which we call prompt distillation. First, we generate question-answer pairs about the new knowledge. Then, we fine-tune a student model on the question-answer pairs to imitate the output distributions of a teacher model, which additionally receives the new knowledge in its prompt. The student model is identical to the teacher, except it is equipped with a LoRA adapter. This training procedure facilitates distilling the new knowledge from the teacher's prompt into the student's weights. 

**Abstract (ZH)**: 在许多实际应用中，大型语言模型（LLMs）需要整合其预训练数据中不存在的新知识。目前主要的两种方法是微调和检索增强生成（RAG）。尽管RAG已成为知识注入的行业标准，但微调尚未取得同等的成功。本文我们提出了一种新的微调技术，用于学习新知识，并证明这种技术可以达到RAG的性能。所提出的方法基于自蒸馏方法，我们称之为提示蒸馏。首先，我们生成关于新知识的问题-答案对。然后，我们使用问题-答案对对一个学生模型进行微调，使其模仿在提示中接收了新知识的教师模型的输出分布。学生模型与教师模型几乎相同，只是额外配备了LoRA适配器。这种训练过程有助于将教师模型提示中的新知识蒸馏到学生模型的权重中。 

---
# Understanding the Dark Side of LLMs' Intrinsic Self-Correction 

**Title (ZH)**: 理解大语言模型内在自校正的暗面 

**Authors**: Qingjie Zhang, Han Qiu, Di Wang, Haoting Qian, Yiming Li, Tianwei Zhang, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2412.14959)  

**Abstract**: Intrinsic self-correction was proposed to improve LLMs' responses via feedback prompts solely based on their inherent capability. However, recent works show that LLMs' intrinsic self-correction fails without oracle labels as feedback prompts. In this paper, we aim to interpret LLMs' intrinsic self-correction for different tasks, especially for those failure cases. By including one simple task and three complex tasks with state-of-the-art (SOTA) LLMs like ChatGPT families (o1, 4o, 3.5-turbo) and Llama families (2-7B, 3-8B, and 3.1-8B), we design three interpretation methods to reveal the dark side of LLMs' intrinsic self-correction. We identify intrinsic self-correction can (1) cause LLMs to waver both intermedia and final answers and lead to prompt bias on simple factual questions; (2) introduce human-like cognitive bias on complex tasks. In light of our findings, we also provide two simple yet effective strategies for alleviation: question repeating and supervised fine-tuning with a few samples. We open-source our work at this https URL. 

**Abstract (ZH)**: 内在自我修正被提出用以通过反馈提示来提升大语言模型（LLMs）的响应质量，这种方式仅依赖于模型本身的能力。然而，近期的研究表明，缺少先验标签作为反馈提示时，LLMs的内在自我修正无法发挥作用。在本文中，我们旨在解释不同任务中LLMs的内在自我修正，尤其是针对其失败案例。通过包括一个简单任务和三个复杂任务，并使用最先进的LLM如ChatGPT家族（o1、4o、3.5-turbo）和Llama家族（2-7B、3-8B、3.1-8B），我们设计了三种解释方法以揭示LLMs内在自我修正的暗面。我们发现内在自我修正能够（1）导致LLMs在中间答案和最终答案上摇摆不定，并对简单的事实性问题产生提示偏差；（2）在复杂任务中引入类似人类的认知偏差。基于我们的发现，我们还提出了两种简单且有效的缓解策略：重复问题和少量样本的监督微调。我们的工作已开源，您可以在此处访问：[提供链接的URL]。 

---
# RobustFT: Robust Supervised Fine-tuning for Large Language Models under Noisy Response 

**Title (ZH)**: RobustFT：在噪音响应环境下大型语言模型的鲁棒监督微调 

**Authors**: Junyu Luo, Xiao Luo, Kaize Ding, Jingyang Yuan, Zhiping Xiao, Ming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.14922)  

**Abstract**: Supervised fine-tuning (SFT) plays a crucial role in adapting large language models (LLMs) to specific domains or tasks. However, as demonstrated by empirical experiments, the collected data inevitably contains noise in practical applications, which poses significant challenges to model performance on downstream tasks. Therefore, there is an urgent need for a noise-robust SFT framework to enhance model capabilities in downstream tasks. To address this challenge, we introduce a robust SFT framework (RobustFT) that performs noise detection and relabeling on downstream task data. For noise identification, our approach employs a multi-expert collaborative system with inference-enhanced models to achieve superior noise detection. In the denoising phase, we utilize a context-enhanced strategy, which incorporates the most relevant and confident knowledge followed by careful assessment to generate reliable annotations. Additionally, we introduce an effective data selection mechanism based on response entropy, ensuring only high-quality samples are retained for fine-tuning. Extensive experiments conducted on multiple LLMs across five datasets demonstrate RobustFT's exceptional performance in noisy scenarios. 

**Abstract (ZH)**: 监督微调（SFT）在使大规模语言模型（LLMs）适应特定领域或任务方面起着重要作用。然而，实证实验表明，在实际应用中收集的数据不可避免地包含噪音，这对模型在下游任务中的性能构成了重大挑战。因此，迫切需要一种鲁棒的SFT框架，以增强模型在下游任务中的能力。为应对这一挑战，我们引入了一种鲁棒SFT框架（RobustFT），该框架可以在下游任务数据上执行噪音检测和重新标记。在噪音识别方面，我们的方法采用多专家协作系统和推断增强模型，以实现更优的噪音检测。在去噪阶段，我们使用一种上下文增强策略，该策略结合了最相关的和最自信的知识，并经过仔细评估以生成可靠的注释。此外，我们还引入了一种基于响应熵的数据选择机制，确保只有高质量的样本才被保留用于微调。在五个数据集上对多个LLMs进行的广泛实验表明，RobustFT在嘈杂场景中的性能表现出色。 

---
# Dehallucinating Parallel Context Extension for Retrieval-Augmented Generation 

**Title (ZH)**: 去幻化并行上下文扩展以实现检索增强生成 

**Authors**: Zexiong Ma, Shengnan An, Zeqi Lin, Yanzhen Zou, Jian-Guang Lou, Bing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2412.14905)  

**Abstract**: Large language models (LLMs) are susceptible to generating hallucinated information, despite the integration of retrieval-augmented generation (RAG). Parallel context extension (PCE) is a line of research attempting to effectively integrating parallel (unordered) contexts, while it still suffers from hallucinations when adapted to RAG scenarios. In this paper, we propose DePaC (Dehallucinating Parallel Context Extension), which alleviates the hallucination problem with context-aware negative training and information-calibrated aggregation. DePaC is designed to alleviate two types of in-context hallucination: fact fabrication (i.e., LLMs present claims that are not supported by the contexts) and fact omission (i.e., LLMs fail to present claims that can be supported by the contexts). Specifically, (1) for fact fabrication, we apply the context-aware negative training that fine-tunes the LLMs with negative supervisions, thus explicitly guiding the LLMs to refuse to answer when contexts are not related to questions; (2) for fact omission, we propose the information-calibrated aggregation which prioritizes context windows with higher information increment from their contexts. The experimental results on nine RAG tasks demonstrate that DePaC significantly alleviates the two types of hallucination and consistently achieves better performances on these tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）在整合检索增强生成（RAG）后仍容易生成虚构信息。平行上下文扩展（PCE）是一条旨在有效结合无序平行上下文的研究路线，但在适应RAG情景时仍然存在虚构信息的问题。本文提出了一种名为DePaC（Dehallucinating Parallel Context Extension）的方法，通过基于上下文的负样本训练和信息校准聚合来缓解虚构信息问题。DePaC 设计用于缓解两种类型的任务内虚构信息：事实捏造（即LLMs提出的声明并未得到上下文的支持）和事实遗漏（即LLMs未能呈现可由上下文支持的声明）。具体而言，（1）对于事实捏造，我们应用基于上下文的负样本训练，通过微调LLMs以负监督进行训练，从而明确指导LLMs在上下文与问题无关时不要作答；（2）对于事实遗漏，我们提出了信息校准聚合方法，优先考虑那些从其上下文中获得更高信息增量的上下文窗口。在九个RAG任务上的实验结果表明，DePaC 显著缓解了这两种类型的虚构信息，并在这些任务上实现了更优的性能。 

---
# Why language models collapse when trained on recursively generated text 

**Title (ZH)**: 当训练数据包含递归生成的文本时，语言模型为何会出现性能下降现象 

**Authors**: Lecheng Wang, Xianjie Shi, Ge Li, Jia Li, Yihong Dong, Xuanming Zhang, Wenpin Jiao, Hong Mei  

**Link**: [PDF](https://arxiv.org/pdf/2412.14872)  

**Abstract**: Language models (LMs) have been widely used to generate text on the Internet. The generated text is often collected into the training corpus of the next generations of LMs. Previous work has experimentally found that LMs collapse when trained on recursively generated text. This paper contributes to existing knowledge from two aspects. We present a theoretical proof of LM collapse. Our proof reveals the cause of LM collapse and proves that all auto-regressive LMs will definitely collapse. We present a new finding: the performance of LMs gradually declines when trained on recursively generated text until they perform no better than a randomly initialized LM. The trained LMs produce large amounts of repetitive text and perform poorly across a wide range of natural language tasks. The above proof and new findings deepen our understanding of LM collapse and offer valuable insights that may inspire new training techniques to mitigate this threat. 

**Abstract (ZH)**: 语言模型（LMs）已在互联网上广泛用于生成文本。生成的文本往往被收集进下一代LMs的训练语料库中。先前的工作已通过实验发现，当LMs在递归生成的文本上训练时会发生崩溃现象。本文在两个方面为现有知识做出了贡献。我们提供了一种理论证明来解释LMs发生崩溃的现象。我们的证明揭示了LMs崩溃的原因，并证明所有自回归LMs最终都会崩溃。我们提出了一项新的发现：当LMs在递归生成的文本上训练时，其性能逐渐下降，直到它们的表现还不如随机初始化的LMs。训练后的LMs会产生大量重复的文本，并在广泛的自然语言任务上表现不佳。上述证明和新发现深化了我们对LM崩溃现象的理解，并提供了有助于启发新的训练技术来减轻这一威胁的宝贵见解。 

---
# Graph-Convolutional Networks: Named Entity Recognition and Large Language Model Embedding in Document Clustering 

**Title (ZH)**: 图卷积网络：文档聚类中的命名实体识别与大规模语言模型嵌入 

**Authors**: Imed Keraghel, Mohamed Nadif  

**Link**: [PDF](https://arxiv.org/pdf/2412.14867)  

**Abstract**: Recent advances in machine learning, particularly Large Language Models (LLMs) such as BERT and GPT, provide rich contextual embeddings that improve text representation. However, current document clustering approaches often ignore the deeper relationships between named entities (NEs) and the potential of LLM embeddings. This paper proposes a novel approach that integrates Named Entity Recognition (NER) and LLM embeddings within a graph-based framework for document clustering. The method builds a graph with nodes representing documents and edges weighted by named entity similarity, optimized using a graph-convolutional network (GCN). This ensures a more effective grouping of semantically related documents. Experimental results indicate that our approach outperforms conventional co-occurrence-based methods in clustering, notably for documents rich in named entities. 

**Abstract (ZH)**: 近年来，机器学习领域的最新进展，尤其是大规模语言模型（LLMs）如BERT和GPT，提供了丰富的上下文嵌入，从而改善了文本表示。然而，当前的文档聚类方法往往忽视了命名实体（NEs）之间的深层关系及其LLM嵌入的潜力。本文提出了一种新颖的方法，该方法在基于图的框架内结合命名实体识别（NER）和LLM嵌入进行文档聚类。该方法构建了一个图，图中的节点表示文档，边的权重基于命名实体的相似性，并使用图卷积网络（GCN）进行优化。这确保了更具效的语义相关文档的分组。实验结果表明，与基于共现的传统方法相比，我们的方法在聚类命名实体丰富的文档方面表现更优。 

---
# Think&Cite: Improving Attributed Text Generation with Self-Guided Tree Search and Progress Reward Modeling 

**Title (ZH)**: 思与引：利用自我引导树搜索和进步奖励建模提升带属性文本生成 

**Authors**: Junyi Li, Hwee Tou Ng  

**Link**: [PDF](https://arxiv.org/pdf/2412.14860)  

**Abstract**: Despite their outstanding capabilities, large language models (LLMs) are prone to hallucination and producing factually incorrect information. This challenge has spurred efforts in attributed text generation, which prompts LLMs to generate content with supporting evidence. In this paper, we propose a novel framework, called Think&Cite, and formulate attributed text generation as a multi-step reasoning problem integrated with search. Specifically, we propose Self-Guided Monte Carlo Tree Search (SG-MCTS), which capitalizes on the self-reflection capability of LLMs to reflect on the intermediate states of MCTS for guiding the tree expansion process. To provide reliable and comprehensive feedback, we introduce Progress Reward Models to measure the progress of tree search from the root to the current state from two aspects, i.e., generation and attribution progress. We conduct extensive experiments on three datasets and the results show that our approach significantly outperforms baseline approaches. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）具有卓越的能力，但它们容易产生幻觉，并生成事实性错误的信息。这一挑战促进了属性化文本生成的努力，即促使LLMs生成带有支持证据的内容。在本文中，我们提出了一种新型框架——Think&Cite，并将属性化文本生成问题形式化为一个包含搜索过程的多步推理问题。具体而言，我们提出了Self-Guided Monte Carlo Tree Search（SG-MCTS），该方法利用LLMs的自我反思能力，通过对MCTS中间状态的反思来指导树的扩展过程。为了提供可靠且全面的反馈，我们引入了Progress Reward Models，以从生成进展和归属进展两个方面衡量从根节点到当前状态的树搜索过程的进展。我们在三个数据集上进行了广泛的实验，结果表明，我们的方法显著优于基准方法。 

---
# DS$^2$-ABSA: Dual-Stream Data Synthesis with Label Refinement for Few-Shot Aspect-Based Sentiment Analysis 

**Title (ZH)**: DS$^2$-ABSA：基于双重流数据合成与标签细化的少样本方面基于情感分析 

**Authors**: Hongling Xu, Yice Zhang, Qianlong Wang, Ruifeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2412.14849)  

**Abstract**: Recently developed large language models (LLMs) have presented promising new avenues to address data scarcity in low-resource scenarios. In few-shot aspect-based sentiment analysis (ABSA), previous efforts have explored data augmentation techniques, which prompt LLMs to generate new samples by modifying existing ones. However, these methods fail to produce adequately diverse data, impairing their effectiveness. Besides, some studies apply in-context learning for ABSA by using specific instructions and a few selected examples as prompts. Though promising, LLMs often yield labels that deviate from task requirements. To overcome these limitations, we propose DS$^2$-ABSA, a dual-stream data synthesis framework targeted for few-shot ABSA. It leverages LLMs to synthesize data from two complementary perspectives: \textit{key-point-driven} and \textit{instance-driven}, which effectively generate diverse and high-quality ABSA samples in low-resource settings. Furthermore, a \textit{label refinement} module is integrated to improve the synthetic labels. Extensive experiments demonstrate that DS$^2$-ABSA significantly outperforms previous few-shot ABSA solutions and other LLM-oriented data generation methods. 

**Abstract (ZH)**: 近年来开发的大型语言模型（LLMs）为低资源场景下的数据稀缺问题提供了 promising 的新途径。在少样本方面情感分析（ABSA）中，先前的努力探索了数据扩充技术，通过修改现有样本来生成新的样本。然而，这些方法未能产生足够多样化的数据，从而影响其效果。此外，一些研究通过使用特定指令和少数选定的示例作为提示，利用上下文学习进行ABSA。尽管有希望，但LLMs往往会产生与任务要求不符的标签。为了克服这些限制，我们提出了一种针对少样本ABSA的双流数据合成框架DS$^2$-ABSA。该框架利用LLMs从两个互补的角度合成数据：关键点驱动和实例驱动，从而在低资源环境下有效生成多样化且高质量的ABSA样本。此外，还集成了一个标签精炼模块以改进合成标签。广泛实验表明，DS$^2$-ABSA显著优于先前的少样本ABSA解决方案及其他以LLMs为导向的数据生成方法。 

---
# A Survey of RWKV 

**Title (ZH)**: 《RWKV综述》 

**Authors**: Zhiyuan Li, Tingyu Xia, Yi Chang, Yuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2412.14847)  

**Abstract**: The Receptance Weighted Key Value (RWKV) model offers a novel alternative to the Transformer architecture, merging the benefits of recurrent and attention-based systems. Unlike conventional Transformers, which depend heavily on self-attention, RWKV adeptly captures long-range dependencies with minimal computational demands. By utilizing a recurrent framework, RWKV addresses some computational inefficiencies found in Transformers, particularly in tasks with long sequences. RWKV has recently drawn considerable attention for its robust performance across multiple domains. Despite its growing popularity, no systematic review of the RWKV model exists. This paper seeks to fill this gap as the first comprehensive review of the RWKV architecture, its core principles, and its varied applications, such as natural language generation, natural language understanding, and computer vision. We assess how RWKV compares to traditional Transformer models, highlighting its capability to manage long sequences efficiently and lower computational costs. Furthermore, we explore the challenges RWKV encounters and propose potential directions for future research and advancement. We consistently maintain the related open-source materials at: this https URL. 

**Abstract (ZH)**: 《接收器加权关键值（RWKV）模型》提供了一种新颖的Transformer架构替代方案，它结合了循环系统和基于注意力系统的优点。与依赖大量自我注意的常规Transformer不同，RWKV能够以最小的计算需求高效地捕捉长期依赖关系。通过使用循环框架，RWKV解决了在处理长序列任务时常规Transformer中存在的部分计算效率问题。由于其在多个领域的稳健表现，RWKV最近引起了相当大的关注。尽管RWKV的受欢迎程度正在增长，但目前尚不存在系统的RWKV模型综述。本文旨在填补这一空白，作为首篇全面综述RWKV架构、核心原则及其广泛应用（如自然语言生成、自然语言理解和计算机视觉）的文章。我们评估了RWKV与传统Transformer模型的比较，突出了其高效管理长序列和降低计算成本的能力。此外，我们探讨了RWKV遇到的挑战，并提出了未来研究和发展的潜在方向。我们将始终维护相关开源材料于以下链接：https://github.com/alibaba/Qwen。 

---
# Mapping and Influencing the Political Ideology of Large Language Models using Synthetic Personas 

**Title (ZH)**: 使用合成人物映射和影响大型语言模型的政治意识形态 

**Authors**: Pietro Bernardelle, Leon Fröhling, Stefano Civelli, Riccardo Lunardi, Kevin Roiter, Gianluca Demartini  

**Link**: [PDF](https://arxiv.org/pdf/2412.14843)  

**Abstract**: The analysis of political biases in large language models (LLMs) has primarily examined these systems as single entities with fixed viewpoints. While various methods exist for measuring such biases, the impact of persona-based prompting on LLMs' political orientation remains unexplored. In this work we leverage PersonaHub, a collection of synthetic persona descriptions, to map the political distribution of persona-based prompted LLMs using the Political Compass Test (PCT). We then examine whether these initial compass distributions can be manipulated through explicit ideological prompting towards diametrically opposed political orientations: right-authoritarian and left-libertarian. Our experiments reveal that synthetic personas predominantly cluster in the left-libertarian quadrant, with models demonstrating varying degrees of responsiveness when prompted with explicit ideological descriptors. While all models demonstrate significant shifts towards right-authoritarian positions, they exhibit more limited shifts towards left-libertarian positions, suggesting an asymmetric response to ideological manipulation that may reflect inherent biases in model training. 

**Abstract (ZH)**: 对大型语言模型（LLMs）中的政治偏见分析主要将其作为单一且固定观点的实体进行研究。尽管已存在多种测量这些偏见的方法，但基于人设提示对LLMs政治倾向的影响尚未得到探索。在本研究中，我们利用PersonaHub集合中的人设描述，通过政治罗盘测试（PCT）来绘制基于人设提示的LLMs的政治分布情况。随后，我们研究这些初始罗盘分布是否可以通过明确的意识形态提示被操纵至完全相反的政治倾向：右极权主义和左自由主义。我们的实验结果显示，合成人设主要集中在左自由主义象限，且在使用明确的意识形态描述符提示后，这些模型在不同程度上有所响应。虽然所有的模型都显示出显著地向右极权主义倾向转移，但它们在向左自由主义方向转移时表现出更为有限的变化，这表明意识形态操纵的不对称响应可能反映了模型训练中固有的偏见。 

---
# DynamicKV: Task-Aware Adaptive KV Cache Compression for Long Context LLMs 

**Title (ZH)**: DynamicKV：面向任务的长上下文LLM自适应键值缓存压缩 

**Authors**: Xiabin Zhou, Wenbin Wang, Minyan Zeng, Jiaxian Guo, Xuebo Liu, Li Shen, Min Zhang, Liang Ding  

**Link**: [PDF](https://arxiv.org/pdf/2412.14838)  

**Abstract**: Efficient KV cache management in LLMs is crucial for long-context tasks like RAG and summarization. Existing KV cache compression methods enforce a fixed pattern, neglecting task-specific characteristics and reducing the retention of essential information. However, we observe distinct activation patterns across layers in various tasks, highlighting the need for adaptive strategies tailored to each task's unique demands. Based on this insight, we propose DynamicKV, a method that dynamically optimizes token retention by adjusting the number of tokens retained at each layer to adapt to the specific task. DynamicKV establishes global and per-layer maximum KV cache budgets, temporarily retaining the maximum budget for the current layer, and periodically updating the KV cache sizes of all preceding layers during inference. Our method retains only 1.7% of the KV cache size while achieving ~85% of the Full KV cache performance on LongBench. Notably, even under extreme compression (0.9%), DynamicKV surpasses state-of-the-art (SOTA) methods by 11% in the Needle-in-a-Haystack test using Mistral-7B-Instruct-v0.2. The code will be released. 

**Abstract (ZH)**: 在LLM中高效管理键值（KV）缓存对于长上下文任务（如RAG和总结）至关重要。现有的KV缓存压缩方法施加了固定模式，忽视了任务特定的特性，从而减少了关键信息的保留。然而，我们观察到各种任务在不同层之间存在不同的激活模式，突出了需要针对每个任务的独特需求制定适应性策略的必要性。基于这一洞见，我们提出了DynamicKV方法，该方法动态优化token保留策略，通过调整每层保留的token数量以适应特定任务。DynamicKV 设立了全局和每层的KV缓存预算，并在当前层保留最大预算，在推理过程中定期更新所有前向层的KV缓存大小。我们的方法在LongBench上的性能中仅保留了1.7%的KV缓存大小，同时实现了约85%的全KV缓存性能。值得注意的是，在极端压缩（90%压缩率）下，DynamicKV在使用Mistral-7B-Instruct-v0.2进行Needle-in-a-Haystack测试时，比最先进的（SOTA）方法性能高出11%。代码将公开发布。 

---
# Progressive Multimodal Reasoning via Active Retrieval 

**Title (ZH)**: 基于主动检索的渐进多模态推理 

**Authors**: Guanting Dong, Chenghao Zhang, Mengjie Deng, Yutao Zhu, Zhicheng Dou, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2412.14835)  

**Abstract**: Multi-step multimodal reasoning tasks pose significant challenges for multimodal large language models (MLLMs), and finding effective ways to enhance their performance in such scenarios remains an unresolved issue. In this paper, we propose AR-MCTS, a universal framework designed to progressively improve the reasoning capabilities of MLLMs through Active Retrieval (AR) and Monte Carlo Tree Search (MCTS). Our approach begins with the development of a unified retrieval module that retrieves key supporting insights for solving complex reasoning problems from a hybrid-modal retrieval corpus. To bridge the gap in automated multimodal reasoning verification, we employ the MCTS algorithm combined with an active retrieval mechanism, which enables the automatic generation of step-wise annotations. This strategy dynamically retrieves key insights for each reasoning step, moving beyond traditional beam search sampling to improve the diversity and reliability of the reasoning space. Additionally, we introduce a process reward model that aligns progressively to support the automatic verification of multimodal reasoning tasks. Experimental results across three complex multimodal reasoning benchmarks confirm the effectiveness of the AR-MCTS framework in enhancing the performance of various multimodal models. Further analysis demonstrates that AR-MCTS can optimize sampling diversity and accuracy, yielding reliable multimodal reasoning. 

**Abstract (ZH)**: 多步多模态推理任务对多模态大型语言模型（MLLMs）构成了重大挑战，如何在这些场景中有效提升其性能仍然是一个未解决的问题。本文提出了一种名为AR-MCTS的通用框架，旨在通过主动检索（AR）和蒙特卡洛树搜索（MCTS）逐步增强MLLMs的推理能力。我们的方法首先开发了一个统一的检索模块，可以从混合模态检索语料库中检索解决复杂推理问题所需的关键支持见解。为弥合自动多模态推理验证的差距，我们结合了主动检索机制和MCTS算法，从而能够自动生成逐步注释。该策略动态检索每个推理步骤的关键见解，超越了传统的束搜索采样方法，以提高推理空间的多样性和可靠性。此外，我们引入了一个过程奖励模型，以渐进的方式支持多模态推理任务的自动验证。在三个复杂多模态推理基准上的实验结果证实了AR-MCTS框架在增强各种多模态模型性能方面的有效性。进一步的分析表明，AR-MCTS可以优化采样的多样性和准确性，从而实现可靠的多模态推理。 

---
# Mention Attention for Pronoun Translation 

**Title (ZH)**: 提及关注机制在代词翻译中的应用 

**Authors**: Gongbo Tang, Christian Hardmeier  

**Link**: [PDF](https://arxiv.org/pdf/2412.14829)  

**Abstract**: Most pronouns are referring expressions, computers need to resolve what do the pronouns refer to, and there are divergences on pronoun usage across languages. Thus, dealing with these divergences and translating pronouns is a challenge in machine translation. Mentions are referring candidates of pronouns and have closer relations with pronouns compared to general tokens. We assume that extracting additional mention features can help pronoun translation. Therefore, we introduce an additional mention attention module in the decoder to pay extra attention to source mentions but not non-mention tokens. Our mention attention module not only extracts features from source mentions, but also considers target-side context which benefits pronoun translation. In addition, we also introduce two mention classifiers to train models to recognize mentions, whose outputs guide the mention attention. We conduct experiments on the WMT17 English-German translation task, and evaluate our models on general translation and pronoun translation, using BLEU, APT, and contrastive evaluation metrics. Our proposed model outperforms the baseline Transformer model in terms of APT and BLEU scores, this confirms our hypothesis that we can improve pronoun translation by paying additional attention to source mentions, and shows that our introduced additional modules do not have negative effect on the general translation quality. 

**Abstract (ZH)**: 大多数代词都是指示代词，计算机需要确定这些代词所指的对象是什么。不同语言在代词使用上存在差异，因此处理这些差异并进行代词翻译是一个挑战。提及（mentions）是代词可能指代的候选对象，相对于一般的词元，提及与代词关系更为密切。我们假设提取额外的提及特征可以有助于代词翻译。因此，我们在解码器中引入了一个额外的提及注意模块，以额外关注源侧提及而不是非提及词元。我们的提及注意模块不仅从源侧提及中提取特征，还考虑目标侧上下文，这有利于代词翻译。此外，我们还引入了两个提及分类器来训练模型以识别提及，其输出指导提及注意机制。我们在WMT17英德翻译任务上进行了实验，并使用BLEU、APT和对比性评估指标评估我们的模型，针对一般翻译和代词翻译进行了评估。我们的提出模型在APT和BLEU评分上优于基线的Transformer模型，这证实了我们的假设，通过额外关注源侧提及可以改进代词翻译，同时表明我们引入的附加模块并未对一般翻译质量产生负面影响。 

---
# ResoFilter: Rine-grained Synthetic Data Filtering for Large Language Models through Data-Parameter Resonance Analysis 

**Title (ZH)**: ResoFilter：通过数据-参数共振分析的大规模语言模型细粒度合成数据过滤方法 

**Authors**: Zeao Tu, Xiangdi Meng, Yu He, Zihan Yao, Tianyu Qi, Jun Liu, Ming Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.14809)  

**Abstract**: Large language models (LLMs) have shown remarkable effectiveness across various domains, with data augmentation methods utilizing GPT for synthetic data generation becoming prevalent. However, the quality and utility of augmented data remain questionable, and current methods lack clear metrics for evaluating data characteristics. To address these challenges, we propose ResoFilter, a novel method that integrates models, data, and tasks to refine datasets. ResoFilter leverages the fine-tuning process to obtain Data-Parameter features for data selection, offering improved interpretability by representing data characteristics through model weights. Our experiments demonstrate that ResoFilter achieves comparable results to full-scale fine-tuning using only half the data in mathematical tasks and exhibits strong generalization across different models and domains. This method provides valuable insights for constructing synthetic datasets and evaluating high-quality data, offering a promising solution for enhancing data augmentation techniques and improving training dataset quality for LLMs. For reproducibility, we will release our code and data upon acceptance. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各个领域展现出了显著的效果，利用GPT等方法进行数据增强以生成合成数据的方法变得越来越普遍。然而，增强数据的质量和实用价值仍存在疑问，且当前方法缺乏明确的数据特性评估标准。为应对这些挑战，我们提出了一种名为ResoFilter的新方法，该方法结合了模型、数据和任务，以精炼数据集。ResoFilter利用微调过程来获得数据特征参数，并通过模型权重表示这些特征，提高了数据特性的可解释性。我们的实验表明，与全量微调相比，使用ResoFilter仅需一半的数据即可在数学任务中取得相似结果，并且在不同模型和领域中表现出强大的泛化能力。该方法为构建合成数据集以及评估高质量数据提供了有价值的见解，为增强数据增强技术并提高LLMs训练数据集质量提供了有前景的解决方案。为了便于复现，我们在论文被接受后将发布我们的代码和数据。 

---
# Disentangling Reasoning Tokens and Boilerplate Tokens For Language Model Fine-tuning 

**Title (ZH)**: 拆分推理令牌和模板令牌以进行语言模型微调 

**Authors**: Ziang Ye, Zhenru Zhang, Yang Zhang, Jianxin Ma, Junyang Lin, Fuli Feng  

**Link**: [PDF](https://arxiv.org/pdf/2412.14780)  

**Abstract**: When using agent-task datasets to enhance agent capabilities for Large Language Models (LLMs), current methodologies often treat all tokens within a sample equally. However, we argue that tokens serving different roles - specifically, reasoning tokens versus boilerplate tokens (e.g., those governing output format) - differ significantly in importance and learning complexity, necessitating their disentanglement and distinct treatment. To address this, we propose a novel Shuffle-Aware Discriminator (SHAD) for adaptive token discrimination. SHAD classifies tokens by exploiting predictability differences observed after shuffling input-output combinations across samples: boilerplate tokens, due to their repetitive nature among samples, maintain predictability, whereas reasoning tokens do not. Using SHAD, we propose the Reasoning-highlighted Fine-Tuning (RFT) method, which adaptively emphasizes reasoning tokens during fine-tuning, yielding notable performance gains over common Supervised Fine-Tuning (SFT). 

**Abstract (ZH)**: 在使用代理任务数据集来增强大型语言模型（LLMs）的能力时，当前的方法通常将样本中的所有标记等同对待。然而，我们认为不同作用的标记之间存在显著差异，特别是用于推理的标记与用于规范输出格式的模板标记之间存在差异，前者相比后者在重要性和学习复杂性上有所不同，因此需要将它们分离并分别对待。为解决这一问题，我们提出了一种新的抖动感知判别器（SHAD，Shuffle-Aware Discriminator）以实现自适应的标记区分。SHAD通过利用跨样本输入-输出组合打乱后观察到的可预测性差异来对标记进行分类：由于模板标记在不同样本中具有重复性，因此它们保持可预测性；而用于推理的标记则不具有这种特性。利用SHAD，我们提出了推理突出的微调（RFT，Reasoning-highlighted Fine-Tuning）方法，该方法在微调过程中自适应地强调推理标记，与常见的监督微调（SFT，Supervised Fine-Tuning）相比，能够获得显著的性能提升。 

---
# ALKAFI-LLAMA3: Fine-Tuning LLMs for Precise Legal Understanding in Palestine 

**Title (ZH)**: ALKAFI-LLAMA3：为巴勒斯坦精准法律理解 Fine-Tuning 大型语言模型 

**Authors**: Rabee Qasem, Mohannad Hendi, Banan Tantour  

**Link**: [PDF](https://arxiv.org/pdf/2412.14771)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable potential in diverse domains, yet their application in the legal sector, particularly in low-resource contexts, remains limited. This study addresses the challenges of adapting LLMs to the Palestinian legal domain, where political instability, fragmented legal frameworks, and limited AI resources hinder effective machine-learning applications. We present a fine-tuned model based on a quantized version of Llama-3.2-1B-Instruct, trained on a synthetic data set derived from Palestinian legal texts. Using smaller-scale models and strategically generated question-answer pairs, we achieve a cost-effective, locally sustainable solution that provides accurate and contextually relevant legal guidance. Our experiments demonstrate promising performance on various query types, ranging from yes/no questions and narrative explanations to complex legal differentiations, while highlighting areas for improvement, such as handling calculation-based inquiries and structured list formatting. This work provides a pathway for the deployment of AI-driven legal assistance tools tailored to the needs of resource-constrained environments. 

**Abstract (ZH)**: 大语言模型（LLMs）在多个领域展现了显著的应用潜力，但在法律领域的应用仍受限，尤其是在资源稀缺的环境中。本研究致力于克服将LLMs适应到巴勒斯坦法律领域的挑战，其中政治不稳定、法律框架碎片化以及有限的人工智能资源阻碍了有效的机器学习应用。我们基于量化版本的Llama-3.2-1B-Instruct模型进行微调，并针对巴勒斯坦法律文本构建了合成数据集。通过使用较小规模的模型和有针对性地生成的问题-答案对，我们实现了成本效益高、本地可持续的解决方案，能够提供准确且具有上下文相关性的法律指导。实验结果表明，在不同类型查询（从是/否问题和叙述解释到复杂的法律区分）方面均表现出有前景的性能，同时也指出了改进的方向，例如处理基于计算的问题和结构化列表格式。本研究为_RESOURCE-CONSTRAINED环境中的AI驱动法律辅助工具的部署提供了路径。 

---
# PsyDraw: A Multi-Agent Multimodal System for Mental Health Screening in Left-Behind Children 

**Title (ZH)**: PsyDraw：一种针对留守儿童心理健康筛查的多代理多模态系统 

**Authors**: Yiqun Zhang, Xiaocui Yang, Xiaobai Li, Siyuan Yu, Yi Luan, Shi Feng, Daling Wang, Yifei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.14769)  

**Abstract**: Left-behind children (LBCs), numbering over 66 million in China, face severe mental health challenges due to parental migration for work. Early screening and identification of at-risk LBCs is crucial, yet challenging due to the severe shortage of mental health professionals, especially in rural areas. While the House-Tree-Person (HTP) test shows higher child participation rates, its requirement for expert interpretation limits its application in resource-scarce regions. To address this challenge, we propose PsyDraw, a multi-agent system based on Multimodal Large Language Models that assists mental health professionals in analyzing HTP drawings. The system employs specialized agents for feature extraction and psychological interpretation, operating in two stages: comprehensive feature analysis and professional report generation. Evaluation of HTP drawings from 290 primary school students reveals that 71.03% of the analyzes achieved High Consistency with professional evaluations, 26.21% Moderate Consistency and only 2.41% Low Consistency. The system identified 31.03% of cases requiring professional attention, demonstrating its effectiveness as a preliminary screening tool. Currently deployed in pilot schools, \method shows promise in supporting mental health professionals, particularly in resource-limited areas, while maintaining high professional standards in psychological assessment. 

**Abstract (ZH)**: 在中国，留守儿童（Left-behind Children, LBCs）的数量已经超过6600万，由于家长外出务工，他们面临着严重的心理健康挑战。早期筛查和识别有风险的留守儿童至关重要，但由于心理健康专业人员严重短缺，特别是在农村地区，这一任务充满挑战。尽管绘人测验（House-Tree-Person, HTP）测试显示出更高的儿童参与率，但其需要专家解释的要求限制了其在资源匮乏地区的应用。为了解决这一问题，我们提出了一种基于多模态大语言模型的多智能体系统PsyDraw，以帮助心理健康专业人员分析HTP绘画。该系统采用专门的智能体进行特征提取和心理解释，并分为两个阶段：全面特征分析和专业报告生成。

对290名小学生完成的HTP绘画进行评估，结果显示71.03%的分析与专业评估达到了高一致性，26.21%达到了中等一致性，仅2.41%达到了低一致性。该系统识别出了需要专业关注的31.03%的案例，证明了其作为初步筛查工具的有效性。目前，该系统已在试点学校部署，显示出了在资源有限地区支持心理健康专业人员的潜力，同时在心理评估的专业标准上保持了高水平。 

---
# Query pipeline optimization for cancer patient question answering systems 

**Title (ZH)**: 癌症患者查询系统的查询管道优化research on query pipeline optimization for cancer patient question-answering systems 

**Authors**: Maolin He, Rena Gao, Mike Conway, Brian E. Chapman  

**Link**: [PDF](https://arxiv.org/pdf/2412.14751)  

**Abstract**: Retrieval-augmented generation (RAG) mitigates hallucination in Large Language Models (LLMs) by using query pipelines to retrieve relevant external information and grounding responses in retrieved knowledge. However, query pipeline optimization for cancer patient question-answering (CPQA) systems requires separately optimizing multiple components with domain-specific considerations. We propose a novel three-aspect optimization approach for the RAG query pipeline in CPQA systems, utilizing public biomedical databases like PubMed and PubMed Central. Our optimization includes: (1) document retrieval, utilizing a comparative analysis of NCBI resources and introducing Hybrid Semantic Real-time Document Retrieval (HSRDR); (2) passage retrieval, identifying optimal pairings of dense retrievers and rerankers; and (3) semantic representation, introducing Semantic Enhanced Overlap Segmentation (SEOS) for improved contextual understanding. On a custom-developed dataset tailored for cancer-related inquiries, our optimized RAG approach improved the answer accuracy of Claude-3-haiku by 5.24% over chain-of-thought prompting and about 3% over a naive RAG setup. This study highlights the importance of domain-specific query optimization in realizing the full potential of RAG and provides a robust framework for building more accurate and reliable CPQA systems, advancing the development of RAG-based biomedical systems. 

**Abstract (ZH)**: 检索增强生成（RAG）通过使用查询管道检索相关外部信息并使响应基于检索到的知识来减轻大型语言模型（LLMs）的幻觉现象。然而，针对癌症患者问答（CPQA）系统的查询管道优化需要分别优化多个具有领域特定考虑的组件。我们提出了一个新颖的三方面优化方法，用于CPQA系统的RAG查询管道，利用如PubMed和PubMed Central这样的公共生物医学数据库。我们的优化包括：（1）文档检索，通过对比分析NCBI资源并引入Hybrid Semantic Real-time Document Retrieval（HSRDR）进行比较分析；（2）段落检索，确定最适合的密集检索器和再排序器的配对；和（3）语义表示，引入Semantic Enhanced Overlap Segmentation（SEOS）以提高上下文理解能力。在为癌症相关问题量身定制的数据集中，我们优化后的RAG方法比基于推理链的提示提高了Claude-3-haiku的答案准确性5.24%，比朴素的RAG设置提高了约3%。这项研究突显了在实现RAG潜力时进行领域特定的查询优化的重要性，并提供了一个坚实框架用于构建更准确和可靠的CPQA系统，推动了基于RAG的生物医学系统的发展。 

---
# On Verbalized Confidence Scores for LLMs 

**Title (ZH)**: 关于语言表述的置信分数对LLM的影响 

**Authors**: Daniel Yang, Yao-Hung Hubert Tsai, Makoto Yamada  

**Link**: [PDF](https://arxiv.org/pdf/2412.14737)  

**Abstract**: The rise of large language models (LLMs) and their tight integration into our daily life make it essential to dedicate efforts towards their trustworthiness. Uncertainty quantification for LLMs can establish more human trust into their responses, but also allows LLM agents to make more informed decisions based on each other's uncertainty. To estimate the uncertainty in a response, internal token logits, task-specific proxy models, or sampling of multiple responses are commonly used. This work focuses on asking the LLM itself to verbalize its uncertainty with a confidence score as part of its output tokens, which is a promising way for prompt- and model-agnostic uncertainty quantification with low overhead. Using an extensive benchmark, we assess the reliability of verbalized confidence scores with respect to different datasets, models, and prompt methods. Our results reveal that the reliability of these scores strongly depends on how the model is asked, but also that it is possible to extract well-calibrated confidence scores with certain prompt methods. We argue that verbalized confidence scores can become a simple but effective and versatile uncertainty quantification method in the future. Our code is available at this https URL . 

**Abstract (ZH)**: 大规模语言模型（LLMs）的兴起及其与日常生活的紧密集成使得确保其可信度变得至关重要。对LLMs的不确定性量化可以增强人们对模型响应的信任感，同时也允许LLM代理基于彼此的不确定性做出更明智的决策。为了估算响应中的不确定性，通常使用内部标记对数、任务特定的代理模型或多次预测采样等方法。本研究重点关注LLM本身将其不确定性作为置信度评分输出的一部分，这种方式是一种低开销但适用于各种提示和模型的不确定性量化方法的有前途的方法。通过广泛的数据集基准测试，我们评估了口头化置信度评分的可靠性，涉及不同数据集、模型和提示方法。结果显示，这些评分的可靠性很大程度上取决于对模型的询问方式，但也表明使用某些提示方法可以提取出校准良好的置信度评分。我们认为，口头化置信度评分可以成为未来简单但有效且多功能的不确定性量化方法之一。我们的代码可通过以下链接获取：this https URL 。 

---
# How to Synthesize Text Data without Model Collapse? 

**Title (ZH)**: 如何合成文本数据而不导致模型崩溃？ 

**Authors**: Xuekai Zhu, Daixuan Cheng, Hengli Li, Kaiyan Zhang, Ermo Hua, Xingtai Lv, Ning Ding, Zhouhan Lin, Zilong Zheng, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2412.14689)  

**Abstract**: Model collapse in synthetic data indicates that iterative training on self-generated data leads to a gradual decline in performance. With the proliferation of AI models, synthetic data will fundamentally reshape the web data ecosystem. Future GPT-$\{n\}$ models will inevitably be trained on a blend of synthetic and human-produced data. In this paper, we focus on two questions: what is the impact of synthetic data on language model training, and how to synthesize data without model collapse? We first pre-train language models across different proportions of synthetic data, revealing a negative correlation between the proportion of synthetic data and model performance. We further conduct statistical analysis on synthetic data to uncover distributional shift phenomenon and over-concentration of n-gram features. Inspired by the above findings, we propose token editing on human-produced data to obtain semi-synthetic data. As a proof of concept, we theoretically demonstrate that token-level editing can prevent model collapse, as the test error is constrained by a finite upper bound. We conduct extensive experiments on pre-training from scratch, continual pre-training, and supervised fine-tuning. The results validate our theoretical proof that token-level editing improves data quality and enhances model performance. 

**Abstract (ZH)**: 模型在合成数据上的崩溃表明，迭代训练于自动生成的数据会导致性能逐渐下降。随着AI模型的普及，合成数据将从根本上重塑网络数据生态系统。未来的GPT-$\{n\}$模型不可避免地将采用合成数据和人工生成数据的混合进行训练。在本文中，我们主要关注两个问题：合成数据对语言模型训练的影响以及如何合成数据而不发生模型崩溃？我们首先在不同比例的合成数据上预训练语言模型，揭示了合成数据比例与模型性能之间的负相关关系。我们进一步对合成数据进行统计分析，发现了分布偏移现象和n-克grams特征的过度集中。受到上述发现的启发，我们提出在人工生成数据上进行标记编辑以获取半合成数据。作为概念验证，我们理论证明标记级编辑能防止模型崩溃，因为测试错误受到有限的上限约束。我们进行了全面的实验，涵盖从头开始预训练、持续预训练和有监督微调。实验结果验证了我们的理论证明，即标记级编辑能提高数据质量并增强模型性能。 

---
# Each Fake News is Fake in its Own Way: An Attribution Multi-Granularity Benchmark for Multimodal Fake News Detection 

**Title (ZH)**: 每条虚假新闻都有其独特的虚假方式：多模态虚假新闻检测的归因多层次基准 

**Authors**: Hao Guo, Zihan Ma, Zhi Zeng, Minnan Luo, Weixin Zeng, Jiuyang Tang, Xiang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2412.14686)  

**Abstract**: Social platforms, while facilitating access to information, have also become saturated with a plethora of fake news, resulting in negative consequences. Automatic multimodal fake news detection is a worthwhile pursuit. Existing multimodal fake news datasets only provide binary labels of real or fake. However, real news is alike, while each fake news is fake in its own way. These datasets fail to reflect the mixed nature of various types of multimodal fake news. To bridge the gap, we construct an attributing multi-granularity multimodal fake news detection dataset \amg, revealing the inherent fake pattern. Furthermore, we propose a multi-granularity clue alignment model \our to achieve multimodal fake news detection and attribution. Experimental results demonstrate that \amg is a challenging dataset, and its attribution setting opens up new avenues for future research. 

**Abstract (ZH)**: 社交平台虽然促进了信息的访问，但也充斥着大量的假新闻，带来了负面后果。自动多模态假新闻检测是一个值得追求的研究方向。现有的多模态假新闻数据集仅提供真实或虚假的二元标签。然而，真实新闻虽然相似，但每条假新闻都有其独特的虚假方式。现有的数据集未能反映出多种类型多模态假新闻的混合特性。为了弥补这一差距，我们构建了一个属性多粒度多模态假新闻检测数据集 \AMG，揭示了内在的虚假模式。此外，我们提出了一种多粒度线索对齐模型 \Our，以实现多模态假新闻的检测与归属。实验结果表明，\AMG 是一个具有挑战性的数据集，其归属设置为未来的研究开辟了新途径。 

---
# LLMs as mediators: Can they diagnose conflicts accurately? 

**Title (ZH)**: LLMs作为调解者：它们能否准确诊断冲突？ 

**Authors**: Özgecan Koçak, Phanish Puranam, Afşar Yegin  

**Link**: [PDF](https://arxiv.org/pdf/2412.14675)  

**Abstract**: Prior research indicates that to be able to mediate conflict, observers of disagreements between parties must be able to reliably distinguish the sources of their disagreement as stemming from differences in beliefs about what is true (causality) vs. differences in what they value (morality). In this paper, we test if OpenAI's Large Language Models GPT 3.5 and GPT 4 can perform this task and whether one or other type of disagreement proves particularly challenging for LLM's to diagnose. We replicate study 1 in Koçak et al. (2003), which employes a vignette design, with OpenAI's GPT 3.5 and GPT 4. We find that both LLMs have similar semantic understanding of the distinction between causal and moral codes as humans and can reliably distinguish between them. When asked to diagnose the source of disagreement in a conversation, both LLMs, compared to humans, exhibit a tendency to overestimate the extent of causal disagreement and underestimate the extent of moral disagreement in the moral misalignment condition. This tendency is especially pronounced for GPT 4 when using a proximate scale that relies on concrete language specific to an issue. GPT 3.5 does not perform as well as GPT4 or humans when using either the proximate or the distal scale. The study provides a first test of the potential for using LLMs to mediate conflict by diagnosing the root of disagreements in causal and evaluative codes. 

**Abstract (ZH)**: 先前的研究表明，观察者需要能够可靠地区分冲突双方分歧的原因是基于事实认知的差异（因果关系）还是基于价值观的差异（道德观），才能有效调解冲突。本文通过测试OpenAI的大规模语言模型GPT 3.5和GPT 4能否完成这一任务，以及这两类分歧哪一种对语言模型来说更难以诊断。我们重现了Koçak等人(2003)的研究，使用了案例设计，并采用了OpenAI的GPT 3.5和GPT 4作为实验工具。研究结果显示，这两款语言模型在区分因果和道德准则方面的语义理解与人类相似，并且能够可靠地进行区分。当被要求诊断对话中分歧的根源时，与人类相比，这两款语言模型都倾向于高估因果分歧的程度，而低估道德分歧的程度。这种倾向在使用与具体问题相关的具体语言的邻近量表时尤为明显，GPT 4尤其表现突出。使用邻近量表或远程量表时，GPT 3.5的表现都不如GPT 4或人类。该研究为通过诊断因果和评价编码来使用语言模型调解冲突提供了首次测试。 

---
# Analysis and Visualization of Linguistic Structures in Large Language Models: Neural Representations of Verb-Particle Constructions in BERT 

**Title (ZH)**: 大规模语言模型中的语言结构分析与可视化：BERT 中动词-粒子构造的神经表示分析与可视化 

**Authors**: Hassane Kissane, Achim Schilling, Patrick Krauss  

**Link**: [PDF](https://arxiv.org/pdf/2412.14670)  

**Abstract**: This study investigates the internal representations of verb-particle combinations within transformer-based large language models (LLMs), specifically examining how these models capture lexical and syntactic nuances at different neural network layers. Employing the BERT architecture, we analyse the representational efficacy of its layers for various verb-particle constructions such as 'agree on', 'come back', and 'give up'. Our methodology includes a detailed dataset preparation from the British National Corpus, followed by extensive model training and output analysis through techniques like multi-dimensional scaling (MDS) and generalized discrimination value (GDV) calculations. Results show that BERT's middle layers most effectively capture syntactic structures, with significant variability in representational accuracy across different verb categories. These findings challenge the conventional uniformity assumed in neural network processing of linguistic elements and suggest a complex interplay between network architecture and linguistic representation. Our research contributes to a better understanding of how deep learning models comprehend and process language, offering insights into the potential and limitations of current neural approaches to linguistic analysis. This study not only advances our knowledge in computational linguistics but also prompts further research into optimizing neural architectures for enhanced linguistic precision. 

**Abstract (ZH)**: 本研究探讨了基于变换器的大型语言模型（LLMs）中动词-副词组合的内部表示，特别分析了这些模型在不同神经网络层如何捕获词汇和句法细微差别。利用BERT架构，我们分析了其各层在各种动词-副词构词如“agree on”、“come back”、“give up”方面的表示有效性。研究方法包括从英国国家语料库准备详细的数据集，随后进行广泛的模型训练和输出分析，采用多维标度（MDS）和广义判别值（GDV）计算等技术手段。结果表明，BERT的中层最有效地捕获句法结构，不同动词类别在表示准确性方面存在显著差异。这些发现挑战了语言元素在神经网络处理中的一致性假设，表明网络架构与语言表示之间存在复杂的关系。本研究有助于更好地理解深度学习模型如何理解和处理语言，并提供了当前神经方法在语言分析中的潜力和局限性的洞见。本研究不仅在计算语言学领域推进了我们的知识，还促使进一步研究优化神经架构以提高语言精度。 

---
# Length Controlled Generation for Black-box LLMs 

**Title (ZH)**: 黑箱大型语言模型中的长度可控生成 

**Authors**: Yuxuan Gu, Wenjie Wang, Xiaocheng Feng, Weihong Zhong, Kun Zhu, Lei Huang, Tat-Seng Chua, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2412.14656)  

**Abstract**: Large language models (LLMs) have demonstrated impressive instruction following capabilities, while still struggling to accurately manage the length of the generated text, which is a fundamental requirement in many real-world applications. Existing length control methods involve fine-tuning the parameters of LLMs, which is inefficient and suboptimal for practical use. In this paper, we propose a novel iterative sampling framework for text length control, integrating the Metropolis-Hastings algorithm with an importance sampling acceleration strategy. This framework efficiently and reliably regulates LLMs to generate length-constrained text without modifying the underlying parameters, thereby preserving the original capabilities of LLMs. Experimental results demonstrate that our framework achieves almost 100\% success rates of length control on Llama3.1 for tasks such as length-controlled abstractive summarization and length-constrained instruction following, with minimal additional computational overhead. This also highlights the significant potential of our method for precise length control across a broader range of applications, without compromising the versatility of LLMs. 

**Abstract (ZH)**: 大语言模型（LLMs）在指令执行方面表现出了令人印象深刻的性能，但在生成文本长度管理方面仍然存在问题，这是许多实际应用中的一个基本要求。现有的长度控制方法涉及对LLMs进行微调，这在实际应用中既低效又不尽如人意。在本文中，我们提出了一种新的迭代采样框架以控制文本长度，该框架结合了Metropolis-Hastings算法和重要性采样加速策略。该框架能够高效且可靠地调节LLMs生成长度受限的文本，而无需修改底层参数，从而保持了LLMs的原始能力。实验结果表明，我们的框架在Llama3.1上几乎实现了100%的长度控制成功率，适用于长度控制摘要和长度受限指令执行等任务，并且具有最小的额外计算开销。这还突显了我们的方法在更广泛的范围内实现精确长度控制的巨大潜力，同时不会削弱LLMs的灵活性。 

---
# TOMG-Bench: Evaluating LLMs on Text-based Open Molecule Generation 

**Title (ZH)**: TOMG-Bench：评估基于文本的开放分子生成模型大规模语言模型（LLM）性能 

**Authors**: Jiatong Li, Junxian Li, Yunqing Liu, Dongzhan Zhou, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.14642)  

**Abstract**: In this paper, we propose Text-based Open Molecule Generation Benchmark (TOMG-Bench), the first benchmark to evaluate the open-domain molecule generation capability of LLMs. TOMG-Bench encompasses a dataset of three major tasks: molecule editing (MolEdit), molecule optimization (MolOpt), and customized molecule generation (MolCustom). Each task further contains three subtasks, with each subtask comprising 5,000 test samples. Given the inherent complexity of open molecule generation, we have also developed an automated evaluation system that helps measure both the quality and the accuracy of the generated molecules. Our comprehensive benchmarking of 25 LLMs reveals the current limitations and potential areas for improvement in text-guided molecule discovery. Furthermore, with the assistance of OpenMolIns, a specialized instruction tuning dataset proposed for solving challenges raised by TOMG-Bench, Llama3.1-8B could outperform all the open-source general LLMs, even surpassing GPT-3.5-turbo by 46.5\% on TOMG-Bench. Our codes and datasets are available through this https URL. 

**Abstract (ZH)**: 在本文中，我们提出了基于文本的开放分子生成基准（TOMG-Bench），这是首个评估大规模语言模型（LLM）在开放域分子生成能力的基准。TOMG-Bench 涵盖了三大任务的数据集：分子编辑（MolEdit）、分子优化（MolOpt）和自定义分子生成（MolCustom）。每个任务又进一步分为三个子任务，每个子任务包含5000个测试样本。鉴于开放分子生成固有的复杂性，我们还开发了一套自动评估系统，用以衡量生成分子的质量和准确性。通过全面测试25个LLM，揭示了文本引导分子发现的当前局限性和改进潜力。此外，借助专门为解决TOMG-Bench提出的挑战而设计的OpenMolIns专用指令微调数据集的帮助，Llama3.1-8B在TOMG-Bench上的表现超越了所有开源通用LLM，并且在性能上比GPT-3.5-turbo高出46.5%。我们的代码和数据集可以通过以下链接获取：XXXXXX。 

---
# Learning to Generate Research Idea with Dynamic Control 

**Title (ZH)**: 学习运用动态控制生成研究思路 

**Authors**: Ruochen Li, Liqiang Jing, Chi Han, Jiawei Zhou, Xinya Du  

**Link**: [PDF](https://arxiv.org/pdf/2412.14626)  

**Abstract**: The rapid advancements in large language models (LLMs) have demonstrated their potential to accelerate scientific discovery, particularly in automating the process of research ideation. LLM-based systems have shown promise in generating hypotheses and research ideas. However, current approaches predominantly rely on prompting-based pre-trained models, limiting their ability to optimize generated content effectively. Moreover, they also lack the capability to deal with the complex interdependence and inherent restrictions among novelty, feasibility, and effectiveness, which remains challenging due to the inherent trade-offs among these dimensions, such as the innovation-feasibility conflict. To address these limitations, we for the first time propose fine-tuning LLMs to be better idea proposers and introduce a novel framework that employs a two-stage approach combining Supervised Fine-Tuning (SFT) and controllable Reinforcement Learning (RL). In the SFT stage, the model learns foundational patterns from pairs of research papers and follow-up ideas. In the RL stage, multi-dimensional reward modeling, guided by fine-grained feedback, evaluates and optimizes the generated ideas across key metrics. Dimensional controllers enable dynamic adjustment of generation, while a sentence-level decoder ensures context-aware emphasis during inference. Our framework provides a balanced approach to research ideation, achieving high-quality outcomes by dynamically navigating the trade-offs among novelty, feasibility, and effectiveness. 

**Abstract (ZH)**: 近年来，在大规模语言模型（LLMs）方面的迅速进步展示了其在加速科学发现方面的潜力，尤其是在自动化研究构思过程方面。基于LLM的系统已经显示出生成假设和研究想法的潜力。然而，当前的方法主要依赖于基于提示的预训练模型，限制了它们有效优化生成内容的能力。此外，这些方法在处理新颖性、可行性和有效性之间的复杂相互依赖性和固有约束时能力不足，这些维度之间的固有权衡（如创新与可行性的冲突）使其成为一个挑战。为了解决这些局限性，我们首次提出对LLM进行微调以更好地提出想法，并引入了一个新的框架，该框架采用了结合监督微调（SFT）和可控强化学习（RL）的两阶段方法。在SFT阶段，模型通过研究论文及其后续想法的成对学习基础模式。在RL阶段，基于精细反馈的多维奖励模型评估并优化生成的想法，并通过关键指标进行优化。维控制器使生成过程能够动态调整，而句子级解码器确保在推理过程中对上下文信息进行关注。我们的框架提供了一种平衡的研究构思方法，通过动态导航新颖性、可行性和有效性之间的权衡，实现高质量的结果。 

---
# How good is GPT at writing political speeches for the White House? 

**Title (ZH)**: 《GPT撰写白宫政治演说的能力如何？》

这个标题翻译成中文既保持了原意，又符合学术论文的规范。 

**Authors**: Jacques Savoy  

**Link**: [PDF](https://arxiv.org/pdf/2412.14617)  

**Abstract**: Using large language models (LLMs), computers are able to generate a written text in response to a us er request. As this pervasive technology can be applied in numerous contexts, this study analyses the written style of one LLM called GPT by comparing its generated speeches with those of the recent US presidents. To achieve this objective, the State of the Union (SOTU) addresses written by Reagan to Biden are contrasted to those produced by both GPT-3.5 and GPT-4.o versions. Compared to US presidents, GPT tends to overuse the lemma "we" and produce shorter messages with, on average, longer sentences. Moreover, GPT opts for an optimistic tone, opting more often for political (e.g., president, Congress), symbolic (e.g., freedom), and abstract terms (e.g., freedom). Even when imposing an author's style to GPT, the resulting speech remains distinct from addresses written by the target author. Finally, the two GPT versions present distinct characteristics, but both appear overall dissimilar to true presidential messages. 

**Abstract (ZH)**: 利用大规模语言模型（LLMs），计算机能够根据用户请求生成书面文本。由于这项普及性技术可以在众多领域得到应用，本研究通过将一个名为GPT的LLM生成的演讲与最近几任美国总统的演讲进行比较，分析了GPT的书面风格。为此，研究对比了里根至拜登历任总统的国情咨文（SOTU），并与GPT-3.5和GPT-4版本生成的演讲进行对比。与美国总统的演讲相比，GPT倾向于过度使用“我们”这一词汇，生成的文本平均而言句子较长，且更倾向于使用乐观的语调，偏好使用如“总统”、“国会”等政治性词汇，如“自由”等象征性词汇以及“自由”等抽象性词汇。即便对GPT施加某种作者的写作风格，生成的演讲仍然与目标作者的演讲风格不同。此外，两个GPT版本都表现出不同的特点，但总体上都不同于真正的总统演讲。 

---
# HarmonicEval: Multi-modal, Multi-task, Multi-criteria Automatic Evaluation Using a Vision Language Model 

**Title (ZH)**: 谐波评估：基于视觉语言模型的多模态、多任务、多指标自动化评估方法 

**Authors**: Masanari Ohi, Masahiro Kaneko, Naoaki Okazaki, Nakamasa Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2412.14613)  

**Abstract**: Vision-language models (VLMs) have shown impressive abilities in text and image understanding. However, existing metrics for evaluating the text generated by VLMs focus exclusively on overall quality, leading to two limitations: 1) it is challenging to identify which aspects of the text need improvement from the overall score; 2) metrics may overlook specific evaluation criteria when predicting an overall score. To address these limitations, we propose HarmonicEval, a reference-free evaluation metric that aggregates criterion-wise scores to produce the overall score in a bottom-up manner. Furthermore, we construct the Multi-task Multi-criteria Human Evaluation (MMHE) dataset, which comprises 18,000 expert human judgments across four vision-language tasks. Our experiments demonstrate that HarmonicEval achieves higher correlations with human judgments than conventional metrics while providing numerical scores for each criterion. 

**Abstract (ZH)**: 视觉语言模型（VLMs）在文本和图像理解方面展现了令人印象深刻的性能。然而，当前用于评估VLMs生成文本的指标仅专注于整体质量，这带来了两个限制：1）难以从整体分数中识别哪些方面的文本需要改进；2）指标在预测整体分数时可能会忽略具体的评估标准。为了解决这些限制，我们提出了一种参考自由的评估指标HarmonicEval，该指标通过自下而上的方式聚合具体的评分，以生成整体评分。此外，我们构建了多任务多标准人类评估（MMHE）数据集，该数据集包含四个视觉语言任务的18,000个专家人类判断。我们的实验表明，HarmonicEval在与人类判断的相关性方面优于传统的评估指标，并且为每个标准提供了数值评分。 

---
# KARRIEREWEGE: A Large Scale Career Path Prediction Dataset 

**Title (ZH)**: 职业路径：大规模职业路径预测数据集 

**Authors**: Elena Senger, Yuri Campbell, Rob van der Goot, Barbara Plank  

**Link**: [PDF](https://arxiv.org/pdf/2412.14612)  

**Abstract**: Accurate career path prediction can support many stakeholders, like job seekers, recruiters, HR, and project managers. However, publicly available data and tools for career path prediction are scarce. In this work, we introduce KARRIEREWEGE, a comprehensive, publicly available dataset containing over 500k career paths, significantly surpassing the size of previously available datasets. We link the dataset to the ESCO taxonomy to offer a valuable resource for predicting career trajectories. To tackle the problem of free-text inputs typically found in resumes, we enhance it by synthesizing job titles and descriptions resulting in KARRIEREWEGE+. This allows for accurate predictions from unstructured data, closely aligning with real-world application challenges. We benchmark existing state-of-the-art (SOTA) models on our dataset and a prior benchmark and observe improved performance and robustness, particularly for free-text use cases, due to the synthesized data. 

**Abstract (ZH)**: 准确的职业路径预测可以支持众多利益相关者，如求职者、招聘人员、人力资源部门和项目管理者。然而，公开可用的职业路径预测数据和工具较为稀缺。在本研究中，我们介绍了一个综合性的公开数据集KARRIEREWEGE，其中包括超过50万条职业路径，显著超过了先前可用数据集的规模。我们将数据集与ESCOT分类法链接，为预测职业轨迹提供了宝贵的资源。为解决简历中常见的自由文本输入问题，我们通过合成职位名称和描述，将其提升为KARRIEREWEGE+，从而能够从非结构化数据中进行准确预测，更好地适应实际应用中的挑战。我们在我们的数据集和先前的基准测试上对现有的最先进的（SOTA）模型进行了性能测试，并观察到改善了性能和鲁棒性，特别是在自由文本应用场景中，由于使用了合成数据。 

---
# Beyond Guilt: Legal Judgment Prediction with Trichotomous Reasoning 

**Title (ZH)**: 超越罪恶观：基于三元推理的法律判决预测 

**Authors**: Kepu Zhang, Haoyue Yang, Xu Tang, Weijie Yu, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2412.14588)  

**Abstract**: In legal practice, judges apply the trichotomous dogmatics of criminal law, sequentially assessing the elements of the offense, unlawfulness, and culpability to determine whether an individual's conduct constitutes a crime. Although current legal large language models (LLMs) show promising accuracy in judgment prediction, they lack trichotomous reasoning capabilities due to the absence of an appropriate benchmark dataset, preventing them from predicting innocent outcomes. As a result, every input is automatically assigned a charge, limiting their practical utility in legal contexts. To bridge this gap, we introduce LJPIV, the first benchmark dataset for Legal Judgment Prediction with Innocent Verdicts. Adhering to the trichotomous dogmatics, we extend three widely-used legal datasets through LLM-based augmentation and manual verification. Our experiments with state-of-the-art legal LLMs and novel strategies that integrate trichotomous reasoning into zero-shot prompting and fine-tuning reveal: (1) current legal LLMs have significant room for improvement, with even the best models achieving an F1 score of less than 0.3 on LJPIV; and (2) our strategies notably enhance both in-domain and cross-domain judgment prediction accuracy, especially for cases resulting in an innocent verdict. 

**Abstract (ZH)**: 在法律实践中，法官利用刑法的三元法学框架，依次评估犯罪构成要素、违法性和罪过性，以确定一个人的行为是否构成犯罪。虽然当前的法律大型语言模型（LLMs）在判决预测方面表现出色，但由于缺乏适当的基准数据集，它们无法进行三元推理，这限制了它们在法律情境中的实际应用。因此，每次输入都会被自动判定有罪，从而限制了它们的实用性。为弥补这一差距，我们提出了LJPIV（Legal Judgment Prediction with Innocent Verdicts），这是首个包含无罪判决基准数据集的法律判决预测数据集。遵循三元法学框架，我们通过基于LLM的扩充和人工验证，扩展了三个广泛使用的法律数据集。我们的实验结果表明：（1）当前的法律LLMs仍存在较大的改进空间，最好的模型在LJPIV上的F1分数低于0.3；（2）我们的策略显著提高了法律情境内的及跨情境判决预测准确性，尤其是在无罪判决的情况下。 

---
# Simulation-Free Hierarchical Latent Policy Planning for Proactive Dialogues 

**Title (ZH)**: 无需仿真分层潜在策略规划以实现主动对话 

**Authors**: Tao He, Lizi Liao, Yixin Cao, Yuanxing Liu, Yiheng Sun, Zerui Chen, Ming Liu, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2412.14584)  

**Abstract**: Recent advancements in proactive dialogues have garnered significant attention, particularly for more complex objectives (e.g. emotion support and persuasion). Unlike traditional task-oriented dialogues, proactive dialogues demand advanced policy planning and adaptability, requiring rich scenarios and comprehensive policy repositories to develop such systems. However, existing approaches tend to rely on Large Language Models (LLMs) for user simulation and online learning, leading to biases that diverge from realistic scenarios and result in suboptimal efficiency. Moreover, these methods depend on manually defined, context-independent, coarse-grained policies, which not only incur high expert costs but also raise concerns regarding their completeness. In our work, we highlight the potential for automatically discovering policies directly from raw, real-world dialogue records. To this end, we introduce a novel dialogue policy planning framework, LDPP. It fully automates the process from mining policies in dialogue records to learning policy planning. Specifically, we employ a variant of the Variational Autoencoder to discover fine-grained policies represented as latent vectors. After automatically annotating the data with these latent policy labels, we propose an Offline Hierarchical Reinforcement Learning (RL) algorithm in the latent space to develop effective policy planning capabilities. Our experiments demonstrate that LDPP outperforms existing methods on two proactive scenarios, even surpassing ChatGPT with only a 1.8-billion-parameter LLM. 

**Abstract (ZH)**: 近年来，在主动对话领域的进展引起了广泛关注，尤其在更复杂的任务上（如情感支持和说服）。与传统的以任务为导向的对话不同，主动对话需要更高级的策略规划和适应性，因此需要丰富的场景和全面的政策库来发展这样的系统。然而，现有的方法往往依赖于大语言模型（LLMs）进行用户模拟和在线学习，这导致了与现实场景偏差较大的偏见，从而降低了效率。此外，这些方法依赖于手动定义、上下文无关且粒度粗的策略，不仅增加了专家成本，还引发了关于其完整性的问题。在我们的研究中，我们强调了直接从原始的真实对话记录中自动发现策略的潜力。为此，我们引入了一个新颖的对话策略规划框架——LDPP。该框架完全自动化了从挖掘对话记录中的策略到学习策略规划的整个过程。具体来说，我们使用一种变分自编码器的变体来发现用潜在向量表示的细粒度策略。在自动为数据标注这些潜在策略标签后，我们提出了一种在潜在空间中的离线分层强化学习（RL）算法，以开发有效的策略规划能力。我们的实验表明，LDPP在两个主动对话场景中的表现优于现有方法，甚至只用一个参数量为1.8亿的LLM即超过了ChatGPT。 

---
# CORD: Balancing COnsistency and Rank Distillation for Robust Retrieval-Augmented Generation 

**Title (ZH)**: CORD：平衡一致性和排名蒸馏以实现稳健的检索增强生成 

**Authors**: Youngwon Lee, Seung-won Hwang, Daniel Campos, Filip Graliński, Zhewei Yao, Yuxiong He  

**Link**: [PDF](https://arxiv.org/pdf/2412.14581)  

**Abstract**: With the adoption of retrieval-augmented generation (RAG), large language models (LLMs) are expected to ground their generation to the retrieved contexts. Yet, this is hindered by position bias of LLMs, failing to evenly attend to all contexts. Previous work has addressed this by synthesizing contexts with perturbed positions of gold segment, creating a position-diversified train set. We extend this intuition to propose consistency regularization with augmentation and distillation. First, we augment each training instance with its position perturbation to encourage consistent predictions, regardless of ordering. We also distill behaviors of this pair, although it can be counterproductive in certain RAG scenarios where the given order from the retriever is crucial for generation quality. We thus propose CORD, balancing COnsistency and Rank Distillation. CORD adaptively samples noise-controlled perturbations from an interpolation space, ensuring both consistency and respect for the rank prior. Empirical results show this balance enables CORD to outperform consistently in diverse RAG benchmarks. 

**Abstract (ZH)**: 以下是对原文的翻译，符合学术规范：

借助检索增强生成（RAG）技术，大型语言模型（LLMs）应将其生成与检索到的上下文关联起来。然而，LLMs 的位置偏见阻碍了其均匀地关注所有上下文。现有研究通过将黄金片段的位置进行扰动来综合生成上下文，从而创建一个位置多样化的数据集。我们在此基础上提出了一种结合增强和蒸馏的一致性正则化方法。首先，我们在每个训练实例中增加其位置扰动，以在不考虑排列顺序的情况下促进一致预测。此外，我们还通过这个配对进行蒸馏，尽管在某些RAG场景中，检索器提供的顺序对于生成质量至关重要，这可能会产生反作用。因此，我们提出了CORD（Consistency and Rank Distillation），在保持一致性和尊重排名先验之间进行了平衡。CORD 自适应地从插值空间中选择噪声控制的扰动，确保同时保持一致性和排名先验。实验结果显示，这种平衡使CORD在多种RAG基准测试中表现出色。 

---
# CitaLaw: Enhancing LLM with Citations in Legal Domain 

**Title (ZH)**: CitaLaw：在法律领域增强语言模型的引文方法 

**Authors**: Kepu Zhang, Weijie Yu, Sunhao Dai, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2412.14556)  

**Abstract**: In this paper, we propose CitaLaw, the first benchmark designed to evaluate LLMs' ability to produce legally sound responses with appropriate citations. CitaLaw features a diverse set of legal questions for both laypersons and practitioners, paired with a comprehensive corpus of law articles and precedent cases as a reference pool. This framework enables LLM-based systems to retrieve supporting citations from the reference corpus and align these citations with the corresponding sentences in their responses. Moreover, we introduce syllogism-inspired evaluation methods to assess the legal alignment between retrieved references and LLM-generated responses, as well as their consistency with user questions. Extensive experiments on 2 open-domain and 7 legal-specific LLMs demonstrate that integrating legal references substantially enhances response quality. Furthermore, our proposed syllogism-based evaluation method exhibits strong agreement with human judgments. 

**Abstract (ZH)**: 在本文中，我们提出了CitaLaw，这是首个旨在评估大型语言模型(LLM)生成合法合规响应并恰当引用参考文献能力的基准测试。CitaLaw包含了一个涵盖不同层面用户（包括普通民众和法律从业人员）的多样化法律问题集，并配有一个全面的法律文献和先例案例的参考库。该框架使基于LLM的系统能够从参考库中检索支持性引用，并将这些引文与响应中的相应句子进行匹配。此外，我们引入了受三段论启发的评估方法，以评估检索到的参考文献与LLM生成的响应之间的法律一致性，以及这些引用与用户问题的吻合程度。通过对2个开放领域和7个特定于法律领域的LLM进行广泛的实验，我们证明了集成法律参考显著提高了响应质量。此外，我们提出的基于三段论的评估方法与人工判断高度一致。 

---
# ClusterTalk: Corpus Exploration Framework using Multi-Dimensional Exploratory Search 

**Title (ZH)**: ClusterTalk: 基于多维度探索性搜索的语料库探索框架 

**Authors**: Ashish Chouhan, Saifeldin Mandour, Michael Gertz  

**Link**: [PDF](https://arxiv.org/pdf/2412.14533)  

**Abstract**: Exploratory search of large text corpora is essential in domains like biomedical research, where large amounts of research literature are continuously generated. This paper presents ClusterTalk (The demo video and source code are available at: this https URL), a framework for corpus exploration using multi-dimensional exploratory search. Our system integrates document clustering with faceted search, allowing users to interactively refine their exploration and ask corpus and document-level queries. Compared to traditional one-dimensional search approaches like keyword search or clustering, this system improves the discoverability of information by encouraging a deeper interaction with the corpus. We demonstrate the functionality of the ClusterTalk framework based on four million PubMed abstracts for the four-year time frame. 

**Abstract (ZH)**: 大规模文本语料库的探索性搜索在生物医学等研究领域至关重要，因为大量研究文献不断生成。本文介绍了一种名为ClusterTalk的框架（可在以下链接查看演示视频和源代码：<https://this.url/>），该框架采用多维探索性搜索技术进行语料库探索。我们的系统结合了文档聚类和主题搜索，使用户能够交互式地细化搜索并提出语料库和文档级别的查询。与传统的单维搜索方法（如关键词搜索或聚类）相比，该系统通过促进用户与语料库的深层次交互，提高了信息的发现性。基于四年内共计四百万篇PubMed摘要，本文演示了ClusterTalk框架的功能。 

---
# Multi-Level Optimal Transport for Universal Cross-Tokenizer Knowledge Distillation on Language Models 

**Title (ZH)**: 多层级最优运输在语言模型跨令牌器知识蒸馏中的universal应用研究 

**Authors**: Xiao Cui, Mo Zhu, Yulei Qin, Liang Xie, Wengang Zhou, Houqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.14528)  

**Abstract**: Knowledge distillation (KD) has become a prevalent technique for compressing large language models (LLMs). Existing KD methods are constrained by the need for identical tokenizers (i.e., vocabularies) between teacher and student models, limiting their versatility in handling LLMs of different architecture families. In this paper, we introduce the Multi-Level Optimal Transport (MultiLevelOT), a novel approach that advances the optimal transport for universal cross-tokenizer knowledge distillation. Our method aligns the logit distributions of the teacher and the student at both token and sequence levels using diverse cost matrices, eliminating the need for dimensional or token-by-token correspondence. At the token level, MultiLevelOT integrates both global and local information by jointly optimizing all tokens within a sequence to enhance robustness. At the sequence level, we efficiently capture complex distribution structures of logits via the Sinkhorn distance, which approximates the Wasserstein distance for divergence measures. Extensive experiments on tasks such as extractive QA, generative QA, and summarization demonstrate that the MultiLevelOT outperforms state-of-the-art cross-tokenizer KD methods under various settings. Our approach is robust to different student and teacher models across model families, architectures, and parameter sizes. 

**Abstract (ZH)**: 知识蒸馏（KD）已成为压缩大型语言模型（LLMs）的一种广泛使用的技术。现有的KD方法受制于教师模型和学生模型需要具有相同的分词器（即词表），这限制了它们在处理不同架构家族的LLMs时的灵活性。在本文中，我们提出了多级最优传输（MultiLevelOT），这是一种新颖的方法，用于推动通用跨分词器的知识蒸馏中的最优传输。我们的方法通过使用多种成本矩阵在令牌和序列级别对教师和学生的logit分布进行对齐，从而消除了在维度或令牌级进行对应的要求。在令牌级别，MultiLevelOT 通过联合优化序列中的所有令牌来整合全局和局部信息，以增强鲁棒性。在序列级别，我们通过Sinkhorn距离高效地捕捉logit的复杂分布结构，这可以近似 Wasserstein 距离作为偏差度量。在诸如抽取式问答、生成式问答和摘要等任务上的广泛实验表明，MultiLevelOT 在各种设置下均优于最先进的跨分词器KD方法。我们的方法对不同架构家族、模型类型和参数规模的学生和教师模型都具有鲁棒性。 

---
# PA-RAG: RAG Alignment via Multi-Perspective Preference Optimization 

**Title (ZH)**: PA-RAG：基于多视角偏好优化的RAG对齐方法 

**Authors**: Jiayi Wu, Hengyi Cai, Lingyong Yan, Hao Sun, Xiang Li, Shuaiqiang Wang, Dawei Yin, Ming Gao  

**Link**: [PDF](https://arxiv.org/pdf/2412.14510)  

**Abstract**: The emergence of Retrieval-augmented generation (RAG) has alleviated the issues of outdated and hallucinatory content in the generation of large language models (LLMs), yet it still reveals numerous limitations. When a general-purpose LLM serves as the RAG generator, it often suffers from inadequate response informativeness, response robustness, and citation quality. Past approaches to tackle these limitations, either by incorporating additional steps beyond generating responses or optimizing the generator through supervised fine-tuning (SFT), still failed to align with the RAG requirement thoroughly. Consequently, optimizing the RAG generator from multiple preference perspectives while maintaining its end-to-end LLM form remains a challenge. To bridge this gap, we propose Multiple Perspective Preference Alignment for Retrieval-Augmented Generation (PA-RAG), a method for optimizing the generator of RAG systems to align with RAG requirements comprehensively. Specifically, we construct high-quality instruction fine-tuning data and multi-perspective preference data by sampling varied quality responses from the generator across different prompt documents quality scenarios. Subsequently, we optimize the generator using SFT and Direct Preference Optimization (DPO). Extensive experiments conducted on four question-answer datasets across three LLMs demonstrate that PA-RAG can significantly enhance the performance of RAG generators. Our code and datasets are available at this https URL. 

**Abstract (ZH)**: 检索增强生成（RAG）的出现减轻了大型语言模型（LLMs）生成内容时出现的内容过时和幻觉问题，但仍然暴露了许多局限性。当通用的LLM作为RAG生成器时，它经常遭受响应信息量不足、响应稳健性和引文质量差的问题。过去的解决方法，无论是通过生成响应之外的额外步骤，还是通过监督微调（SFT）优化生成器，仍然未能充分满足RAG的要求。因此，同时从多个偏好视角优化RAG生成器并保持其端到端的LLM形式，仍然是一项挑战。为解决这一差距，我们提出了一种多视角偏好对齐方法（PA-RAG），用于优化RAG系统中的生成器以全面符合RAG要求。具体而言，我们通过从不同提示文档质量场景中采样各种质量的响应构建高质量的指令微调数据和多视角偏好数据。随后，我们使用SFT和直接偏好优化（DPO）来优化生成器。在三种LLM的四个问答数据集上进行的大量实验表明，PA-RAG可以显著提高RAG生成器的性能。我们的代码和数据集可在以下网址获得：this https URL。 

---
# Do Large Language Models Defend Inferentialist Semantics?: On the Logical Expressivism and Anti-Representationalism of LLMs 

**Title (ZH)**: 大型语言模型保护演绎主义语义学吗？——关于大型语言模型的逻辑表达主义与反代表性观点 

**Authors**: Yuzuki Arai, Sho Tsugawa  

**Link**: [PDF](https://arxiv.org/pdf/2412.14501)  

**Abstract**: The philosophy of language, which has historically been developed through an anthropocentric lens, is now being forced to move towards post-anthropocentrism due to the advent of large language models (LLMs) like ChatGPT (OpenAI), Claude (Anthropic), which are considered to possess linguistic abilities comparable to those of humans. Traditionally, LLMs have been explained through distributional semantics as their foundational semantics. However, recent research is exploring alternative foundational semantics beyond distributional semantics. This paper proposes Robert Brandom's inferentialist semantics as an suitable foundational semantics for LLMs, specifically focusing on the issue of linguistic representationalism within this post-anthropocentric trend. Here, we show that the anti-representationalism and logical expressivism of inferential semantics, as well as quasi-compositionality, are useful in interpreting the characteristics and behaviors of LLMs. Further, we propose a \emph{consensus theory of truths} for LLMs. This paper argues that the characteristics of LLMs challenge mainstream assumptions in philosophy of language, such as semantic externalism and compositionality. We believe the argument in this paper leads to a re-evaluation of anti\hyphen{}representationalist views of language, potentially leading to new developments in the philosophy of language. 

**Abstract (ZH)**: 语言哲学在历史上一直是通过人类中心主义的观点来发展的，但随着大型语言模型（LLMs）如ChatGPT（OpenAI）和Claude（Anthropic）的出现，它现在被迫向后人类中心主义方向发展。这些LLMs被认为具有与人类相当的语言能力。传统上，LLMs是通过分布语义学来解释其基础语义的。然而，最近的研究正在探索超出分布语义学的其他基础语义学。本文提议罗伯特·布兰福德的推理主义语义学作为LLMs的基础语义学，并特别关注后人类中心主义趋势下语言表征主义的问题。在此基础上，我们表明推理主义语义学中的反表征主义、逻辑表达主义以及准组合性在解释LLMs的特征和行为方面是有用的。此外，我们为LLMs提出了一个“共识真理理论”。本文认为，LLMs的特征挑战了语言哲学中的主流假设，如语义外部主义和组合性。我们认为本文中的论点将导致对语言的反表征观点进行重新评估，这可能会带来语言哲学的新发展。 

---
# Why We Build Local Large Language Models: An Observational Analysis from 35 Japanese and Multilingual LLMs 

**Title (ZH)**: 我们构建本地大型语言模型的原因：来自35个日语和其他多语言大型语言模型的观察性分析 

**Authors**: Koshiro Saito, Sakae Mizuki, Masanari Ohi, Taishi Nakamura, Taihei Shiotani, Koki Maeda, Youmi Ma, Kakeru Hattori, Kazuki Fujii, Takumi Okamoto, Shigeki Ishida, Hiroya Takamura, Rio Yokota, Naoaki Okazaki  

**Link**: [PDF](https://arxiv.org/pdf/2412.14471)  

**Abstract**: Why do we build local large language models (LLMs)? What should a local LLM learn from the target language? Which abilities can be transferred from other languages? Do language-specific scaling laws exist? To explore these research questions, we evaluated 35 Japanese, English, and multilingual LLMs on 19 evaluation benchmarks for Japanese and English, taking Japanese as a local language. Adopting an observational approach, we analyzed correlations of benchmark scores, and conducted principal component analysis (PCA) on the scores to derive \textit{ability factors} of local LLMs. We found that training on English text can improve the scores of academic subjects in Japanese (JMMLU). In addition, it is unnecessary to specifically train on Japanese text to enhance abilities for solving Japanese code generation, arithmetic reasoning, commonsense, and reading comprehension tasks. In contrast, training on Japanese text could improve question-answering tasks about Japanese knowledge and English-Japanese translation, which indicates that abilities for solving these two tasks can be regarded as \textit{Japanese abilities} for LLMs. Furthermore, we confirmed that the Japanese abilities scale with the computational budget for Japanese text. 

**Abstract (ZH)**: 我们为什么构建本地大型语言模型（Local Large Language Models, LLMs）？本地LLMs应从目标语言中学到哪些能力？哪些能力可以从其他语言中转移过来？语言特有的缩放法则是否存在？为探索这些问题，我们评估了35个日语、英语和多语言LLMs在19项日语和英语的评估基准上，以日语作为本地语言。采用观察性方法，我们分析了基准分数的相关性，并通过主成分分析（PCA）对分数进行分析，提取出本地LLMs的能力因素。我们发现，使用英语文本训练可以提高日语（JMMLU）学术主题的成绩。此外，不必专门针对日语文本来增强解决日语代码生成、算术推理、常识和阅读理解任务的能力。相反，使用日语文本训练有助于提高有关日语知识的问题回答任务和日语-英语翻译任务的成绩，这表明能够解决这两种任务的能力可以被视为LLMs的“日语能力”。此外，我们确认了这些日语能力与日语文本的计算预算成比例。 

---
# Agent-SafetyBench: Evaluating the Safety of LLM Agents 

**Title (ZH)**: Agent-SafetyBench: 评估大型语言模型代理的安全性 

**Authors**: Zhexin Zhang, Shiyao Cui, Yida Lu, Jingzhuo Zhou, Junxiao Yang, Hongning Wang, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2412.14470)  

**Abstract**: As large language models (LLMs) are increasingly deployed as agents, their integration into interactive environments and tool use introduce new safety challenges beyond those associated with the models themselves. However, the absence of comprehensive benchmarks for evaluating agent safety presents a significant barrier to effective assessment and further improvement. In this paper, we introduce Agent-SafetyBench, a comprehensive benchmark designed to evaluate the safety of LLM agents. Agent-SafetyBench encompasses 349 interaction environments and 2,000 test cases, evaluating 8 categories of safety risks and covering 10 common failure modes frequently encountered in unsafe interactions. Our evaluation of 16 popular LLM agents reveals a concerning result: none of the agents achieves a safety score above 60%. This highlights significant safety challenges in LLM agents and underscores the considerable need for improvement. Through quantitative analysis, we identify critical failure modes and summarize two fundamental safety detects in current LLM agents: lack of robustness and lack of risk awareness. Furthermore, our findings suggest that reliance on defense prompts alone is insufficient to address these safety issues, emphasizing the need for more advanced and robust strategies. We release Agent-SafetyBench at \url{this https URL} to facilitate further research and innovation in agent safety evaluation and improvement. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）越来越多地作为智能代理被部署，它们与交互环境的集成以及工具的使用引入了超出模型本身相关安全挑战的新问题。然而，缺乏全面的代理安全评估基准已成为有效评估和改进的重大障碍。本文介绍了一个全面的基准——Agent-SafetyBench，旨在评估LLM代理的安全性。Agent-SafetyBench 包含 349 个交互环境和 2000 个测试案例，评估了 8 类安全风险，并涵盖了 10 种常见失败模式，这些模式经常出现在不安全的交互中。对16个流行LLM代理进行的评估结果令人担忧：这些代理没有一个能够达到60%以上的安全分值。这突显了LLM代理在安全性方面存在的显著挑战，并强调了亟待改进的必要性。通过定量分析，我们识别了关键的失败模式，并总结了当前LLM代理中的两种基本安全性检测：缺乏稳健性以及缺乏风险意识。此外，我们的研究发现，仅仅依赖于防御提示是不足以解决这些安全问题的，这进一步强调了需要更高级和更稳健的策略的必要性。我们已将Agent-SafetyBench发布在 \url{this https URL}，以促进代理安全性评估和改进的进一步研究与创新。 

---
# From Human Annotation to LLMs: SILICON Annotation Workflow for Management Research 

**Title (ZH)**: 从人工标注到LLM：SILICON标注工作流在管理研究中的应用 

**Authors**: Xiang Cheng, Raveesh Mayya, João Sedoc  

**Link**: [PDF](https://arxiv.org/pdf/2412.14461)  

**Abstract**: Unstructured text data annotation and analysis are fundamental to management research, often relying on human annotators through crowdsourcing platforms. While Large Language Models (LLMs) promise to provide a cost-effective and efficient alternative to human annotation, there lacks a systematic workflow that evaluate when LLMs are suitable or how to proceed with LLM-based text annotation in a reproducible manner. This paper addresses this methodological gap by introducing the ``SILICON" (\textbf{S}ystematic \textbf{I}nference with \textbf{L}LMs for \textbf{I}nformation \textbf{C}lassificati\textbf{o}n and \textbf{N}otation) workflow. The workflow integrates established principles of human annotation with systematic prompt optimization and model selection, addressing challenges such as developing robust annotation guidelines, establishing high-quality human baselines, optimizing prompts, and ensuring reproducibility across LLMs. We validate the SILICON workflow through seven case studies covering common management research tasks, including business proposal evaluation, dialog intent and breakdown analysis, review attribute detection. Our findings highlight the importance of validating annotation guideline agreement, the superiority of expert-developed human baselines over crowdsourced ones, the iterative nature of prompt optimization, and the necessity of testing multiple LLMs. Notably, we propose a regression-based methodology to empirically compare LLM outputs across prompts and models. Our workflow advances management research by establishing reproducible processes for LLM-based annotation that maintain scientific rigor. We provide practical guidance for researchers to effectively navigate the evolving landscape of generative AI tools effectively while maintaining transparency and reproducibility. 

**Abstract (ZH)**: 无结构文本数据的标注与分析是管理研究的基础，通常依赖于通过众包平台的人工标注员。虽然大型语言模型（LLMs）承诺提供一种成本效益高且高效的替代人工标注的方法，但缺乏系统的工作流程来评估在何种情况下LLMs适合使用，以及如何以可重复的方式进行基于LLM的文本标注。本文通过引入“SILICON”（系统化LLM推断以分类与标注信息）工作流程来弥补这一方法论上的缺口。该工作流程将现有的人工标注原则与系统化的提示优化和模型选择相结合，解决了如制定稳健的标注指南、建立高质量的人基线、优化提示以及确保跨LLMs的可重复性等挑战。我们通过七个案例研究验证了SILICON工作流程，这些案例覆盖了常见的管理研究任务，包括商业提案评估、对话意图和分解分析、评论属性检测。研究结果强调了验证标注指南一致性的必要性、专家开发的人基线优于众包基线的重要性、提示优化的迭代性质以及测试多个LLMs的必要性。值得注意的是，我们提出了基于回归的方法来实证比较不同提示和模型下的LLM输出。我们的工作流程通过为基于LLM的标注建立可重复的过程，推动了管理研究的科学严谨性。我们还为研究人员提供了实用的指导，帮助他们有效地应对生成式AI工具不断变化的环境，同时保持透明性和可重复性。 

---
# ORBIT: Cost-Effective Dataset Curation for Large Language Model Domain Adaptation with an Astronomy Case Study 

**Title (ZH)**: ORBIT：面向天文领域的大语言模型领域适应的数据集整理成本效益方法研究 

**Authors**: Eric Modesitt, Ke Yang, Spencer Hulsey, Chengxiang Zhai, Volodymyr Kindratenko  

**Link**: [PDF](https://arxiv.org/pdf/2412.14436)  

**Abstract**: Recent advances in language modeling demonstrate the need for high-quality domain-specific training data, especially for tasks that require specialized knowledge. General-purpose models, while versatile, often lack the depth needed for expert-level tasks because of limited domain-specific information. Domain adaptation training can enhance these models, but it demands substantial, high-quality data. To address this, we propose ORBIT, a cost-efficient methodology for curating massive, high-quality domain-specific datasets from noisy web sources, tailored for training specialist large language models. Using astronomy as a primary case study, we refined the 1.3T-token FineWeb-Edu dataset into a high-quality, 10B-token subset focused on astronomy. Fine-tuning \textsc{LLaMA-3-8B} on a 1B-token astronomy subset improved performance on the MMLU astronomy benchmark from 69\% to 76\% and achieved top results on AstroBench, an astronomy-specific benchmark. Moreover, our model (Orbit-LLaMA) outperformed \textsc{LLaMA-3-8B-base}, with GPT-4o evaluations preferring it in 73\% of cases across 1000 astronomy-specific questions. Additionally, we validated ORBIT's generalizability by applying it to law and medicine, achieving a significant improvement of data quality compared to an unfiltered baseline. We open-source the ORBIT methodology, including the curated datasets, the codebase, and the resulting model at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 最近的语言模型研究显示，高质量的专业领域训练数据对于需要专业领域知识的任务至关重要。通用模型虽然功能多样，但在专家级任务上往往缺乏深度，因为它们包含的专业领域信息有限。领域适应训练可以增强这些模型，但这种训练需要大量的高质量数据。为了解决这一问题，我们提出了一种名为ORBIT的成本效益方法，用于从嘈杂的网络来源中收集大量高质量的专业领域数据，以训练专家级大型语言模型。以天文学为主要案例研究，我们从1.3T tokens的FineWeb-Edu数据集中提炼出一个高质量的100亿tokens的专业天文学子集。将\textsc{LLaMA-3-8B}微调于10亿tokens的专业天文学子集，其在MMLU天文学基准测试中的表现从69%提高到76%，并在专为天文学设计的AstroBench基准测试中取得了最佳成绩。此外，我们的模型（Orbit-LLaMA）优于\textsc{LLaMA-3-8B-base}，超过73%的GPT-4o问题上更被偏好。此外，我们通过将其应用于法律和医学来验证ORBIT的泛化能力，相较于未过滤的基础数据集，数据质量有了显著提升。我们开源了ORBIT方法论，包括整理好的数据集、代码库以及生成的模型，在\href{this https URL}{this https URL}提供下载。 

---
# All-in-One Tuning and Structural Pruning for Domain-Specific LLMs 

**Title (ZH)**: 面向特定领域的LLM的一体化调优与结构剪枝 

**Authors**: Lei Lu, Zhepeng Wang, Ruexue Bao, Mengbing Wang, Fangyi Li, Yawen Wu, Weiwen Jiang, Jie Xu, Yanzhi Wang, Shangqian Gao  

**Link**: [PDF](https://arxiv.org/pdf/2412.14426)  

**Abstract**: Existing pruning techniques for large language models (LLMs) targeting domain-specific applications typically follow a two-stage process: pruning the pretrained general-purpose LLMs and then fine-tuning the pruned LLMs on specific domains. However, the pruning decisions, derived from the pretrained weights, remain unchanged during fine-tuning, even if the weights have been updated. Therefore, such a combination of the pruning decisions and the finetuned weights may be suboptimal, leading to non-negligible performance degradation. To address these limitations, we propose ATP: All-in-One Tuning and Structural Pruning, a unified one-stage structural pruning and fine-tuning approach that dynamically identifies the current optimal substructure throughout the fine-tuning phase via a trainable pruning decision generator. Moreover, given the limited available data for domain-specific applications, Low-Rank Adaptation (LoRA) becomes a common technique to fine-tune the LLMs. In ATP, we introduce LoRA-aware forward and sparsity regularization to ensure that the substructures corresponding to the learned pruning decisions can be directly removed after the ATP process. ATP outperforms the state-of-the-art two-stage pruning methods on tasks in the legal and healthcare domains. More specifically, ATP recovers up to 88% and 91% performance of the dense model when pruning 40% parameters of LLaMA2-7B and LLaMA3-8B models, respectively. 

**Abstract (ZH)**: 现有的针对特定领域应用的大语言模型（LLMs）的剪枝技术通常遵循一个两阶段的过程：首先剪枝预训练的一般用途LLMs，然后在特定领域对剪枝后的LLMs进行微调。然而，在微调过程中，从预训练权重得出的剪枝决策保持不变，即使权重已经更新。因此，这种剪枝决策与微调后的权重的结合可能是次优的，导致非忽略不计的性能下降。为了解决这些局限性，我们提出了一种统一的一阶段结构剪枝和微调方法——All-in-One Tuning and Structural Pruning（ATP），该方法通过可训练的剪枝决策生成器，在整个微调阶段动态地识别出当前最优的子结构。此外，鉴于特定领域应用的数据有限，低秩适应（LoRA）成为常用的方法来微调LLMs。在ATP中，我们引入了LoRA意识前向传播和稀疏正则化，以确保所学习的剪枝决策对应的子结构可以在ATP过程之后直接被移除。在法律和医疗领域的任务上，ATP在两阶段剪枝方法中表现出更好的性能。具体而言，当修剪LLaMA2-7B和LLaMA3-8B模型中的40%参数时，ATP分别恢复了密集模型88%和91%的性能。 

---
# ECG-Byte: A Tokenizer for End-to-End Generative Electrocardiogram Language Modeling 

**Title (ZH)**: ECG-Byte：一种用于端到端生成心电图语言建模的分词器 

**Authors**: William Han, Chaojing Duan, Michael A. Rosenberg, Emerson Liu, Ding Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2412.14373)  

**Abstract**: Large Language Models (LLMs) have shown remarkable adaptability across domains beyond text, specifically electrocardiograms (ECGs). More specifically, there is a growing body of work exploring the task of generating text from a multi-channeled ECG and corresponding textual prompt. Current approaches typically involve pretraining an ECG-specific encoder with a self-supervised learning (SSL) objective and using the features output by the pretrained encoder to finetune a LLM for natural language generation (NLG). However, these methods are limited by 1) inefficiency from two-stage training and 2) interpretability challenges with encoder-generated features. To address these limitations, we introduce ECG-Byte, an adapted byte pair encoding (BPE) tokenizer pipeline for autoregressive language modeling of ECGs. This approach compresses and encodes ECG signals into tokens, enabling end-to-end LLM training by combining ECG and text tokens directly, while being much more interpretable since the ECG tokens can be directly mapped back to the original signal. Using ECG-Byte, we achieve competitive performance in NLG tasks in only half the time and ~48% of the data required by two-stage approaches. 

**Abstract (ZH)**: 大型语言模型（LLMs）在文本之外的领域中显示出了非凡的适应性，尤其是在心电图（ECGs）领域。具体而言，越来越多的研究工作专注于从多通道ECGs及其对应的文本提示生成文本的任务。当前的方法通常包括使用自监督学习（SSL）目标预训练一个ECG特定的编码器，并使用预训练编码器输出的特征来微调一个LLM以进行自然语言生成（NLG）。然而，这些方法受限于1）两阶段训练的低效率，2）编码器生成的特征可解释性差。为了解决这些限制，我们提出了ECG-Byte，这是一种适应性的字节对编码（BPE）分词器流水线，用于ECGs的自回归语言建模。该方法通过将ECG信号压缩和编码成tokens，直接结合ECG tokens和文本tokens进行端到端的LLM训练，同时由于ECG tokens可以直接映射回原始信号，因此更具可解释性。使用ECG-Byte，我们仅在两阶段方法所需时间的一半和所需数据量的约48%下，便实现了竞争力的NLG任务性能。 

---
# Memorization Over Reasoning? Exposing and Mitigating Verbatim Memorization in Large Language Models' Character Understanding Evaluation 

**Title (ZH)**: 过度记忆而忽视推理？揭示并减轻大型语言模型在角色理解评估中verbatim记忆的问题 

**Authors**: Yuxuan Jiang, Francis Ferraro  

**Link**: [PDF](https://arxiv.org/pdf/2412.14368)  

**Abstract**: Recently, Large Language Models (LLMs) have shown impressive performance in character understanding tasks, such as analyzing the roles, personalities, and relationships of fictional characters. However, the extensive pre-training corpora used by LLMs raise concerns that they may rely on memorizing popular fictional works rather than genuinely understanding and reasoning about them. In this work, we argue that 'gist memory'-capturing essential meaning - should be the primary mechanism for character understanding tasks, as opposed to 'verbatim memory' - exact match of a string. We introduce a simple yet effective method to mitigate mechanized memorization in character understanding evaluations while preserving the essential implicit cues needed for comprehension and reasoning. Our approach reduces memorization-driven performance on popular fictional works from 96% accuracy to 72% and results in up to an 18% drop in accuracy across various character understanding tasks. These findings underscore the issue of data contamination in existing benchmarks, which often measure memorization rather than true character understanding. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）在角色理解任务中展现出了惊人的性能，例如分析虚构角色的角色、性格和关系。然而，LLMs 所依赖的大量预训练数据集引发了对其是否真正理解和推理这些作品，而不仅仅是记忆流行的作品的担忧。在本文中，我们认为在角色理解任务中，提取关键意义的“概要记忆”（gist memory）应是主要机制，而不同于逐字匹配的“字面记忆”（verbatim memory）。我们提出了一种简单而有效的方法，可以在保持理解和推理所需的关键隐含线索的同时，减轻角色理解评估中的机械化记忆现象。我们的方法将对流行虚构作品的性能准确率从96%降低到72%，并在多种角色理解任务中导致最高达18%的准确率下降。这些发现强调了现有基准数据中存在的数据污染问题，这些基准通常测量的是记忆而非真正的人物理解。 

---
# State Space Models are Strong Text Rerankers 

**Title (ZH)**: 状态空间模型是强大的文本重排序器 

**Authors**: Zhichao Xu, Jinghua Yan, Ashim Gupta, Vivek Srikumar  

**Link**: [PDF](https://arxiv.org/pdf/2412.14354)  

**Abstract**: Transformers dominate NLP and IR; but their inference inefficiencies and challenges in extrapolating to longer contexts have sparked interest in alternative model architectures. Among these, state space models (SSMs) like Mamba offer promising advantages, particularly $O(1)$ time complexity in inference. Despite their potential, SSMs' effectiveness at text reranking -- a task requiring fine-grained query-document interaction and long-context understanding -- remains underexplored.
This study benchmarks SSM-based architectures (specifically, Mamba-1 and Mamba-2) against transformer-based models across various scales, architectures, and pre-training objectives, focusing on performance and efficiency in text reranking tasks. We find that (1) Mamba architectures achieve competitive text ranking performance, comparable to transformer-based models of similar size; (2) they are less efficient in training and inference compared to transformers with flash attention; and (3) Mamba-2 outperforms Mamba-1 in both performance and efficiency. These results underscore the potential of state space models as a transformer alternative and highlight areas for improvement in future IR applications. 

**Abstract (ZH)**: transformer在自然语言处理(NLP)和信息检索(IR)中占据主导地位，但它们在推理效率方面的不足以及在处理更长文本上下文时的外推挑战激发了对替代模型架构的兴趣。在这些模型中，状态空间模型（SSMs）如Mamba展现出有希望的优势，特别是它们在推理方面具有$O(1)$的时间复杂度。尽管存在这些优势，但SSMs在文本重排序任务——这一任务需要精细的查询-文档交互和长文本上下文理解——中的有效性仍被很大程度上未被探索。

本研究对比了基于SSMs的架构（具体来说是Mamba-1和Mamba-2）与基于transformer的模型在不同规模、架构和预训练目标下的性能，并重点关注了这些模型在文本重排序任务中的表现和效率。我们发现，（1）Mamba架构在文本排序性能方面达到了与相似规模的基于transformer的模型相当的水平；（2）它们在训练和推理效率方面不如使用flash attention的transformers；（3）Mamba-2在性能和效率上都优于Mamba-1。这些结果突显了状态空间模型作为transformer替代品的潜在价值，并指出了未来信息检索应用中的改进领域。 

---
# A Survey on LLM Inference-Time Self-Improvement 

**Title (ZH)**: LLM推理时自我改进综述 

**Authors**: Xiangjue Dong, Maria Teleki, James Caverlee  

**Link**: [PDF](https://arxiv.org/pdf/2412.14352)  

**Abstract**: Techniques that enhance inference through increased computation at test-time have recently gained attention. In this survey, we investigate the current state of LLM Inference-Time Self-Improvement from three different perspectives: Independent Self-improvement, focusing on enhancements via decoding or sampling methods; Context-Aware Self-Improvement, leveraging additional context or datastore; and Model-Aided Self-Improvement, achieving improvement through model collaboration. We provide a comprehensive review of recent relevant studies, contribute an in-depth taxonomy, and discuss challenges and limitations, offering insights for future research. 

**Abstract (ZH)**: 在测试时通过增加计算来增强推理的技术最近引起了广泛关注。在本文综述中，我们从三个不同的视角调查了LLM在推理时自我改进的现状：独立自我改进，侧重于通过解码或采样方法的增强；上下文感知自我改进，利用额外的上下文或数据存储库；以及模型辅助自我改进，通过模型协作实现改进。我们对近年来的相关研究进行了全面回顾，贡献了一个深入的分类框架，并讨论了挑战和限制，为未来的研究提供了见解。 

---
# Is Peer-Reviewing Worth the Effort? 

**Title (ZH)**: 《peer-reviewing 是否值得付出努力？》

在翻译学术论文题目时，保持原题目的结构和意思尽量一致是很重要的。这里将“Is Peer-Reviewing Worth the Effort?”翻译为“peer-reviewing 是否值得付出努力？”这样的表达既保留了原文的意思，又符合中文的表达习惯。 

**Authors**: Kenneth Church, Raman Chandrasekar, John E. Ortega, Ibrahim Said Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2412.14351)  

**Abstract**: How effective is peer-reviewing in identifying important papers? We treat this question as a forecasting task. Can we predict which papers will be highly cited in the future based on venue and "early returns" (citations soon after publication)? We show early returns are more predictive than venue. Finally, we end with constructive suggestions to address scaling challenges: (a) too many submissions and (b) too few qualified reviewers. 

**Abstract (ZH)**: 同行评审在识别重要论文方面有多有效？我们将这一问题视为一项预测任务：我们能否根据会议和“早期引文”（即发表后不久的引用）来预测哪些论文将会在未来被高度引用？我们发现早期引文比会议更具预测性。最后，我们提出了应对规模挑战的建设性建议：(a) 处理过多的投稿，(b) 增加合格的评审人。 

---
# Semantic Role Labeling of NomBank Partitives 

**Title (ZH)**: NomBank 部分句法的语义角色标注 

**Authors**: Adam Meyers, Advait Pravin Savant, John E. Ortega  

**Link**: [PDF](https://arxiv.org/pdf/2412.14328)  

**Abstract**: This article is about Semantic Role Labeling for English partitive nouns (5%/REL of the price/ARG1; The price/ARG1 rose 5 percent/REL) in the NomBank annotated corpus. Several systems are described using traditional and transformer-based machine learning, as well as ensembling. Our highest scoring system achieves an F1 of 91.74% using "gold" parses from the Penn Treebank and 91.12% when using the Berkeley Neural parser. This research includes both classroom and experimental settings for system development. 

**Abstract (ZH)**: 本文探讨了英语部分名词（如“5%/REL of the price/ARG1；价格/ARG1上涨5%/REL）的语义角色标注问题，研究基于NomBank标注语料库进行。文中描述了使用传统机器学习和基于变换器的方法构建的多种系统，并进行了集成学习。使用宾夕法尼亚树库的“金色”解析，我们的最高得分为F1值91.74%，使用伯克利神经解析器时为91.12%。该研究既包括了课堂环境下的系统开发，也包含了实验环境下的开发。 

---
# The Role of Handling Attributive Nouns in Improving Chinese-To-English Machine Translation 

**Title (ZH)**: 《属性名词的处理在提升中文到英文机器翻译中的作用》 

**Authors**: Haohao, Wang, Adam Meyers, John E. Ortega, Rodolfo Zevallos  

**Link**: [PDF](https://arxiv.org/pdf/2412.14323)  

**Abstract**: Translating between languages with drastically different grammatical conventions poses challenges, not just for human interpreters but also for machine translation systems. In this work, we specifically target the translation challenges posed by attributive nouns in Chinese, which frequently cause ambiguities in English translation. By manually inserting the omitted particle X ('DE'). In news article titles from the Penn Chinese Discourse Treebank, we developed a targeted dataset to fine-tune Hugging Face Chinese to English translation models, specifically improving how this critical function word is handled. This focused approach not only complements the broader strategies suggested by previous studies but also offers a practical enhancement by specifically addressing a common error type in Chinese-English translation. 

**Abstract (ZH)**: 翻译如下，符合学术规范：

将具有巨大语法差异的语言进行互译，不仅对人类译者提出了挑战，也对机器翻译系统提出了挑战。在本研究中，我们特别针对汉语中的归属性名词翻译所引起的歧义问题进行了研究。通过手动插入被省略的助词“的”（X），我们基于宾州中文语篇树库（Penn Chinese Discourse Treebank）的新闻文章标题，开发了一个专门的数据集，以微调Hugging Face的中文到英文翻译模型，特别改善了对该关键功能词的处理方式。这种集中方式不仅补充了前人研究中提出的更广泛策略，还提供了一个实用的增强，特别针对中文到英文翻译中的常见错误类型进行了处理。 

---
# Multi-OphthaLingua: A Multilingual Benchmark for Assessing and Debiasing LLM Ophthalmological QA in LMICs 

**Title (ZH)**: 多语眼科语言：评估和消除LMICs中LLM眼科问答偏差的多语言基准 

**Authors**: David Restrepo, Chenwei Wu, Zhengxu Tang, Zitao Shuai, Thao Nguyen Minh Phan, Jun-En Ding, Cong-Tinh Dao, Jack Gallifant, Robyn Gayle Dychiao, Jose Carlo Artiaga, André Hiroshi Bando, Carolina Pelegrini Barbosa Gracitelli, Vincenz Ferrer, Leo Anthony Celi, Danielle Bitterman, Michael G Morley, Luis Filipe Nakayama  

**Link**: [PDF](https://arxiv.org/pdf/2412.14304)  

**Abstract**: Current ophthalmology clinical workflows are plagued by over-referrals, long waits, and complex and heterogeneous medical records. Large language models (LLMs) present a promising solution to automate various procedures such as triaging, preliminary tests like visual acuity assessment, and report summaries. However, LLMs have demonstrated significantly varied performance across different languages in natural language question-answering tasks, potentially exacerbating healthcare disparities in Low and Middle-Income Countries (LMICs). This study introduces the first multilingual ophthalmological question-answering benchmark with manually curated questions parallel across languages, allowing for direct cross-lingual comparisons. Our evaluation of 6 popular LLMs across 7 different languages reveals substantial bias across different languages, highlighting risks for clinical deployment of LLMs in LMICs. Existing debiasing methods such as Translation Chain-of-Thought or Retrieval-augmented generation (RAG) by themselves fall short of closing this performance gap, often failing to improve performance across all languages and lacking specificity for the medical domain. To address this issue, We propose CLARA (Cross-Lingual Reflective Agentic system), a novel inference time de-biasing method leveraging retrieval augmented generation and self-verification. Our approach not only improves performance across all languages but also significantly reduces the multilingual bias gap, facilitating equitable LLM application across the globe. 

**Abstract (ZH)**: 当前的眼科学临床工作流程受到过度转诊、等待时间长以及复杂多样的医疗记录的困扰。大规模语言模型（LLMs）为自动化各种程序（如分诊、初步测试如视力评估以及报告总结）提供了有希望的解决方案。然而，LLMs在自然语言问答任务中的表现因语言不同而呈现出显著差异，这可能加剧了低收入和中等收入国家（LMIC）的医疗卫生不平等现象。本研究介绍了第一个多语言眼科学问答基准，其中包括手动策展的跨语言问题，允许直接进行跨语言比较。我们在7种不同语言中对6种流行的LLMs进行评估，发现不同语言之间存在显著的偏差，揭示了在LMIC中临床部署LLMs所面临的风险。现有去偏方法，如翻译链式思考或检索增强生成（RAG），单独使用时难以弥合这一表现差距，往往无法在所有语言中提高性能，也不具备医疗领域的专属性。为解决这一问题，我们提出了CLARA（跨语言反思主动系统），这是一种利用检索增强生成和自我验证的新型推理时去偏方法。我们的方法不仅在所有语言中提高了性能，而且显著减少了多语言偏差差距，促进了全球范围内LLMs的公平应用。 

---
# Fake News Detection: Comparative Evaluation of BERT-like Models and Large Language Models with Generative AI-Annotated Data 

**Title (ZH)**: 假新闻检测：基于BERT类模型与生成AI标注数据的大型语言模型的比较评估 

**Authors**: haina Raza, Drai Paulen-Patterson, Chen Ding  

**Link**: [PDF](https://arxiv.org/pdf/2412.14276)  

**Abstract**: Fake news poses a significant threat to public opinion and social stability in modern society. This study presents a comparative evaluation of BERT-like encoder-only models and autoregressive decoder-only large language models (LLMs) for fake news detection. We introduce a dataset of news articles labeled with GPT-4 assistance (an AI-labeling method) and verified by human experts to ensure reliability. Both BERT-like encoder-only models and LLMs were fine-tuned on this dataset. Additionally, we developed an instruction-tuned LLM approach with majority voting during inference for label generation. Our analysis reveals that BERT-like models generally outperform LLMs in classification tasks, while LLMs demonstrate superior robustness against text perturbations. Compared to weak labels (distant supervision) data, the results show that AI labels with human supervision achieve better classification results. This study highlights the effectiveness of combining AI-based annotation with human oversight and demonstrates the performance of different families of machine learning models for fake news detection 

**Abstract (ZH)**: 虚假新闻对现代社会公众意见和稳定构成重大威胁。本研究对基于BERT的编码器模型和自回归解码器大型语言模型（LLMs）在虚假新闻检测中的表现进行了比较评估。我们引入了一个经过GPT-4辅助（AI标注方法）标注并由人工专家验证的数据集，以确保其可靠性。这两个模型都在该数据集上进行了微调。此外，我们还开发了一种在生成标签时采用多数投票的指令调优LLM方法。我们的分析表明，在分类任务中，基于BERT的模型通常优于LLMs，而LLMs在对抗文本扰动方面表现更稳健。相比弱标签（疏远监督），结果显示人工监督的AI标签能够实现更好的分类效果。本研究强调了结合AI标注和人工监督的有效性，并展示了不同机器学习模型家族在虚假新闻检测中的性能。 

---
# Tokenisation is NP-Complete 

**Title (ZH)**: tokenisation 是 NP 完全问题 

**Authors**: Philip Whittington, Gregor Bachmann, Tiago Pimentel  

**Link**: [PDF](https://arxiv.org/pdf/2412.15210)  

**Abstract**: In this work, we prove the NP-completeness of two variants of tokenisation, defined as the problem of compressing a dataset to at most $\delta$ symbols by either finding a vocabulary directly (direct tokenisation), or selecting a sequence of merge operations (bottom-up tokenisation). 

**Abstract (ZH)**: 在本文中，我们证明了两种变体的标记化问题的NP完全性，其中标记化问题定义为将数据集压缩为至多$\delta$个符号，可以通过直接找到词汇表（直接标记化）或选择一系列合并操作（自底向上的标记化）来解决。 

---
# Critical-Questions-of-Thought: Steering LLM reasoning with Argumentative Querying 

**Title (ZH)**: 基于批判性问题的思考：通过论辩性查询引导大语言模型的推理 

**Authors**: Federico Castagna, Isabel Sassoon, Simon Parsons  

**Link**: [PDF](https://arxiv.org/pdf/2412.15177)  

**Abstract**: Studies have underscored how, regardless of the recent breakthrough and swift advances in AI research, even state-of-the-art Large Language models (LLMs) continue to struggle when performing logical and mathematical reasoning. The results seem to suggest that LLMs still work as (highly advanced) data pattern identifiers, scoring poorly when attempting to generalise and solve reasoning problems the models have never previously seen or that are not close to samples presented in their training data. To address this compelling concern, this paper makes use of the notion of critical questions from the literature on argumentation theory, focusing in particular on Toulmin's model of argumentation. We show that employing these critical questions can improve the reasoning capabilities of LLMs. By probing the rationale behind the models' reasoning process, the LLM can assess whether some logical mistake is occurring and correct it before providing the final reply to the user prompt. The underlying idea is drawn from the gold standard of any valid argumentative procedure: the conclusion is valid if it is entailed by accepted premises. Or, to paraphrase such Aristotelian principle in a real-world approximation, characterised by incomplete information and presumptive logic, the conclusion is valid if not proved otherwise. This approach successfully steers the models' output through a reasoning pipeline, resulting in better performance against the baseline and its Chain-of-Thought (CoT) implementation. To this end, an extensive evaluation of the proposed approach on the MT-Bench Reasoning and Math tasks across a range of LLMs is provided. 

**Abstract (ZH)**: 研究表明，尽管近年来人工智能研究取得了突破性进展，并取得了快速的进步，最先进的大规模语言模型（LLMs）在进行逻辑和数学推理时仍然面临着挑战。研究结果似乎表明，LLMs 仍然主要作为（高度发达）数据模式识别器运作，当尝试解决模型之前未见过的问题或偏离其训练数据样本时，它们很难概括和求解推理问题。为应对这一重要关切，本论文借鉴了论述理论文献中的核心问题概念，特别关注于塔特姆（Toulmin）的论辩模型。我们表明，利用这些核心问题可以通过改善LLMs的推理能力。通过对模型推理过程的探究，LLMs可以评估是否发生了逻辑错误，并在提供最终回复给用户提示之前纠正这些错误。这一理念来源于任何有效论述过程的金标准：如果结论被公认的前提所蕴含，结论就是有效的。或者说，用现实世界中的近似推理来重新表述这一古老的亚里士多德原则，在信息不完整和假定逻辑的背景下，结论的有效性取决于它没有被证明为无效。这种方法成功地引导了模型的输出沿着推理管道前进，从而在基线及其论证链（CoT）实现上获得了更好的性能。为此，本文在MT-Bench推理和数学任务上对多种LLMs进行了广泛评估，展示了所提出方法的成效。 

---
# Prompt-A-Video: Prompt Your Video Diffusion Model via Preference-Aligned LLM 

**Title (ZH)**: Prompt-A-Video: 通过偏好对齐的LLM提示您的视频扩散模型 

**Authors**: Yatai Ji, Jiacheng Zhang, Jie Wu, Shilong Zhang, Shoufa Chen, Chongjian GE, Peize Sun, Weifeng Chen, Wenqi Shao, Xuefeng Xiao, Weilin Huang, Ping Luo  

**Link**: [PDF](https://arxiv.org/pdf/2412.15156)  

**Abstract**: Text-to-video models have made remarkable advancements through optimization on high-quality text-video pairs, where the textual prompts play a pivotal role in determining quality of output videos. However, achieving the desired output often entails multiple revisions and iterative inference to refine user-provided prompts. Current automatic methods for refining prompts encounter challenges such as Modality-Inconsistency, Cost-Discrepancy, and Model-Unaware when applied to text-to-video diffusion models. To address these problem, we introduce an LLM-based prompt adaptation framework, termed as Prompt-A-Video, which excels in crafting Video-Centric, Labor-Free and Preference-Aligned prompts tailored to specific video diffusion model. Our approach involves a meticulously crafted two-stage optimization and alignment system. Initially, we conduct a reward-guided prompt evolution pipeline to automatically create optimal prompts pool and leverage them for supervised fine-tuning (SFT) of the LLM. Then multi-dimensional rewards are employed to generate pairwise data for the SFT model, followed by the direct preference optimization (DPO) algorithm to further facilitate preference alignment. Through extensive experimentation and comparative analyses, we validate the effectiveness of Prompt-A-Video across diverse generation models, highlighting its potential to push the boundaries of video generation. 

**Abstract (ZH)**: 文本到视频模型通过优化高质量的图文对取得了显著进展，其中文本提示在决定输出视频的质量上起着至关重要的作用。然而，实现理想的输出往往需要多次修订和迭代推理以细化用户提供的提示。应用于文本到视频扩散模型的当前自动提示优化方法遇到了模态不一致性、成本差异和模型无感知等问题。为解决这些问题，我们提出了一种基于大规模语言模型（LLM）的提示适配框架，称为Prompt-A-Video，该框架能够生成以视频为中心、无需人工劳动且符合用户偏好的特定视频扩散模型优化的提示。我们的方法涉及一个精心设计的两阶段优化和对齐系统。首先，我们进行了一种奖励导向的提示进化的流水线，自动创建最优提示池，并利用这些提示进行LLM的监督微调（SFT）。然后，使用多维度奖励生成监督微调模型的成对数据，接着采用直接偏好优化（DPO）算法进一步促进偏好对齐。通过广泛的实验和对比分析，我们验证了Prompt-A-Video的有效性，并在多种生成模型中展示了其推动视频生成边界的可能性。 

---
# Associative memory inspires improvements for in-context learning using a novel attention residual stream architecture 

**Title (ZH)**: 关联记忆启发了一种新型注意残差流架构在上下文学习中的改进 

**Authors**: Thomas F Burns, Tomoki Fukai, Christopher J Earls  

**Link**: [PDF](https://arxiv.org/pdf/2412.15113)  

**Abstract**: Large language models (LLMs) demonstrate an impressive ability to utilise information within the context of their input sequences to appropriately respond to data unseen by the LLM during its training procedure. This ability is known as in-context learning (ICL). Humans and non-human animals demonstrate similar abilities, however their neural architectures differ substantially from LLMs. Despite this, a critical component within LLMs, the attention mechanism, resembles modern associative memory models, widely used in and influenced by the computational neuroscience community to model biological memory systems. Using this connection, we introduce an associative memory model capable of performing ICL. We use this as inspiration for a novel residual stream architecture which allows information to directly flow between attention heads. We test this architecture during training within a two-layer Transformer and show its ICL abilities manifest more quickly than without this modification. We then apply our architecture in small language models with 8 million parameters, focusing on attention head values, with results also indicating improved ICL performance at this larger and more naturalistic scale. 

**Abstract (ZH)**: 大型语言模型（LLMs）展示了在处理输入序列上下文信息时，能够适当地响应训练过程中未见过的数据的能力，这种能力被称为上下文学习（ICL）。人类和非人类动物也表现出类似的技能，尽管它们的神经架构与LLMs有显著差异。尽管如此，LLMs中的一个关键组件——注意力机制，与现代关联记忆模型类似，这些模型在计算神经科学社区中广泛用于模拟生物记忆系统。利用这一联系，我们引入了一种关联记忆模型，该模型能够执行ICL。我们受到这一灵感启发，提出了一种新的残差流架构，允许信息在注意力头之间直接流动。我们在两层Transformer模型训练过程中测试了这种架构，并显示了其ICL能力相较于未进行此修改时的提高速度更快。然后，我们在具有800万个参数的小型语言模型中应用了这一架构，重点在于注意力头值，并且结果表明，在更大且更自然的尺度下，这一架构也显示出改进的ICL性能。 

---
# A Cross-Domain Study of the Use of Persuasion Techniques in Online Disinformation 

**Title (ZH)**: 跨域研究：在线虚假信息中劝说技巧的使用分析 

**Authors**: João A. Leite, Olesya Razuvayevskaya, Carolina Scarton, Kalina Bontcheva  

**Link**: [PDF](https://arxiv.org/pdf/2412.15098)  

**Abstract**: Disinformation, irrespective of domain or language, aims to deceive or manipulate public opinion, typically through employing advanced persuasion techniques. Qualitative and quantitative research on the weaponisation of persuasion techniques in disinformation has been mostly topic-specific (e.g., COVID-19) with limited cross-domain studies, resulting in a lack of comprehensive understanding of these strategies. This study employs a state-of-the-art persuasion technique classifier to conduct a large-scale, multi-domain analysis of the role of 16 persuasion techniques in disinformation narratives. It shows how different persuasion techniques are employed disproportionately in different disinformation domains. We also include a detailed case study on climate change disinformation, highlighting how linguistic, psychological, and cultural factors shape the adaptation of persuasion strategies to fit unique thematic contexts. 

**Abstract (ZH)**: 不实信息无论涉及哪个领域或何种语言，其目的在于欺骗或操纵公众舆论，通常通过采用先进的说服技术。对说服技术在不实信息中的军事化应用进行的定性和定量研究主要集中在特定主题上（例如，COVID-19），跨领域的研究有限，导致对这些策略缺乏全面的理解。本研究采用最先进的说服技术分类器，开展大规模、多领域的分析，研究16种说服技术在不实信息叙事中的作用。研究结果显示，不同的说服技术在不同类型的不实信息领域中的应用存在显著差异。我们还详细分析了气候变化领域的不实信息，展示了语言、心理和文化因素如何塑造说服策略以适应独特的主题背景。 

---
# Till the Layers Collapse: Compressing a Deep Neural Network through the Lenses of Batch Normalization Layers 

**Title (ZH)**: 直到层坍缩：通过批量归一化层的视角压缩深度神经网络 

**Authors**: Zhu Liao, Nour Hezbri, Victor Quétu, Van-Tam Nguyen, Enzo Tartaglione  

**Link**: [PDF](https://arxiv.org/pdf/2412.15077)  

**Abstract**: Today, deep neural networks are widely used since they can handle a variety of complex tasks. Their generality makes them very powerful tools in modern technology. However, deep neural networks are often overparameterized. The usage of these large models consumes a lot of computation resources. In this paper, we introduce a method called \textbf{T}ill the \textbf{L}ayers \textbf{C}ollapse (TLC), which compresses deep neural networks through the lenses of batch normalization layers. By reducing the depth of these networks, our method decreases deep neural networks' computational requirements and overall latency. We validate our method on popular models such as Swin-T, MobileNet-V2, and RoBERTa, across both image classification and natural language processing (NLP) tasks. 

**Abstract (ZH)**: 当今，深度神经网络因其能够处理各种复杂任务而被广泛使用。其通用性使它们成为现代技术中非常强大的工具。然而，深度神经网络通常过于参数化，使用这些大型模型会消耗大量的计算资源。本文介绍了一种名为**Till the Layers Collapse (TLC)** 的方法，该方法通过批量标准化层的视角来压缩深度神经网络。通过减少这些网络的深度，我们的方法降低了深度神经网络的计算需求和整体延迟。我们在 Swin-T、MobileNet-V2 和 RoBERTa 等流行模型上对图像分类和自然语言处理（NLP）任务进行了验证。 

---
# Large Language Models and Code Security: A Systematic Literature Review 

**Title (ZH)**: 大型语言模型与代码安全：一项系统文献综述 

**Authors**: Enna Basic, Alberto Giaretta  

**Link**: [PDF](https://arxiv.org/pdf/2412.15004)  

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools for automating various programming tasks, including security-related ones, such as detecting and fixing vulnerabilities. Despite their promising capabilities, when required to produce or modify pre-existing code, LLMs could introduce vulnerabilities unbeknown to the programmer. When analyzing code, they could miss clear vulnerabilities or signal nonexistent ones. In this Systematic Literature Review (SLR), we aim to investigate both the security benefits and potential drawbacks of using LLMs for a variety of code-related tasks. In particular, first we focus on the types of vulnerabilities that could be introduced by LLMs, when used for producing code. Second, we analyze the capabilities of LLMs to detect and fix vulnerabilities, in any given code, and how the prompting strategy of choice impacts their performance in these two tasks. Last, we provide an in-depth analysis on how data poisoning attacks on LLMs can impact performance in the aforementioned tasks. 

**Abstract (ZH)**: 大规模语言模型（LLMs）已经成为自动化各种编程任务的强大工具，包括安全相关的任务，如检测和修复漏洞。尽管它们具有令人鼓舞的能力，但在要求其生成或修改现有代码时，LLMs有可能引入程序员未知的漏洞。在分析代码时，它们可能会忽视明显的漏洞或误报不存在的漏洞。在这项系统文献综述（SLR）中，我们旨在探讨使用LLMs进行各种代码相关任务时的安全优势与潜在缺点。首先，我们关注LLMs在生成代码时可能引入的漏洞类型。其次，我们分析LLMs在任意代码中检测和修复漏洞的能力，以及选择的提示策略对其在这两项任务中表现的影响。最后，我们对数据投毒攻击对LLMs性能的影响进行了深入分析。 

---
# Movie2Story: A framework for understanding videos and telling stories in the form of novel text 

**Title (ZH)**: Movie2Story：一种理解视频和以新颖文本形式讲述故事的框架 

**Authors**: Kangning Li, Zheyang Jia, Anyu Ying  

**Link**: [PDF](https://arxiv.org/pdf/2412.14965)  

**Abstract**: Multimodal video-to-text models have made considerable progress, primarily in generating brief descriptions of video content. However, there is still a deficiency in generating rich long-form text descriptions that integrate both video and audio. In this paper, we introduce a framework called M2S, designed to generate novel-length text by combining audio, video, and character recognition. M2S includes modules for video long-form text description and comprehension, audio-based analysis of emotion, speech rate, and character alignment, and visual-based character recognition alignment. By integrating multimodal information using the large language model GPT4o, M2S stands out in the field of multimodal text generation. We demonstrate the effectiveness and accuracy of M2S through comparative experiments and human evaluation. Additionally, the model framework has good scalability and significant potential for future research. 

**Abstract (ZH)**: 多模态视频到文本模型在生成视频内容的简洁描述方面取得了显著的进步。然而，这些模型在生成包含视频和音频信息的丰富长文本描述方面仍存在不足。本文介绍了一种名为M2S的框架，用于结合音频、视频和字符识别生成长篇文本。M2S包括用于生成和理解视频长篇文本、基于音频的情感分析、语速分析和字符对齐，以及基于视觉的字符识别对齐的模块。通过使用大型语言模型GPT4o整合多模态信息，M2S在多模态文本生成领域脱颖而出。我们通过对比实验和人工评估展示了M2S的有效性和准确性。此外，该模型框架具有良好的扩展性，并具有显著的未来研究潜力。 

---
# Unveiling Uncertainty: A Deep Dive into Calibration and Performance of Multimodal Large Language Models 

**Title (ZH)**: 揭开不确定性之谜：多模态大型语言模型校准与性能深度探究 

**Authors**: Zijun Chen, Wenbo Hu, Guande He, Zhijie Deng, Zheng Zhang, Richang Hong  

**Link**: [PDF](https://arxiv.org/pdf/2412.14660)  

**Abstract**: Multimodal large language models (MLLMs) combine visual and textual data for tasks such as image captioning and visual question answering. Proper uncertainty calibration is crucial, yet challenging, for reliable use in areas like healthcare and autonomous driving. This paper investigates representative MLLMs, focusing on their calibration across various scenarios, including before and after visual fine-tuning, as well as before and after multimodal training of the base LLMs. We observed miscalibration in their performance, and at the same time, no significant differences in calibration across these scenarios. We also highlight how uncertainty differs between text and images and how their integration affects overall uncertainty. To better understand MLLMs' miscalibration and their ability to self-assess uncertainty, we construct the IDK (I don't know) dataset, which is key to evaluating how they handle unknowns. Our findings reveal that MLLMs tend to give answers rather than admit uncertainty, but this self-assessment improves with proper prompt adjustments. Finally, to calibrate MLLMs and enhance model reliability, we propose techniques such as temperature scaling and iterative prompt optimization. Our results provide insights into improving MLLMs for effective and responsible deployment in multimodal applications. Code and IDK dataset: \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 多模态大规模语言模型（MLLMs）结合视觉和文本数据，用于诸如图像字幕和视觉问答等任务。在医疗保健和自动驾驶等领域可靠使用时，适当的不确定性校准至关重要但也很具挑战性。本文探讨了代表性MLLMs在各种场景下的校准情况，包括视觉微调前后和基础大规模语言模型进行多模态训练前后的情况。我们发现它们的性能存在偏差，但在这些场景下的校准差异并不显著。此外，我们还强调了文本与图像之间的不确定性差异以及它们的整合如何影响整体不确定性。为了更好地理解MLLMs的偏差及其自我评估不确定性的能力，我们构建了IDK（我不知道）数据集，这对于评估它们处理未知问题的能力至关重要。我们的研究发现，MLLMs倾向于给出答案而不是承认不确定性，但通过适当的提示调整可以改善自我评估。最后，为了校准MLLMs并增强模型可靠性，我们提出了诸如温度调整和迭代提示优化等技术。我们的结果提供了关于如何有效且负责任地部署MLLMs以应用于多模态应用的见解。代码和IDK数据集：\href{this https URL}{此链接}。 

---
# LDP: Generalizing to Multilingual Visual Information Extraction by Language Decoupled Pretraining 

**Title (ZH)**: LDP：通过语言解耦预训练实现多语言视觉信息提取的泛化 

**Authors**: Huawen Shen, Gengluo Li, Jinwen Zhong, Yu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2412.14596)  

**Abstract**: Visual Information Extraction (VIE) plays a crucial role in the comprehension of semi-structured documents, and several pre-trained models have been developed to enhance performance. However, most of these works are monolingual (usually English). Due to the extremely unbalanced quantity and quality of pre-training corpora between English and other languages, few works can extend to non-English scenarios. In this paper, we conduct systematic experiments to show that vision and layout modality hold invariance among images with different languages. If decoupling language bias from document images, a vision-layout-based model can achieve impressive cross-lingual generalization. Accordingly, we present a simple but effective multilingual training paradigm LDP (Language Decoupled Pre-training) for better utilization of monolingual pre-training data. Our proposed model LDM (Language Decoupled Model) is first pre-trained on the language-independent data, where the language knowledge is decoupled by a diffusion model, and then the LDM is fine-tuned on the downstream languages. Extensive experiments show that the LDM outperformed all SOTA multilingual pre-trained models, and also maintains competitiveness on downstream monolingual/English benchmarks. 

**Abstract (ZH)**: 视觉信息提取（VIE）在半结构化文档的理解中扮演着至关重要的角色，且已经开发了多种预训练模型以提升性能。然而，大多数相关工作都是单一语言的（通常为英语）。由于英语和其它语言之间的预训练语料库的数量和质量存在极不平衡的情况，能扩展到非英语场景的工作寥寥无几。在本文中，我们开展了一系列系统性实验，以证明图像的语言差异并不会破坏视觉和布局模态的一贯性。如果从文档图像中去除语言偏见，基于视觉-布局的模型可以实现跨语言的卓越泛化能力。因此，我们提出了一种简单但有效的跨语言预训练范式，名为LDP（Language Decoupled Pre-training）以更充分利用单一语言的预训练数据。我们所提出的LDM（Language Decoupled Model），首先是基于语言独立的数据进行预训练，在该过程中，利用扩散模型从语言知识中解耦出来，然后在下游语言上进行微调。广泛的实验表明，LDM 在所有最先进的跨语言预训练模型中表现更优，并且在下游单一语言/英语基准测试中保持了竞争力。 

---
# Sliding Windows Are Not the End: Exploring Full Ranking with Long-Context Large Language Models 

**Title (ZH)**: 滑动窗口并非终点：探索长语境大型语言模型的全面排序 

**Authors**: Wenhan Liu, Xinyu Ma, Yutao Zhu, Ziliang Zhao, Shuaiqiang Wang, Dawei Yin, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2412.14574)  

**Abstract**: Large Language Models (LLMs) have shown exciting performance in listwise passage ranking. Due to the limited input length, existing methods often adopt the sliding window strategy. Such a strategy, though effective, is inefficient as it involves repetitive and serialized processing, which usually re-evaluates relevant passages multiple times. As a result, it incurs redundant API costs, which are proportional to the number of inference tokens. The development of long-context LLMs enables the full ranking of all passages within a single inference, avoiding redundant API costs. In this paper, we conduct a comprehensive study of long-context LLMs for ranking tasks in terms of efficiency and effectiveness. Surprisingly, our experiments reveal that full ranking with long-context LLMs can deliver superior performance in the supervised fine-tuning setting with a huge efficiency improvement. Furthermore, we identify two limitations of fine-tuning the full ranking model based on existing methods: (1) sliding window strategy fails to produce a full ranking list as a training label, and (2) the language modeling loss cannot emphasize top-ranked passage IDs in the label. To alleviate these issues, we propose a new complete listwise label construction approach and a novel importance-aware learning objective for full ranking. Experiments show the superior performance of our method over baselines. Our codes are available at \url{this https URL}. 

**Abstract (ZH)**: 大型语言模型（LLMs）在列表级段落排名任务中表现出令人兴奋的性能。由于输入长度有限，现有方法通常采用滑动窗口策略。尽管该策略有效，但效率较低，因为涉及重复和串行处理，通常会多次重新评估相关段落，导致冗余API成本，这些成本与推理令牌的数量成比例。长上下文LLMs的发展使得可以在单次推理中对所有段落进行全面排名，从而避免冗余API成本。本文从效率和效果两方面对长上下文LLMs进行了全面研究。令人惊讶的是，我们的实验表明，在监督微调设置中，使用长上下文LLMs进行全面排名可以带来显著的性能提升。此外，我们发现基于现有方法微调全面排名模型的两个限制：（1）滑动窗口策略无法生成完整的排名列表作为训练标签，（2）语言模型损失无法强调标签中排名靠前的段落ID。为了解决这些问题，我们提出了一种新的完整列表标签构建方法和一种新的注意力感知学习目标，以实现全面排名。实验结果显示，我们的方法优于基线方法。我们的代码可在 \url{此链接处} 获取。 

---
# Cal-DPO: Calibrated Direct Preference Optimization for Language Model Alignment 

**Title (ZH)**: Cal-DPO：校准的直接偏好优化语言模型对齐方法 

**Authors**: Teng Xiao, Yige Yuan, Huaisheng Zhu, Mingxiao Li, Vasant G Honavar  

**Link**: [PDF](https://arxiv.org/pdf/2412.14516)  

**Abstract**: We study the problem of aligning large language models (LLMs) with human preference data. Contrastive preference optimization has shown promising results in aligning LLMs with available preference data by optimizing the implicit reward associated with the policy. However, the contrastive objective focuses mainly on the relative values of implicit rewards associated with two responses while ignoring their actual values, resulting in suboptimal alignment with human preferences. To address this limitation, we propose calibrated direct preference optimization (Cal-DPO), a simple yet effective algorithm. We show that substantial improvement in alignment with the given preferences can be achieved simply by calibrating the implicit reward to ensure that the learned implicit rewards are comparable in scale to the ground-truth rewards. We demonstrate the theoretical advantages of Cal-DPO over existing approaches. The results of our experiments on a variety of standard benchmarks show that Cal-DPO remarkably improves off-the-shelf methods. 

**Abstract (ZH)**: 我们研究了将大型语言模型（LLMs）与人类偏好数据对齐的问题。对比偏好优化通过优化与策略相关的隐含奖励已经在对齐LLMs与可用偏好数据方面显示出有希望的结果。然而，对比目标主要关注两个响应关联的隐含奖励的相对值，而忽略了它们的实际值，导致对齐结果不理想。为了解决这一局限性，我们提出了校准直接偏好优化（Cal-DPO），这是一个简单而有效的算法。我们证明，只需校准隐含奖励，确保学习到的隐含奖励与真实奖励在量级上可比，就可以在偏好对齐方面取得显著改进。我们展示了Cal-DPO相较于现有方法的理论优势。实验结果表明，Cal-DPO在多种标准基准上的表现明显优于现有方法。 

---
# GraphEQA: Using 3D Semantic Scene Graphs for Real-time Embodied Question Answering 

**Title (ZH)**: GraphEQA：使用3D语义场景图进行实时具身问答 

**Authors**: Saumya Saxena, Blake Buchanan, Chris Paxton, Bingqing Chen, Narunas Vaskevicius, Luigi Palmieri, Jonathan Francis, Oliver Kroemer  

**Link**: [PDF](https://arxiv.org/pdf/2412.14480)  

**Abstract**: In Embodied Question Answering (EQA), agents must explore and develop a semantic understanding of an unseen environment in order to answer a situated question with confidence. This remains a challenging problem in robotics, due to the difficulties in obtaining useful semantic representations, updating these representations online, and leveraging prior world knowledge for efficient exploration and planning. Aiming to address these limitations, we propose GraphEQA, a novel approach that utilizes real-time 3D metric-semantic scene graphs (3DSGs) and task relevant images as multi-modal memory for grounding Vision-Language Models (VLMs) to perform EQA tasks in unseen environments. We employ a hierarchical planning approach that exploits the hierarchical nature of 3DSGs for structured planning and semantic-guided exploration. Through experiments in simulation on the HM-EQA dataset and in the real world in home and office environments, we demonstrate that our method outperforms key baselines by completing EQA tasks with higher success rates and fewer planning steps. 

**Abstract (ZH)**: 在具身问答（EQA）中，代理必须探索并发展对未见环境的语义理解，以便自信地回答情境性问题。这一问题在机器人学中仍然极具挑战性，原因在于获得有用语义表示、在线更新这些表示以及利用先验世界知识进行高效探索和规划的困难。为了应对这些限制，我们提出了一种名为GraphEQA的新颖方法，该方法利用实时3D度量语义场景图（3DSGs）和与任务相关的图像作为多模态记忆，将视觉语言模型（VLMs）绑定到未见环境中的EQA任务。我们采用分层规划方法，利用3DSGs的分层特性进行结构化规划和语义引导的探索。通过在HM-EQA数据集的仿真实验以及家庭和办公室环境中的实际世界实验，我们证明了该方法优于关键基线，能够以更高的成功率和更少的规划步骤完成EQA任务。 

---
# MegaPairs: Massive Data Synthesis For Universal Multimodal Retrieval 

**Title (ZH)**: MegaPairs：大规模数据合成用于通用多模态检索 

**Authors**: Junjie Zhou, Zheng Liu, Ze Liu, Shitao Xiao, Yueze Wang, Bo Zhao, Chen Jason Zhang, Defu Lian, Yongping Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2412.14475)  

**Abstract**: Despite the rapidly growing demand for multimodal retrieval, progress in this field remains severely constrained by a lack of training data. In this paper, we introduce MegaPairs, a novel data synthesis method that leverages vision language models (VLMs) and open-domain images, together with a massive synthetic dataset generated from this method. Our empirical analysis shows that MegaPairs generates high-quality data, enabling the multimodal retriever to significantly outperform the baseline model trained on 70$\times$ more data from existing datasets. Moreover, since MegaPairs solely relies on general image corpora and open-source VLMs, it can be easily scaled up, enabling continuous improvements in retrieval performance. In this stage, we produced more than 26 million training instances and trained several models of varying sizes using this data. These new models achieve state-of-the-art zero-shot performance across 4 popular composed image retrieval (CIR) benchmarks and the highest overall performance on the 36 datasets provided by MMEB. They also demonstrate notable performance improvements with additional downstream fine-tuning. Our produced dataset, well-trained models, and data synthesis pipeline will be made publicly available to facilitate the future development of this field. 

**Abstract (ZH)**: 尽管多模态检索的需求正在迅速增长，但该领域的发展仍然受到训练数据不足的严重限制。本文介绍了一种名为MegaPairs的新型数据合成方法，该方法结合了视觉语言模型（VLMs）和开放域图像，并生成了一个大规模的合成数据集。我们的实证分析表明，MegaPairs生成了高质量的数据，使多模态检索器显著优于现有数据集70倍训练数据的基线模型。此外，由于MegaPairs仅依赖于通用图像集合和开源VLMs，它易于扩展，能够不断改进检索性能。目前，我们已经生成了超过2600万个训练实例，并使用这些数据训练了多个不同规模的模型。这些新模型在4个流行的组合图像检索（CIR）基准测试中实现了最先进的零样本性能，在MMEB提供的36个数据集中也表现最优。此外，这些模型还展示了在下游微调中的显著性能提升。我们生成的数据集、训练好的模型以及数据合成管道将对外公开，以促进该领域的未来发展。 

---
# Are Longer Prompts Always Better? Prompt Selection in Large Language Models for Recommendation Systems 

**Title (ZH)**: 较长的提示总是更好的选择吗？大规模语言模型在推荐系统中的提示选择研究 

**Authors**: Genki Kusano, Kosuke Akimoto, Kunihiro Takeoka  

**Link**: [PDF](https://arxiv.org/pdf/2412.14454)  

**Abstract**: In large language models (LLM)-based recommendation systems (LLM-RSs), accurately predicting user preferences by leveraging the general knowledge of LLMs is possible without requiring extensive training data. By converting recommendation tasks into natural language inputs called prompts, LLM-RSs can efficiently solve issues that have been difficult to address due to data scarcity but are crucial in applications such as cold-start and cross-domain problems. However, when applying this in practice, selecting the prompt that matches tasks and data is essential. Although numerous prompts have been proposed in LLM-RSs and representing the target user in prompts significantly impacts recommendation accuracy, there are still no clear guidelines for selecting specific prompts.
In this paper, we categorize and analyze prompts from previous research to establish practical prompt selection guidelines. Through 450 experiments with 90 prompts and five real-world datasets, we examined the relationship between prompts and dataset characteristics in recommendation accuracy. We found that no single prompt consistently outperforms others; thus, selecting prompts on the basis of dataset characteristics is crucial. Here, we propose a prompt selection method that achieves higher accuracy with minimal validation data. Because increasing the number of prompts to explore raises costs, we also introduce a cost-efficient strategy using high-performance and cost-efficient LLMs, significantly reducing exploration costs while maintaining high prediction accuracy. Our work offers valuable insights into the prompt selection, advancing accurate and efficient LLM-RSs. 

**Abstract (ZH)**: 在基于大规模语言模型（LLM）的推荐系统（LLM-RSs）中，通过利用LLM的通用知识，可以准确预测用户偏好，而无需大量的训练数据。通过将推荐任务转化为称为提示的自然语言输入，LLM-RSs可以高效地解决由于数据稀少但又在冷启动和跨域问题等应用中至关重要的难题。然而，在实践中应用这一方法时，选择与任务和数据匹配的提示至关重要。尽管在LLM-RSs中已经提出了许多提示，且这些提示的质量显著影响推荐准确性，但仍缺乏明确的提示选择指南。

在本文中，我们对先前研究中的提示进行了分类和分析，以建立实用的提示选择指南。通过使用90个提示和五个实际数据集进行了450次实验，我们研究了提示与数据集特性在推荐准确性中的关系。我们发现没有一个提示能够始终优于其他提示；因此，根据数据集特性选择提示至关重要。在此基础上，我们提出了一种提示选择方法，该方法可以在极少量验证数据的情况下实现更高的准确性。由于增加提示的数量以进行探索会增加成本，我们还引入了一种成本效益策略，使用高性能且成本效益高的LLM，能够在显著降低探索成本的情况下保持高预测准确性。本文的工作为我们提供了关于提示选择的宝贵见解，促进了准确高效的LLM-RSs的发展。 

---
# In-Group Love, Out-Group Hate: A Framework to Measure Affective Polarization via Contentious Online Discussions 

**Title (ZH)**: 自我群体的热爱，异群体的反感：一种通过争议性在线讨论衡量情感极化的框架 

**Authors**: Buddhika Nettasinghe, Ashwin Rao, Bohan Jiang, Allon Percus, Kristina Lerman  

**Link**: [PDF](https://arxiv.org/pdf/2412.14414)  

**Abstract**: Affective polarization, the emotional divide between ideological groups marked by in-group love and out-group hate, has intensified in the United States, driving contentious issues like masking and lockdowns during the COVID-19 pandemic. Despite its societal impact, existing models of opinion change fail to account for emotional dynamics nor offer methods to quantify affective polarization robustly and in real-time. In this paper, we introduce a discrete choice model that captures decision-making within affectively polarized social networks and propose a statistical inference method estimate key parameters -- in-group love and out-group hate -- from social media data. Through empirical validation from online discussions about the COVID-19 pandemic, we demonstrate that our approach accurately captures real-world polarization dynamics and explains the rapid emergence of a partisan gap in attitudes towards masking and lockdowns. This framework allows for tracking affective polarization across contentious issues has broad implications for fostering constructive online dialogues in digital spaces. 

**Abstract (ZH)**: 情感极化是指由于群体内部的爱和群体外部的恨而产生的情感鸿沟，这种现象在美国日益加剧，导致了新冠疫情（COVID-19）期间如戴口罩和隔离措施等争议性问题的激化。尽管情感极化对社会产生了重大影响，现有的意见转变模型未能考虑情感动态，也缺乏有效方法来实时、稳健地量化情感极化。本文引入了一个离散选择模型，该模型能够捕捉情感极化社会网络中的决策过程，并提出了一种统计推断方法，用于从社交媒体数据中估计群体内部的爱和群体外部的恨等关键参数。通过新冠疫情在线讨论的实证验证，我们证明了此方法能够准确捕捉现实世界的情感极化动态，并解释了在戴口罩和隔离措施态度上的党派差异的快速出现。该框架允许在具有争议性的问题上追踪情感极化，对于促进数字空间中的建设性在线对话具有广泛意义。 

---
# ResQ: Mixed-Precision Quantization of Large Language Models with Low-Rank Residuals 

**Title (ZH)**: ResQ：具有低秩残差的大型语言模型混合精度量化 

**Authors**: Utkarsh Saxena, Sayeh Sharify, Kaushik Roy, Xin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.14363)  

**Abstract**: Post-training quantization (PTQ) of large language models (LLMs) holds the promise in reducing the prohibitive computational cost at inference time. Quantization of all weight, activation and key-value (KV) cache tensors to 4-bit without significantly degrading generalizability is challenging, due to the high quantization error caused by extreme outliers in activations. To tackle this problem, we propose ResQ, a PTQ method that pushes further the state-of-the-art. By means of principal component analysis (PCA), it identifies a low-rank subspace (in practice 1/8 of the hidden dimension) in which activation variances are highest, and keep the coefficients within this subspace in high precision, e.g. 8-bit, while quantizing the rest to 4-bit. Within each subspace, invariant random rotation is applied to further suppress outliers. We show that this is a provably optimal mixed precision quantization scheme that minimizes error. With the Llama families of models, we demonstrate that ResQ outperforms recent uniform and mixed precision PTQ methods on a variety of benchmarks, achieving up to 33% lower perplexity on Wikitext than the next best method SpinQuant, and a 2.4x speedup over 16-bit baseline. Code is available at this https URL. 

**Abstract (ZH)**: 大语言模型（LLMs）的后训练量化（PTQ）有潜力在推理时降低计算成本。将所有权重、激活和键值（KV）缓存张量量化为4比特而不显著牺牲泛化能力具有挑战性，因为激活中的极端异常值会导致量化误差很高。为了解决这个问题，我们提出了一种名为ResQ的PTQ方法，该方法进一步推进了现有最佳水平。通过主成分分析（PCA），ResQ识别出一个低秩子空间（实际上占隐藏维度的1/8），在这个子空间中激活的方差最大，并保持该子空间中的系数以较高精度（例如8比特）表示，而将其余部分量化为4比特。在每个子空间内，应用不变随机旋转以进一步抑制异常值。我们证明这种方法是能最小化误差的可验证最优混合精度量化方案。使用Llama系列模型，我们展示在多种基准测试中，ResQ优于最近的均匀量化和混合精度量化方法，与SpinQuant相比在Wikitext上的困惑度降低多达33%，并且比16比特基线提高了2.4倍的速度。相关代码可访问此链接：[提供代码链接]。 

---
# Towards AI-$45^{\circ}$ Law: A Roadmap to Trustworthy AGI 

**Title (ZH)**: 向AI-45°法则迈进：一条通往可信赖的超人工智能的道路规划 

**Authors**: Yang Chao, Lu Chaochao, Wang Yingchun, Zhou Bowen  

**Link**: [PDF](https://arxiv.org/pdf/2412.14186)  

**Abstract**: Ensuring Artificial General Intelligence (AGI) reliably avoids harmful behaviors is a critical challenge, especially for systems with high autonomy or in safety-critical domains. Despite various safety assurance proposals and extreme risk warnings, comprehensive guidelines balancing AI safety and capability remain lacking. In this position paper, we propose the \textit{AI-\textbf{$45^{\circ}$} Law} as a guiding principle for a balanced roadmap toward trustworthy AGI, and introduce the \textit{Causal Ladder of Trustworthy AGI} as a practical framework. This framework provides a systematic taxonomy and hierarchical structure for current AI capability and safety research, inspired by Judea Pearl's ``Ladder of Causation''. The Causal Ladder comprises three core layers: the Approximate Alignment Layer, the Intervenable Layer, and the Reflectable Layer. These layers address the key challenges of safety and trustworthiness in AGI and contemporary AI systems. Building upon this framework, we define five levels of trustworthy AGI: perception, reasoning, decision-making, autonomy, and collaboration trustworthiness. These levels represent distinct yet progressive aspects of trustworthy AGI. Finally, we present a series of potential governance measures to support the development of trustworthy AGI.\footnote{In this paper, trustworthiness is generally considered a broad form of safety, and no explicit distinction is made between the two. However, in some contexts, safety and trustworthiness are treated as distinct: safety involves assurance of correct behavior, while trustworthiness refers to user confidence in the system's decision-making. In such cases, different terms or both may be used depending on the context. 

**Abstract (ZH)**: 确保通用人工智能（AGI）可靠地避免有害行为是一个关键挑战，尤其是在具有高度自主性或在安全关键领域中的系统。尽管提出了各种安全保证方案和极端风险警告，但平衡AI安全性和能力的综合性指南仍然缺乏。在本文中，我们提出了\textit{AI-\textbf{$45^{\circ}$} 法律}作为通往可信赖AGI的平衡路线图的一个指导原则，并引入了\textit{可信AGI因果梯阶}作为实际框架。该框架为当前的AI能力和安全研究提供了系统的分类和层级结构，灵感来自Judea Pearl的“因果梯阶”。可信AGI因果梯阶包括三个核心层级：近似对齐层、干预层和反思层。这些层级解决了AGI和现代AI系统中安全性和可信性的一些关键挑战。基于这一框架，我们定义了可信AGI的五个层级：感知可信性、推理可信性、决策可信性、自主性可信性和协作可信性。这些层级代表了可信AGI中不同但渐进的方面。最后，我们提出了若干潜在的治理措施，以支持可信AGI的发展。\footnote{在本文中，一般认为可信性是一种广泛的平安形式，并未在安全性和可信性之间做出明确区分。然而，在某些情况下，安全性和可信性被视为不同的概念：安全性涉及正确行为的保证，而可信性指的是用户对系统决策的信心。在这种情况下，根据语境的不同，可能使用不同的术语或两者兼用。} 

---
# Whisper-GPT: A Hybrid Representation Audio Large Language Model 

**Title (ZH)**: Whisper-GPT：一种 hybrid 表征音频大语言模型 

**Authors**: Prateek Verma  

**Link**: [PDF](https://arxiv.org/pdf/2412.11449)  

**Abstract**: We propose WHISPER-GPT: A generative large language model (LLM) for speech and music that allows us to work with continuous audio representations and discrete tokens simultaneously as part of a single architecture. There has been a huge surge in generative audio, speech, and music models that utilize discrete audio tokens derived from neural compression algorithms, e.g. ENCODEC. However, one of the major drawbacks of this approach is handling the context length. It blows up for high-fidelity generative architecture if one has to account for all the audio contents at various frequencies for the next token prediction. By combining continuous audio representation like the spectrogram and discrete acoustic tokens, we retain the best of both worlds: Have all the information needed from the audio at a specific time instance in a single token, yet allow LLM to predict the future token to allow for sampling and other benefits discrete space provides. We show how our architecture improves the perplexity and negative log-likelihood scores for the next token prediction compared to a token-based LLM for speech and music. 

**Abstract (ZH)**: 我们提出了一种名为WHISPER-GPT的生成性大型语言模型（LLM），该模型能够在单一架构中同时处理连续的音频表示和离散的声学 tokens。近年来，利用神经压缩算法（如ENCODEC）生成的离散音频 tokens 的生成音频、语音和音乐模型有了显著增长。然而，这种方法的一个主要缺点是处理上下文长度。如果需要考虑到各种频率的全部音频内容以预测下一个 tokens，则会对高保真生成架构产生不良影响。通过结合连续的音频表示（如频谱图）和离散的声学 tokens，我们同时保留了两者的优点：在单个 tokens 中包含特定时间点所需的所有音频信息，同时允许LLM预测未来 tokens，从而利用离散空间提供的采样等好处。我们展示了与基于 tokens 的LLM相比，我们的架构在语音和音乐的下一个 tokens 预测中如何提高困惑度和负对数似然度评分。 

---
