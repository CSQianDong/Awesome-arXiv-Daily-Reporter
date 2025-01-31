# Thoughts Are All Over the Place: On the Underthinking of o1-Like LLMs 

**Title (ZH)**: 思绪纷繁杂乱：关于o1-类语言模型的过度简略思考 

**Authors**: Yue Wang, Qiuzhi Liu, Jiahao Xu, Tian Liang, Xingyu Chen, Zhiwei He, Linfeng Song, Dian Yu, Juntao Li, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2501.18585)  

**Abstract**: Large language models (LLMs) such as OpenAI's o1 have demonstrated remarkable abilities in complex reasoning tasks by scaling test-time compute and exhibiting human-like deep thinking. However, we identify a phenomenon we term underthinking, where o1-like LLMs frequently switch between different reasoning thoughts without sufficiently exploring promising paths to reach a correct solution. This behavior leads to inadequate depth of reasoning and decreased performance, particularly on challenging mathematical problems. To systematically analyze this issue, we conduct experiments on three challenging test sets and two representative open-source o1-like models, revealing that frequent thought switching correlates with incorrect responses. We introduce a novel metric to quantify underthinking by measuring token efficiency in incorrect answers. To address underthinking, we propose a decoding strategy with thought switching penalty TIP that discourages premature transitions between thoughts, encouraging deeper exploration of each reasoning path. Experimental results demonstrate that our approach improves accuracy across challenging datasets without requiring model fine-tuning. Our findings contribute to understanding reasoning inefficiencies in o1-like LLMs and offer a practical solution to enhance their problem-solving capabilities. 

**Abstract (ZH)**: 大型语言模型（LLMs）如OpenAI的o1在通过扩展测试时计算能力和展现类人深度思维的能力上，已经展现出了在复杂推理任务中的显著能力。然而，我们发现了一种我们称之为“欠思考”的现象，即o1类似的LLMs在解决问题时频繁在不同的推理思路之间切换，而未能充分探索有希望的路径以达到正确解。这种行为导致了推理深度不足和性能下降，尤其是在解决复杂的数学问题时。为了系统地分析这一问题，我们在三个具有挑战性的测试集中对两种代表性的开源o1类似模型进行了实验，结果显示频繁的思维切换与错误响应有关。我们引入了一个新的度量标准来量化“欠思考”，通过测量在错误答案中关键词的有效性来进行评估。为解决“欠思考”问题，我们提出了一种带有思维切换惩罚（TIP）的解码策略，该策略抑制过早的思维转换，鼓励深入探索每条推理路径。实验结果表明，我们的方法可以在无需对模型进行微调的情况下提高复杂数据集的准确性。我们的研究成果有助于理解o1类似LLMs在推理效率方面的问题，并提供一种实用的解决方案，以提升其问题解决能力。 

---
# R.I.P.: Better Models by Survival of the Fittest Prompts 

**Title (ZH)**: R.I.P.: 通过适者生存的提示生成更好的模型 

**Authors**: Ping Yu, Weizhe Yuan, Olga Golovneva, Tianhao Wu, Sainbayar Sukhbaatar, Jason Weston, Jing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2501.18578)  

**Abstract**: Training data quality is one of the most important drivers of final model quality. In this work, we introduce a method for evaluating data integrity based on the assumption that low-quality input prompts result in high variance and low quality responses. This is achieved by measuring the rejected response quality and the reward gap between the chosen and rejected preference pair. Our method, Rejecting Instruction Preferences (RIP) can be used to filter prompts from existing training sets, or to make high quality synthetic datasets, yielding large performance gains across various benchmarks compared to unfiltered data. Using Llama 3.1-8B-Instruct, RIP improves AlpacaEval2 LC Win Rate by 9.4%, Arena-Hard by 8.7%, and WildBench by 9.9%. Using Llama 3.3-70B-Instruct, RIP improves Arena-Hard from 67.5 to 82.9, which is from 18th place to 6th overall in the leaderboard. 

**Abstract (ZH)**: 训练数据质量是最终模型质量的重要驱动因素之一。在本研究中，我们提出了一种基于以下假设的方法：低质量的输入提示会导致高方差和低质量的响应。这种方法通过测量被拒绝的响应质量以及所选和被拒绝的偏好对之间的奖励差距来实现。我们提出的方法，名为拒绝指令偏好（RIP），可以用于从现有训练集中过滤提示，或用于创建高质量的合成数据集，在各种基准测试中，与未过滤的数据相比，其性能显著提高。使用Llama 3.1-8B-Instruct，RIP在AlpacaEval2 LC Win Rate 上提高了9.4%，在Arena-Hard 上提高了8.7%，在WildBench 上提高了9.9%。使用Llama 3.3-70B-Instruct，RIP将Arena-Hard 的得分从67.5提高到82.9，在排行榜上的排名从第18位上升到第6位。 

---
# Can we Retrieve Everything All at Once? ARM: An Alignment-Oriented LLM-based Retrieval Method 

**Title (ZH)**: 当然，以下是翻译后的标题和内容，符合学术规范：

标题：
一次检索所有内容可行吗？ARM：一种基于语言模型的对齐导向检索方法

内容摘要：
本研究探讨了一次性检索所有相关信息的可能性。我们提出了一种名为ARM的方法，该方法基于语言模型，并采用对齐导向的策略。 

**Authors**: Peter Baile Chen, Yi Zhang, Michael Cafarella, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2501.18539)  

**Abstract**: Real-world open-domain questions can be complicated, particularly when answering them involves information from multiple information sources. LLMs have demonstrated impressive performance in decomposing complex tasks into simpler steps, and previous work has used it for better retrieval in support of complex questions. However, LLM's decomposition of questions is unaware of what data is available and how data is organized, often leading to a sub-optimal retrieval performance. Recent effort in agentic RAG proposes to perform retrieval in an iterative fashion, where a followup query is derived as an action based on previous rounds of retrieval. While this provides one way of interacting with the data collection, agentic RAG's exploration of data is inefficient because successive queries depend on previous results rather than being guided by the organization of available data in the collection. To address this problem, we propose an LLM-based retrieval method -- ARM, that aims to better align the question with the organization of the data collection by exploring relationships among data objects beyond matching the utterance of the query, thus leading to a retrieve-all-at-once solution for complex queries. We evaluated ARM on two datasets, Bird and OTT-QA. On Bird, it outperforms standard RAG with query decomposition by up to 5.2 pt in execution accuracy and agentic RAG (ReAct) by up to 15.9 pt. On OTT-QA, it achieves up to 5.5 pt and 19.3 pt higher F1 match scores compared to these approaches. 

**Abstract (ZH)**: 现实世界的开放式问题可能非常复杂，尤其是当回答这些问题需要从多个信息源获取信息时。大规模语言模型（LLMs）在分解复杂任务为更简单步骤方面展现了出色的性能，先前的工作已经利用这一点来提高复杂问题的支持检索效果。然而，LLMs对问题的分解并不了解哪些数据可用以及数据是如何组织的，这常常导致检索性能不佳。近期关于代理型检索-生成（RAG）的努力提出了一种迭代检索的方法，其中后续查询是基于前几轮检索的结果而采取的一种行动。虽然这种方法提供了一种与数据集交互的方式，但代理型RAG对数据的探索不够高效，因为后续查询依赖于前几轮的结果而非数据集内可用数据的组织结构。为了解决这一问题，我们提出了一种基于LLM的检索方法——ARM，其目的是通过探索数据对象之间的关系，而不仅仅是匹配查询的表述，更好地将问题与数据集的组织结构对齐，从而为复杂问题提供一个一次性检索所有信息的解决方案。

我们使用两个数据集BIRD和OTT-QA对ARM进行了评估。在BIRD数据集上，ARM在执行准确性上比具有查询分解标准RAG方法提高了最多5.2个百分点，比代理型RAG（ReAct）提高了最多15.9个百分点。在OTT-QA数据集上，ARM的F1匹配得分比这些方法分别提高了最多5.5个百分点和19.3个百分点。 

---
# Differentially Private Steering for Large Language Model Alignment 

**Title (ZH)**: 大型语言模型对齐的差异隐私指引 

**Authors**: Anmol Goel, Yaxi Hu, Iryna Gurevych, Amartya Sanyal  

**Link**: [PDF](https://arxiv.org/pdf/2501.18532)  

**Abstract**: Aligning Large Language Models (LLMs) with human values and away from undesirable behaviors (such as hallucination) has become increasingly important. Recently, steering LLMs towards a desired behavior via activation editing has emerged as an effective method to mitigate harmful generations at inference-time. Activation editing modifies LLM representations by preserving information from positive demonstrations (e.g., truthful) and minimising information from negative demonstrations (e.g., hallucinations). When these demonstrations come from a private dataset, the aligned LLM may leak private information contained in those private samples. In this work, we present the first study of aligning LLM behavior with private datasets. Our work proposes the \textit{\underline{P}rivate \underline{S}teering for LLM \underline{A}lignment (PSA)} algorithm to edit LLM activations with differential privacy (DP) guarantees. We conduct extensive experiments on seven different benchmarks with open-source LLMs of different sizes (0.5B to 7B) and model families (LlaMa, Qwen, Mistral and Gemma). Our results show that PSA achieves DP guarantees for LLM alignment with minimal loss in performance, including alignment metrics, open-ended text generation quality, and general-purpose reasoning. We also develop the first Membership Inference Attack (MIA) for evaluating and auditing the empirical privacy for the problem of LLM steering via activation editing. Our attack is tailored for activation editing and relies solely on the generated texts without their associated probabilities. Our experiments support the theoretical guarantees by showing improved guarantees for our \textit{PSA} algorithm compared to several existing non-private techniques. 

**Abstract (ZH)**: 将大型语言模型（LLMs）与人类价值观对齐，并远离不良行为（如幻觉）变得越来越重要。最近，通过激活编辑引导LLMs朝着期望的行为发展，已成为在推理时减轻有害生成的有效方法。激活编辑通过保留正面示例（例如真实的）的信息并最小化负面示例（例如幻觉）的信息来修改LLM表示。当这些示例来自私人数据集时，对齐后的LLM可能会泄露私人样本中包含的私人信息。本文首次探讨了将LLM行为与私人数据集对齐的问题。我们的工作提出了\textit{\underline{P}rivate \underline{S}teering for L\underline{L}anguage \underline{M}odel \underline{A}lignment (PSA)}算法，该算法在差分隐私（DP）保证下编辑LLM激活。我们使用不同大小（0.5B到7B）和不同模型家族（LlaMa、Qwen、Mistral和Gemma）的开源LLM在七个不同的基准上进行了广泛的实验。我们的结果表明，PSA在保持性能（包括对齐指标、开放式文本生成质量和通用推理能力）几乎无损失的情况下实现了DP保证。我们还开发了第一个评估和审计通过激活编辑引导LLM问题中的实际隐私的成员推理攻击（MIA）。我们的攻击专门针对激活编辑，仅依赖于生成的文字而不需要其相关的概率。我们的实验通过显示与几种现有非隐私技术相比\textit{PSA}算法的理论保证改善，支持了理论保证。 

---
# Streaming DiLoCo with overlapping communication: Towards a Distributed Free Lunch 

**Title (ZH)**: 带有重叠通信的流式DiLoCo：迈向分布式免费午餐 

**Authors**: Arthur Douillard, Yanislav Donchev, Keith Rush, Satyen Kale, Zachary Charles, Zachary Garrett, Gabriel Teston, Dave Lacey, Ross McIlroy, Jiajun Shen, Alexandre Ramé, Arthur Szlam, Marc'Aurelio Ranzato, Paul Barham  

**Link**: [PDF](https://arxiv.org/pdf/2501.18512)  

**Abstract**: Training of large language models (LLMs) is typically distributed across a large number of accelerators to reduce training time. Since internal states and parameter gradients need to be exchanged at each and every single gradient step, all devices need to be co-located using low-latency high-bandwidth communication links to support the required high volume of exchanged bits. Recently, distributed algorithms like DiLoCo have relaxed such co-location constraint: accelerators can be grouped into ``workers'', where synchronizations between workers only occur infrequently. This in turn means that workers can afford being connected by lower bandwidth communication links without affecting learning quality. However, in these methods, communication across workers still requires the same peak bandwidth as before, as the synchronizations require all parameters to be exchanged across all workers. In this paper, we improve DiLoCo in three ways. First, we synchronize only subsets of parameters in sequence, rather than all at once, which greatly reduces peak bandwidth. Second, we allow workers to continue training while synchronizing, which decreases wall clock time. Third, we quantize the data exchanged by workers, which further reduces bandwidth across workers. By properly combining these modifications, we show experimentally that we can distribute training of billion-scale parameters and reach similar quality as before, but reducing required bandwidth by two orders of magnitude. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的训练通常分散在大量加速器上，以减少训练时间。由于在每次梯度步骤中都需要交换内部状态和参数梯度，所有设备必须使用低延迟高带宽的通信链接进行共定位，以支持所需的大量数据交换。最近，像DiLoCo这样的分布式算法放松了这种共定位的约束：加速器可以被分组为“工作者”，其中只在不频繁的同步过程中发生工作者之间的同步。这反过来意味着，工作者可以通过较低带宽的通信链接彼此连接，而不会影响学习质量。然而，在这些方法中，工作者之间的通信仍然需要相同的峰值带宽，因为同步过程需要在所有工作者之间交换所有参数。在本文中，我们通过三种方式改进了DiLoCo。首先，我们按顺序同步部分参数，而不是一次性同步所有参数，这极大地降低了峰值带宽。其次，我们允许工作者在同步期间继续训练，从而减少实际运行时间。第三，我们量化了工作者所交换的数据，进一步减少了工作者之间的带宽需求。通过适当结合这些改进，我们实验证明，我们能够分布式训练具有十亿级参数的模型，并达到与之前相似的质量，但只需原来的十分之一甚至更低的带宽需求。 

---
# CALM: Unleashing the Cross-Lingual Self-Aligning Ability of Language Model Question Answering 

**Title (ZH)**: CALM：释放语言模型问答中的跨语言自我对齐能力 

**Authors**: Yumeng Wang, Zhiyuan Fan, Qingyun Wang, May Fung, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2501.18457)  

**Abstract**: Large Language Models (LLMs) are pretrained on extensive multilingual corpora to acquire both language-specific cultural knowledge and general knowledge. Ideally, while LLMs should provide consistent responses to culture-independent questions across languages, we observe significant performance disparities. To address this, we explore the Cross-Lingual Self-Aligning ability of Language Models (CALM) to align knowledge across languages. Specifically, for a given question, we sample multiple responses across different languages, and select the most self-consistent response as the target, leaving the remaining responses as negative examples. We then employ direct preference optimization (DPO) to align the model's knowledge across different languages. Evaluations on the MEDQA and X-CSQA datasets demonstrate CALM's effectiveness in enhancing cross-lingual knowledge question answering, both in zero-shot and retrieval augmented settings. We also found that increasing the number of languages involved in CALM training leads to even higher accuracy and consistency. We offer a qualitative analysis of how cross-lingual consistency can enhance knowledge alignment and explore the method's generalizability. The source code and data of this paper are available on GitHub. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过大量多语言语料库进行预训练，以获取特定文化和通用知识。理想情况下，LLMs 应该能在不同语言中提供一致的答案以回答与文化无关的问题，但我们在实践中观察到显著的性能差异。为了解决这一问题，我们探索了语言模型的跨语言自我对齐能力（Cross-Lingual Self-Aligning，简称CALM），以实现跨语言知识的对齐。具体来说，对于给定的问题，我们会在不同的语言中采样多个响应，并选择最自洽的响应作为目标，其余响应作为负样本。然后，我们使用直接偏好优化（DPO）来跨语言对齐模型的知识。在 MEDQA 和 X-CSQA 数据集上的评估表明，CALM 在零样本和检索增强设置下都有效提升了跨语言知识问答。我们还发现，在CALM 训练过程中涉及的语言数量越多，可以获得更高的准确度和一致性。我们对跨语言一致性如何增强知识对齐进行了定性分析，并探讨了该方法的通用性。本文的源代码和数据已发布在 GitHub 上。 

---
# GENIE: Generative Note Information Extraction model for structuring EHR data 

**Title (ZH)**: GENIE：生成性笔记信息抽取模型，用于结构化电子健康记录数据 

**Authors**: Huaiyuan Ying, Hongyi Yuan, Jinsen Lu, Zitian Qu, Yang Zhao, Zhengyun Zhao, Isaac Kohane, Tianxi Cai, Sheng Yu  

**Link**: [PDF](https://arxiv.org/pdf/2501.18435)  

**Abstract**: Electronic Health Records (EHRs) hold immense potential for advancing healthcare, offering rich, longitudinal data that combines structured information with valuable insights from unstructured clinical notes. However, the unstructured nature of clinical text poses significant challenges for secondary applications. Traditional methods for structuring EHR free-text data, such as rule-based systems and multi-stage pipelines, are often limited by their time-consuming configurations and inability to adapt across clinical notes from diverse healthcare settings. Few systems provide a comprehensive attribute extraction for terminologies. While giant large language models (LLMs) like GPT-4 and LLaMA 405B excel at structuring tasks, they are slow, costly, and impractical for large-scale use. To overcome these limitations, we introduce GENIE, a Generative Note Information Extraction system that leverages LLMs to streamline the structuring of unstructured clinical text into usable data with standardized format. GENIE processes entire paragraphs in a single pass, extracting entities, assertion statuses, locations, modifiers, values, and purposes with high accuracy. Its unified, end-to-end approach simplifies workflows, reduces errors, and eliminates the need for extensive manual intervention. Using a robust data preparation pipeline and fine-tuned small scale LLMs, GENIE achieves competitive performance across multiple information extraction tasks, outperforming traditional tools like cTAKES and MetaMap and can handle extra attributes to be extracted. GENIE strongly enhances real-world applicability and scalability in healthcare systems. By open-sourcing the model and test data, we aim to encourage collaboration and drive further advancements in EHR structurization. 

**Abstract (ZH)**: 电子健康记录（EHRs）在推动医疗保健方面具有巨大的潜力，提供丰富的纵向数据，结合了结构化的信息和临床笔记中的宝贵见解。然而，临床文本的非结构化特性对二次应用造成了重大挑战。传统的EHR自由文本数据结构化方法，如基于规则的系统和多阶段流水线，通常限于耗时的配置和无法适应不同医疗保健环境中的临床笔记。很少有系统提供全面的术语提取。虽然像GPT-4和LLaMA 405B这样的大型语言模型（LLMs）在结构化任务上表现出色，但它们速度缓慢、成本高昂且不适合大规模使用。为克服这些限制，我们介绍了一种名为GENIE的生成性笔记信息提取系统，该系统利用LLMs简化无结构临床文本的结构化过程，使其形成标准格式的可使用数据。GENIE一次处理整个段落，提取实体、断言状态、位置、修饰语、值和目的，并具有很高的准确性。其统一的端到端方法简化了工作流程，减少了错误，并消除了对大量人工干预的需要。通过使用稳健的数据准备流水线和微调的小规模LLMs，GENIE在多个信息提取任务上取得了竞争力的表现，超越了传统的工具如cTAKES和MetaMap，并能够处理需要额外提取的属性。GENIE大大增强了在医疗系统中的实际适用性和可扩展性。通过开源模型和测试数据，我们旨在促进合作并推动EHR结构化方面的进一步发展。 

---
# RbFT: Robust Fine-tuning for Retrieval-Augmented Generation against Retrieval Defects 

**Title (ZH)**: RbFT：对抗检索缺陷的鲁棒微调以增强生成检索方法 

**Authors**: Yiteng Tu, Weihang Su, Yujia Zhou, Yiqun Liu, Qingyao Ai  

**Link**: [PDF](https://arxiv.org/pdf/2501.18365)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models (LLMs) by integrating external knowledge retrieved from a knowledge base. However, its effectiveness is fundamentally constrained by the reliability of both the retriever and the knowledge base. In real-world scenarios, imperfections in these components often lead to the retrieval of noisy, irrelevant, or misleading counterfactual information, ultimately undermining the trustworthiness of RAG systems. To address this challenge, we propose Robust Fine-Tuning (RbFT), a method designed to enhance the resilience of LLMs against retrieval defects through two targeted fine-tuning tasks. Experimental results demonstrate that RbFT significantly improves the robustness of RAG systems across diverse retrieval conditions, surpassing existing methods while maintaining high inference efficiency and compatibility with other robustness techniques. 

**Abstract (ZH)**: 检索增强生成（RAG）通过将知识库中检索到的外部知识整合到大型语言模型（LLMs）中来增强其性能。然而，其效果从根本上受到检索器和知识库可靠性的限制。在实际场景中，这些组件的缺陷往往会导致检索到嘈杂的、不相关信息或误导性的反事实信息，从而削弱RAG系统的可信度。为了解决这一挑战，我们提出了一种名为鲁棒微调（RbFT）的方法，该方法通过两个针对特定问题的微调任务来增强LLMs对检索缺陷的鲁棒性。实验结果表明，RbFT显著提高了RAG系统在各种检索条件下的鲁棒性，超越了现有方法，同时保持了高效性和与其他鲁棒性技术的良好兼容性。 

---
# Mining for Species, Locations, Habitats, and Ecosystems from Scientific Papers in Invasion Biology: A Large-Scale Exploratory Study with Large Language Models 

**Title (ZH)**: 从入侵生物学的科学论文中挖掘物种、地理位置、栖息地和生态系统：一种大规模探索性研究，利用大型语言模型 

**Authors**: Jennifer D'Souza, Zachary Laubach, Tarek Al Mustafa, Sina Zarrieß, Robert Frühstückl, Phyllis Illari  

**Link**: [PDF](https://arxiv.org/pdf/2501.18287)  

**Abstract**: This paper presents an exploratory study that harnesses the capabilities of large language models (LLMs) to mine key ecological entities from invasion biology literature. Specifically, we focus on extracting species names, their locations, associated habitats, and ecosystems, information that is critical for understanding species spread, predicting future invasions, and informing conservation efforts. Traditional text mining approaches often struggle with the complexity of ecological terminology and the subtle linguistic patterns found in these texts. By applying general-purpose LLMs without domain-specific fine-tuning, we uncover both the promise and limitations of using these models for ecological entity extraction. In doing so, this study lays the groundwork for more advanced, automated knowledge extraction tools that can aid researchers and practitioners in understanding and managing biological invasions. 

**Abstract (ZH)**: 本文呈现了一项探索性研究，利用大型语言模型（LLM）的能力从入侵生物学文献中挖掘关键生态实体。具体而言，我们关注从物种名称、其分布地点、相关的栖息地和生态系统中提取信息，这些信息对于理解物种扩散、预测未来入侵事件以及指导保护工作至关重要。传统文本挖掘方法常常难以应对生态术语的复杂性及其文本中的微妙语言模式。通过应用一般用途的LLM而无需特定领域的微调，我们揭示了这些模型在生态实体提取中的潜力和局限性。通过这项研究，为开发更高级的自动化知识提取工具奠定了基础，这些工具可以帮助研究人员和实践者理解和管理生物入侵。 

---
# Jailbreaking LLMs' Safeguard with Universal Magic Words for Text Embedding Models 

**Title (ZH)**: 使用通用魔法词突破LLMs的安全防护以供文本嵌入模型使用 

**Authors**: Haoyu Liang, Youran Sun, Yunfeng Cai, Jun Zhu, Bo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.18280)  

**Abstract**: The security issue of large language models (LLMs) has gained significant attention recently, with various defense mechanisms developed to prevent harmful outputs, among which safeguards based on text embedding models serve as a fundamental defense. Through testing, we discover that the distribution of text embedding model outputs is significantly biased with a large mean. Inspired by this observation, we propose novel efficient methods to search for universal magic words that can attack text embedding models. The universal magic words as suffixes can move the embedding of any text towards the bias direction, therefore manipulate the similarity of any text pair and mislead safeguards. By appending magic words to user prompts and requiring LLMs to end answers with magic words, attackers can jailbreak the safeguard. To eradicate this security risk, we also propose defense mechanisms against such attacks, which can correct the biased distribution of text embeddings in a train-free manner. 

**Abstract (ZH)**: 大型语言模型（LLMs）的安全问题最近引起了广泛关注，开发出了各种防御机制以防止产生有害输出，其中基于文本嵌入模型的安全保障成为基础性的防御手段。通过测试，我们发现文本嵌入模型的输出分布存在显著偏差，且具有较大的均值。基于这一观察，我们提出了新的高效方法来寻找通用的“魔法词”，这些“魔法词”能够攻击文本嵌入模型。作为后缀的“魔法词”可以将任何文本的嵌入向偏差方向移动，从而操控任何文本对之间的相似度，并误导安全保障。通过在用户提示中附加“魔法词”，并在要求LLMs以“魔法词”结尾的条件下生成答案，攻击者可以突破安全保障。为消除这一安全风险，我们还提出了一种无需训练的防御机制，该机制能够以无监督的方式纠正文本嵌入的偏差分布。 

---
# How to Select Datapoints for Efficient Human Evaluation of NLG Models? 

**Title (ZH)**: 如何选择数据点以提高NLG模型的人工评估效率？ 

**Authors**: Vilém Zouhar, Peng Cui, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2501.18251)  

**Abstract**: Human evaluation is the gold-standard for evaluating text generation models. It is also expensive, and to fit budgetary constraints, a random subset of the test data is often chosen in practice. The randomly selected data may not accurately represent test performance, making this approach economically inefficient for model comparison. Thus, in this work, we develop a suite of selectors to get the most informative datapoints for human evaluation while taking the evaluation costs into account. We show that selectors based on variance in automated metric scores, diversity in model outputs, or Item Response Theory outperform random selection. We further develop an approach to distill these selectors to the scenario where the model outputs are not yet available. In particular, we introduce source-based estimators, which predict item usefulness for human evaluation just based on the source texts. We demonstrate the efficacy of our selectors in two common NLG tasks, machine translation and summarization, and show that up to only ~50% of the test data is needed to produce the same evaluation result as the entire data. Our implementations are published in the subset2evaluate package. 

**Abstract (ZH)**: 人类评估是评价文本生成模型的黄金标准。然而，这种方式成本高昂，为了符合预算限制，实践中通常会选择测试数据的一个随机子集进行评估。随机选取的数据可能无法准确反映模型的整体表现，使得这种方法在模型对比方面经济效率低下。因此，在本文中，我们开发了一整套选择器，以在考虑评估成本的同时获得最具信息量的数据点用于人工评估。我们证明了基于自动化评估指标的方差、模型输出的多样性或项目反应理论的选择器在性能上优于随机选择。我们还进一步提出了一种方法，用于在模型输出尚未可用的情况下进行这些选择器的提炼。具体而言，我们引入了基于源文本的估计器，这些估计器仅根据源文本预测项目对人工评估的有用性。我们展示了我们在机器翻译和总结这两项常见自然语言生成任务中的选择器的有效性，并证明仅需测试数据的大约50%即可达到与使用完整数据集相同的评估结果。我们的实现已发布在subset2evaluate包中。 

---
# Contextually Structured Token Dependency Encoding for Large Language Models 

**Title (ZH)**: 面向上下文的结构化词元依赖编码在大规模语言模型中的应用 

**Authors**: James Blades, Frederick Somerfield, William Langley, Susan Everingham, Maurice Witherington  

**Link**: [PDF](https://arxiv.org/pdf/2501.18205)  

**Abstract**: Token representation strategies within large-scale neural architectures often rely on contextually refined embeddings, yet conventional approaches seldom encode structured relationships explicitly within token interactions. Self-attention mechanisms effectively capture dynamic contextual dependencies, but their reliance on learned weight distributions limits the preservation of long-range hierarchical structures in generated sequences. Dependency-aware token encoding introduces a structured approach to embedding initialization, ensuring that relational constraints are embedded within token representations rather than inferred solely through attention dynamics. The proposed encoding mechanism refines token interactions through dependency-weighted attention computations, ensuring that syntactic and semantic dependencies are retained across multiple processing layers. Empirical evaluations indicate reductions in perplexity across diverse linguistic benchmarks, suggesting improvements in contextual coherence and predictive consistency in autoregressive text generation. Computational efficiency assessments reveal a moderate increase in memory consumption and training time, attributed to additional matrix computations within the encoding module, yet scalability remains feasible within conventional transformer architectures. Structured encoding enhances lexical variation and dependency retention, reinforcing linguistic coherence without requiring external syntactic annotations or auxiliary training objectives. Statistical comparisons highlight improvements in dependency alignment, particularly in longer sequences where conventional self-attention models exhibit degradation in hierarchical consistency. Sentence length distributions indicate a reduction in abrupt phrase transitions, further supporting the hypothesis that explicit dependency encoding facilitates more structured phrase generation. 

**Abstract (ZH)**: 大规模神经架构中的令牌表示策略通常依赖于上下文精炼的嵌入表示，但传统方法很少在令牌交互中显式地编码结构化的关系。自我注意机制能够有效捕获动态上下文依赖关系，但它们对学习权重分布的高度依赖限制了在生成序列中长距离层次结构结构的保留。依赖关系感知的令牌编码引入了一种结构化的嵌入初始化方法，确保关系约束在令牌表示中嵌入，而不仅仅是通过注意动态推断。所提出的编码机制通过依赖加权注意力计算细化令牌交互，确保在多个处理层中保留语法和语义依赖关系。实证评估表明，在多种语言基准上困惑度的降低，这表明在自回归文本生成中上下文连贯性和预测一致性有所提高。计算效率评估显示内存消耗和训练时间有适度增加，这归因于编码模块内的额外矩阵计算，但结构化编码在传统变压器架构中的可扩展性仍然可行。结构化编码增强了词汇变异性和依赖关系保留，无需外部句法注释或辅助训练目标，即可增强语言连贯性。统计比较突显了在较长序列中的依赖对齐改进，特别是在传统自我注意模型表现退化的场合。句子长度分布表明语句不自然转变的减少，进一步支持显式依赖编码有助于更结构化短语生成的假设。 

---
# Mixed-Precision Graph Neural Quantization for Low Bit Large Language Models 

**Title (ZH)**: 混合精度图神经量化在低比特大规模语言模型中的应用 

**Authors**: Wanlong Liu, Yichen Xiao, Dingyi Zeng, Hongyang Zhao, Wenyu Chen, Malu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.18154)  

**Abstract**: Post-Training Quantization (PTQ) is pivotal for deploying large language models (LLMs) within resource-limited settings by significantly reducing resource demands. However, existing PTQ strategies underperform at low bit levels < 3 bits due to the significant difference between the quantized and original weights. To enhance the quantization performance at low bit widths, we introduce a Mixed-precision Graph Neural PTQ (MG-PTQ) approach, employing a graph neural network (GNN) module to capture dependencies among weights and adaptively assign quantization bit-widths. Through the information propagation of the GNN module, our method more effectively captures dependencies among target weights, leading to a more accurate assessment of weight importance and optimized allocation of quantization strategies. Extensive experiments on the WikiText2 and C4 datasets demonstrate that our MG-PTQ method outperforms previous state-of-the-art PTQ method GPTQ, setting new benchmarks for quantization performance under low-bit conditions. 

**Abstract (ZH)**: 后训练量化（PTQ）对于在资源受限环境中部署大型语言模型（LLMs）至关重要，它能显著降低资源需求。然而，现有的PTQ策略在位数低于3位时表现不佳，主要原因在于量化权重与原始权重之间的显著差异。为了提升在低位宽条件下的量化性能，我们提出了一种混合精度图神经网络PTQ（MG-PTQ）方法，通过引入图神经网络（GNN）模块来捕捉权重之间的依赖关系，并自适应地分配量化位宽。借助GNN模块的信息传播，我们的方法更有效地捕捉目标权重之间的依赖关系，从而更准确地评估权重的重要性并优化量化策略的选择。在WikiText2和C4数据集上的广泛实验表明，我们的MG-PTQ方法在低位宽条件下超越了之前的最先进的PTQ方法GPTQ，建立了新的量化性能基准。 

---
# Unraveling the Capabilities of Language Models in News Summarization 

**Title (ZH)**: 探究语言模型在新闻摘要生成中的能力 

**Authors**: Abdurrahman Odabaşı, Göksel Biricik  

**Link**: [PDF](https://arxiv.org/pdf/2501.18128)  

**Abstract**: Given the recent introduction of multiple language models and the ongoing demand for improved Natural Language Processing tasks, particularly summarization, this work provides a comprehensive benchmarking of 20 recent language models, focusing on smaller ones for the news summarization task. In this work, we systematically test the capabilities and effectiveness of these models in summarizing news article texts which are written in different styles and presented in three distinct datasets. Specifically, we focus in this study on zero-shot and few-shot learning settings and we apply a robust evaluation methodology that combines different evaluation concepts including automatic metrics, human evaluation, and LLM-as-a-judge. Interestingly, including demonstration examples in the few-shot learning setting did not enhance models' performance and, in some cases, even led to worse quality of the generated summaries. This issue arises mainly due to the poor quality of the gold summaries that have been used as reference summaries, which negatively impacts the models' performance. Furthermore, our study's results highlight the exceptional performance of GPT-3.5-Turbo and GPT-4, which generally dominate due to their advanced capabilities. However, among the public models evaluated, certain models such as Qwen1.5-7B, SOLAR-10.7B-Instruct-v1.0, Meta-Llama-3-8B and Zephyr-7B-Beta demonstrated promising results. These models showed significant potential, positioning them as competitive alternatives to large models for the task of news summarization. 

**Abstract (ZH)**: 近年来，多种语言模型的推出和对改进自然语言处理任务（尤其是摘要任务）的持续需求，促使我们对20个最近的语言模型进行了全面的基准测试，特别是在新闻摘要任务中侧重于较小规模的语言模型。在这项研究中，我们系统地测试了这些模型在不同风格的新闻文章文本上的总结能力，并在三个不同的数据集上进行了实验。具体而言，我们在这项研究中重点关注零样本学习和少样本学习场景，并采用了一种稳健的评估方法，结合了多种评估概念，包括自动评估指标、人工评估和LLM作为法官的评估。

有趣的是，在少样本学习场景中包括示范示例并没有提升模型的性能，在某些情况下甚至导致生成的摘要质量更差。这一问题主要是由于作为参考摘要使用的“金标准”摘要质量低下所致，这直接影响了模型的性能。此外，我们研究的结果还突显了GPT-3.5-Turbo和GPT-4表现出色，因为它们通常由于高级功能而在性能上占据主导地位。然而，在评估的公共模型中，某些模型，如Qwen1.5-7B、SOLAR-10.7B-Instruct-v1.0、Meta-Llama-3-8B和Zephyr-7B-Beta也展现出令人鼓舞的结果。这些模型表现出了显著的潜力，使它们成为大模型在新闻摘要任务上的有力竞争者。 

---
# Self-supervised Quantized Representation for Seamlessly Integrating Knowledge Graphs with Large Language Models 

**Title (ZH)**: 自我监督量化表示以无缝整合知识图谱与大规模语言模型 

**Authors**: Qika Lin, Tianzhe Zhao, Kai He, Zhen Peng, Fangzhi Xu, Ling Huang, Jingying Ma, Mengling Feng  

**Link**: [PDF](https://arxiv.org/pdf/2501.18119)  

**Abstract**: Due to the presence of the natural gap between Knowledge Graph (KG) structures and the natural language, the effective integration of holistic structural information of KGs with Large Language Models (LLMs) has emerged as a significant question. To this end, we propose a two-stage framework to learn and apply quantized codes for each entity, aiming for the seamless integration of KGs with LLMs. Firstly, a self-supervised quantized representation (SSQR) method is proposed to compress both KG structural and semantic knowledge into discrete codes (\ie, tokens) that align the format of language sentences. We further design KG instruction-following data by viewing these learned codes as features to directly input to LLMs, thereby achieving seamless integration. The experiment results demonstrate that SSQR outperforms existing unsupervised quantized methods, producing more distinguishable codes. Further, the fine-tuned LLaMA2 and LLaMA3.1 also have superior performance on KG link prediction and triple classification tasks, utilizing only 16 tokens per entity instead of thousands in conventional prompting methods. 

**Abstract (ZH)**: 由于知识图谱（KG）结构与自然语言之间存在天然差距，如何有效地将KG的全面结构信息与大型语言模型（LLMs）结合起来成为一个重要的问题。为了解决这一问题，我们提出了一种两阶段框架，旨在学习和应用每个实体的量化代码，实现KG与LLMs的无缝集成。首先，我们提出了自监督量化表示（SSQR）方法，将KG的结构知识和语义知识压缩成离散的代码（即，标记），以匹配语言句子的格式。在此基础上，我们进一步设计了KG指示遵循数据，将这些学习到的代码作为特征直接输入到LLMs中，从而实现无缝集成。实验结果表明，SSQR方法在现有无监督量化方法中表现更优，生成了更为可区分的代码。此外，微调的LLaMA2和LLaMA3.1在KG链接预测和三元组分类任务中也表现出色，仅使用每实体16个标记，而传统的提示方法中则需要成千上万的标记。 

---
# Diverse Preference Optimization 

**Title (ZH)**: 多样化的偏好优化 

**Authors**: Jack Lanchantin, Angelica Chen, Shehzaad Dhuliawala, Ping Yu, Jason Weston, Sainbayar Sukhbaatar, Ilia Kulikov  

**Link**: [PDF](https://arxiv.org/pdf/2501.18101)  

**Abstract**: Post-training of language models, either through reinforcement learning, preference optimization or supervised finetuning, tends to sharpen the output probability distribution and reduce the diversity of generated responses. This is particularly a problem for creative generative tasks where varied responses are desired. %This impacts the ability to generate high quality synthetic data which is becoming a vital component of model training. In this work we introduce Diverse Preference Optimization (DivPO), an online optimization method which learns to generate much more diverse responses than standard pipelines, while maintaining the quality of the generations. In DivPO, preference pairs are selected by first considering a pool of responses, and a measure of diversity among them, and selecting chosen examples as being more rare but high quality, while rejected examples are more common, but low quality. DivPO results in generating 45.6% more diverse persona attributes, and an 74.6% increase in story diversity, while maintaining similar win rates as standard baselines. 

**Abstract (ZH)**: 通过强化学习、偏好优化或监督微调等方法对语言模型进行后训练，往往会使输出的概率分布更加尖锐，减少生成响应的多样性。这一点在需要多种多样响应的创造性生成任务中尤为成问题。这影响了生成高质量合成数据的能力，而高质量合成数据正成为模型训练的重要组成部分。在这项工作中，我们提出了一种在线优化方法——多样偏好优化（DivPO），该方法能够在保持生成质量的同时，生成比标准流水线更为多样的响应。在DivPO中，首先考虑一个响应池及其多样性度量，选择更为罕见但高质量的示例作为首选，而选择更为普通但质量较低的示例作为次选。通过DivPO，生成的个性特征多样性增加了45.6%，故事多样性增加了74.6%，同时保持与标准基线相似的胜出率。 

---
# Panacea: Mitigating Harmful Fine-tuning for Large Language Models via Post-fine-tuning Perturbation 

**Title (ZH)**: 帕纳塞斯：通过后微调扰动减轻大型语言模型有害微调的影响 

**Authors**: Yibo Wang, Tiansheng Huang, Li Shen, Huanjin Yao, Haotian Luo, Rui Liu, Naiqiang Tan, Jiaxing Huang, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2501.18100)  

**Abstract**: Harmful fine-tuning attack introduces significant security risks to the fine-tuning services. Mainstream defenses aim to vaccinate the model such that the later harmful fine-tuning attack is less effective. However, our evaluation results show that such defenses are fragile -- with a few fine-tuning steps, the model still can learn the harmful knowledge. To this end, we do further experiment and find that an embarrassingly simple solution -- adding purely random perturbations to the fine-tuned model, can recover the model from harmful behavior, though it leads to a degradation in the model's fine-tuning performance. To address the degradation of fine-tuning performance, we further propose Panacea, which optimizes an adaptive perturbation that will be applied to the model after fine-tuning. Panacea maintains model's safety alignment performance without compromising downstream fine-tuning performance. Comprehensive experiments are conducted on different harmful ratios, fine-tuning tasks and mainstream LLMs, where the average harmful scores are reduced by up-to 21.5%, while maintaining fine-tuning performance. As a by-product, we analyze the optimized perturbation and show that different layers in various LLMs have distinct safety coefficients. Source code available at this https URL 

**Abstract (ZH)**: 有害微调攻击会给微调服务带来重要的安全风险。主流的防御措施旨在通过“接种”模型来减少后续有害微调攻击的有效性。然而，我们的评估结果表明，这些防御措施是脆弱的——即使进行少量的微调步骤，模型仍然可以学习到有害知识。为此，我们进一步进行了实验，并发现一个极其简单的解决方案——向微调后的模型添加纯粹的随机扰动，可以恢复模型的正常行为，尽管这会导致模型微调性能的下降。为了应对微调性能的下降，我们进一步提出了Panacea，该方法优化了一种适应性的扰动，在微调后应用于模型，以保持模型的安全对齐性能而不牺牲下游微调性能。我们在不同有害比例、微调任务和主流的大规模语言模型（LLMs）上进行了全面的实验，在保持微调性能的同时，将平均有害评分降低了高达21.5%。作为副产品，我们分析了优化后的扰动，并展示了不同LLM的各个层具有不同的安全系数。源代码可在以下链接获取：this https URL 

---
# InnerThoughts: Disentangling Representations and Predictions in Large Language Models 

**Title (ZH)**: 《内在意图：分离大语言模型中的表示和预测》

这个标题翻译成中文时，为了更符合中文的表达习惯和学术规范，可以稍微调整为：

《内在意图：大型语言模型中表示与预测的分离》 

**Authors**: Didier Chételat, Joseph Cotnareanu, Rylee Thompson, Yingxue Zhang, Mark Coates  

**Link**: [PDF](https://arxiv.org/pdf/2501.17994)  

**Abstract**: Large language models (LLMs) contain substantial factual knowledge which is commonly elicited by multiple-choice question-answering prompts. Internally, such models process the prompt through multiple transformer layers, building varying representations of the problem within its hidden states. Ultimately, however, only the hidden state corresponding to the final layer and token position are used to predict the answer label. In this work, we propose instead to learn a small separate neural network predictor module on a collection of training questions, that take the hidden states from all the layers at the last temporal position as input and outputs predictions. In effect, such a framework disentangles the representational abilities of LLMs from their predictive abilities. On a collection of hard benchmarks, our method achieves considerable improvements in performance, sometimes comparable to supervised fine-tuning procedures, but at a fraction of the computational cost. 

**Abstract (ZH)**: 大语言模型（LLMs）包含大量的事实性知识，这些知识通常通过多项选择题式的问答提示来提取。内部而言，这些模型通过多层变换器处理提示，构建问题在隐藏状态中的不同表示。然而，最终用于预测答案标签的仅是最后一个层和标记位置的隐藏状态。在本项工作中，我们提出了一种替代方法，即在一组训练问题上学习一个小的独立神经网络预测模块，该模块接受所有层在最后一个时间位置的隐藏状态作为输入，并输出预测结果。实际上，这种框架将LLMs的表示能力与其预测能力分离。在一组具有挑战性的基准测试中，我们的方法在性能上取得了显著的提升，有时与监督微调程序相当，但计算成本却低得多。 

---
# Docling: An Efficient Open-Source Toolkit for AI-driven Document Conversion 

**Title (ZH)**: Docling：一种高效的开源工具包，用于AI驱动的文档转换 

**Authors**: Nikolaos Livathinos, Christoph Auer, Maksym Lysak, Ahmed Nassar, Michele Dolfi, Panos Vagenas, Cesar Berrospi Ramis, Matteo Omenetti, Kasper Dinkla, Yusik Kim, Shubham Gupta, Rafael Teixeira de Lima, Valery Weber, Lucas Morin, Ingmar Meijer, Viktor Kuropiatnyk, Peter W. J. Staar  

**Link**: [PDF](https://arxiv.org/pdf/2501.17887)  

**Abstract**: We introduce Docling, an easy-to-use, self-contained, MIT-licensed, open-source toolkit for document conversion, that can parse several types of popular document formats into a unified, richly structured representation. It is powered by state-of-the-art specialized AI models for layout analysis (DocLayNet) and table structure recognition (TableFormer), and runs efficiently on commodity hardware in a small resource budget. Docling is released as a Python package and can be used as a Python API or as a CLI tool. Docling's modular architecture and efficient document representation make it easy to implement extensions, new features, models, and customizations. Docling has been already integrated in other popular open-source frameworks (e.g., LangChain, LlamaIndex, spaCy), making it a natural fit for the processing of documents and the development of high-end applications. The open-source community has fully engaged in using, promoting, and developing for Docling, which gathered 10k stars on GitHub in less than a month and was reported as the No. 1 trending repository in GitHub worldwide in November 2024. 

**Abstract (ZH)**: 我们介绍了一个名为Docling的工具包，这是一个易于使用、自包含、采用MIT许可证的开源文档转换工具。它能够将多种流行的文档格式解析为统一的、结构丰富的表示形式。Docling由最先进的专用AI模型（如DocLayNet用于布局分析、TableFormer用于表结构识别）驱动，并且能够在低成本的计算资源下高效运行于通用硬件上。Docling作为一个Python包发行，并可以作为Python API或CLI工具使用。Docling的模块化架构和高效的文档表示使其易于实现扩展、新功能、模型和自定义。Docling已被集成到其他流行的开源框架（例如LangChain、LlamaIndex、spaCy）中，使其成为处理文档和开发高端应用程序的自然选择。开源社区已经积极参与使用、推广和为Docling开发，该工具包在GitHub上线不到一个月便获得了10k颗星，并在2024年11月成为GitHub上最热门的仓库之一。 

---
# Rethinking Bottlenecks in Safety Fine-Tuning of Vision Language Models 

**Title (ZH)**: 重新思考视觉语言模型安全性微调中的瓶颈问题 

**Authors**: Yi Ding, Lijun Li, Bing Cao, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2501.18533)  

**Abstract**: Large Vision-Language Models (VLMs) have achieved remarkable performance across a wide range of tasks. However, their deployment in safety-critical domains poses significant challenges. Existing safety fine-tuning methods, which focus on textual or multimodal content, fall short in addressing challenging cases or disrupt the balance between helpfulness and harmlessness. Our evaluation highlights a safety reasoning gap: these methods lack safety visual reasoning ability, leading to such bottlenecks. To address this limitation and enhance both visual perception and reasoning in safety-critical contexts, we propose a novel dataset that integrates multi-image inputs with safety Chain-of-Thought (CoT) labels as fine-grained reasoning logic to improve model performance. Specifically, we introduce the Multi-Image Safety (MIS) dataset, an instruction-following dataset tailored for multi-image safety scenarios, consisting of training and test splits. Our experiments demonstrate that fine-tuning InternVL2.5-8B with MIS significantly outperforms both powerful open-source models and API-based models in challenging multi-image tasks requiring safety-related visual reasoning. This approach not only delivers exceptional safety performance but also preserves general capabilities without any trade-offs. Specifically, fine-tuning with MIS increases average accuracy by 0.83% across five general benchmarks and reduces the Attack Success Rate (ASR) on multiple safety benchmarks by a large margin. Data and Models are released under: \href{this https URL}{\texttt{this https URL}} 

**Abstract (ZH)**: 大型视觉-语言模型（VLMs）在众多任务中取得了显著的性能。然而，在安全性关键领域中的部署面临重大挑战。现有的安全微调方法集中在文本或多模态内容上，无法有效应对复杂情况或在有用性和无害性之间保持平衡。我们的评估揭示了一个安全推理的缺口：这些方法缺乏表观安全推理的能力，导致了这一瓶颈。为了解决这一局限性并增强安全性关键场景下的视觉感知和推理能力，我们提出了一种新数据集，该数据集将多张图片输入与安全推理链（CoT）标签集成，以提高模型性能。具体而言，我们引入了多张图片安全（MIS）数据集，这是一个专门针对多张图片安全场景的指令遵循数据集，包含训练集和测试集。实验结果表明，使用MIS微调InternVL2.5-8B模型在复杂的多张图片任务中要求在与安全性相关的视觉推理方面，显著优于强大的开源模型和基于API的模型。这种方法不仅提供了卓越的安全性能，还保留了一般功能而没有任何交易。具体而言，使用MIS进行微调使五个通用基准的平均准确率提高了0.83%，并大幅降低了多个安全性基准上的攻击成功率（ASR）。数据和模型已在此处发布：\href{this https URL}{this https URL} 

---
# WILDCHAT-50M: A Deep Dive Into the Role of Synthetic Data in Post-Training 

**Title (ZH)**: WILDCHAT-50M：合成数据在后训练中的作用探究 

**Authors**: Benjamin Feuer, Chinmay Hegde  

**Link**: [PDF](https://arxiv.org/pdf/2501.18511)  

**Abstract**: Language model (LLM) post-training, from DPO to distillation, can refine behaviors and unlock new skills, but the open science supporting these post-training techniques is still in its infancy. One limiting factor has been the difficulty of conducting large-scale comparative analyses of synthetic data generating models and LLM judges. To close this gap, we introduce WILDCHAT-50M, the largest public chat dataset to date. We extend the existing WildChat dataset to include responses not only from GPT, but from over 50 different open-weight models, ranging in size from 0.5B to 104B parameters. We conduct an extensive comparative analysis and demonstrate the potential of this dataset by creating RE-WILD, our own public SFT mix, which outperforms the recent Tulu-3 SFT mixture from Allen AI with only 40% as many samples. Our dataset, samples and code are available at this https URL. 

**Abstract (ZH)**: 经过训练的语言模型（LLM）从DPO到蒸馏技术可用于细化行为并解锁新技能，但当前对这些后训练技术的支持仍处于初级阶段。其中一个限制因素是难以进行大规模合成数据生成模型和LLM评判器的比较分析。为了解决这一问题，我们引入了WILDCHAT-50M，这是迄今为止最大的公共聊天数据集。我们扩展了现有的WildChat数据集，不仅包括GPT的响应，还包括超过50个不同参数量的开源权重模型的响应，这些模型的参数量从5亿到1040亿不等。我们进行了广泛比较分析，并通过创建RE-WILD，我们自己的公共SFT混合数据集，展示了该数据集的潜力。RE-WILD在样本数量仅为Allen AI最近的Tulu-3 SFT混合数据集40%的情况下，表现更优。我们的数据集、样本和代码可以从以下网址获得：this https URL。 

---
# MedXpertQA: Benchmarking Expert-Level Medical Reasoning and Understanding 

**Title (ZH)**: MedXpertQA：专家级医学推理与理解的标准基准 

**Authors**: Yuxin Zuo, Shang Qu, Yifei Li, Zhangren Chen, Xuekai Zhu, Ermo Hua, Kaiyan Zhang, Ning Ding, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2501.18362)  

**Abstract**: We introduce MedXpertQA, a highly challenging and comprehensive benchmark to evaluate expert-level medical knowledge and advanced reasoning. MedXpertQA includes 4,460 questions spanning 17 specialties and 11 body systems. It includes two subsets, Text for text evaluation and MM for multimodal evaluation. Notably, MM introduces expert-level exam questions with diverse images and rich clinical information, including patient records and examination results, setting it apart from traditional medical multimodal benchmarks with simple QA pairs generated from image captions. MedXpertQA applies rigorous filtering and augmentation to address the insufficient difficulty of existing benchmarks like MedQA, and incorporates specialty board questions to improve clinical relevance and comprehensiveness. We perform data synthesis to mitigate data leakage risk and conduct multiple rounds of expert reviews to ensure accuracy and reliability. We evaluate 16 leading models on MedXpertQA. Moreover, medicine is deeply connected to real-world decision-making, providing a rich and representative setting for assessing reasoning abilities beyond mathematics and code. To this end, we develop a reasoning-oriented subset to facilitate the assessment of o1-like models. 

**Abstract (ZH)**: 我们将引入MedXpertQA，这是一个极具挑战性和全面性的基准测试，用于评估专业知识水平的医学知识和高级推理能力。MedXpertQA 包含4,460个问题，涵盖了17个专科和11个身体系统。它包括两个子集：Text用于文本评估，MM用于多模态评估。值得注意的是，MM引入了包含多样化图像和丰富临床信息（包括患者记录和检查结果）的专家级考题，这使其不同于传统简单基于图像描述生成问答对的医学多模态基准测试。MedXpertQA 通过严格的筛选和增强处理，解决了现有基准如MedQA 因问题难度不足而导致的问题，并引入了专科考试题目，以提高临床相关性和全面性。我们进行了数据合成以降低数据泄露的风险，并进行了多轮专家评审以确保准确性和可靠性。我们在MedXpertQA 上评估了16个领先的模型。此外，医学与现实生活中的决策紧密相关，提供了一个丰富而代表性的环境，用于评估超出数学和代码之外的推理能力。为此，我们开发了一个基于推理的子集，以促进o1-like模型的评估。 

---
# State Stream Transformer (SST) : Emergent Metacognitive Behaviours Through Latent State Persistence 

**Title (ZH)**: 状态流变换器（SST）：通过潜在状态持久化实现 emergent 元认知行为 

**Authors**: Thea Aviss  

**Link**: [PDF](https://arxiv.org/pdf/2501.18356)  

**Abstract**: We introduce the State Stream Transformer (SST), a novel LLM architecture that reveals emergent reasoning behaviours and capabilities latent in pretrained weights through addressing a fundamental limitation in traditional transformer models: the lack of latent computational continuity across autoregressive generations in the state space. SST introduces a sliding window latent state (FFN) cache with weighted decay that maintains and evolves persistent latent processes throughout autoregressive generations. Through controlled experiments comparing base and SST architectures using the same frozen weights, we demonstrate that this architectural modification alone enables enhanced reasoning capabilities which appear best explained by some form of potential higher-order processing, as evidenced by emergent metacognitive behaviours. These behaviours persist under controlled conditions designed to eliminate confounding factors such as stochastic variation or learned response patterns. Analysis of latent state distributions and processing dynamics provides evidence that it is solely the 'state stream' that is responsible for these phenomena. In quantitative evaluations, the SST achieves substantial performance improvements over the base model on two reasoning benchmarks, reaching 89.01\% accuracy on GSM-8K (0-shot) and 91.04\% on ARC Challenge (0-shot CoT). These findings indicate that persistent computation in the latent state space enables fundamentally different information processing and internal reasoning strategies, with implications for our understanding of artificial intelligence systems. 

**Abstract (ZH)**: 我们引入了State Stream Transformer (SST)，这是一种新型的大型语言模型（LLM）架构，通过解决传统变压器模型的基本限制——状态空间中自回归生成过程中缺乏潜在的连续计算能力——揭示了预训练权重中潜藏的新兴推理行为和能力。SST 引入了一个具有加权衰减的滑动窗口潜在状态（FFN）缓存，以保持并发展自回归生成过程中的持久潜在过程。通过使用相同的冻结权重进行控制实验，我们将基本架构和SST架构进行比较，证明这种架构修改本身就使模型具备了增强的推理能力，这些能力似乎是某种形式的潜在高阶处理所解释的，这些行为在旨在消除混淆因素（如随机波动或学习到的响应模式）的设计条件下仍然持续存在。对潜在状态分布和处理动态的分析显示，正是“状态流”导致了这些现象。在定量评估中，SST 在两个推理基准测试中比基线模型取得了显著的性能提升，在 GSM-8K（零样本）测试中达到了 89.01% 的准确率，在 ARC 挑战赛（零样本推理）中达到了 91.04%。这些发现表明，潜在状态空间中的持续计算能够实现根本不同的信息处理和内部推理策略，这对理解人工智能系统具有重要意义。 

---
# A Video-grounded Dialogue Dataset and Metric for Event-driven Activities 

**Title (ZH)**: 一个基于视频的对话数据集及其评价指标，用于事件驱动的活动 

**Authors**: Wiradee Imrattanatrai, Masaki Asada, Kimihiro Hasegawa, Zhi-Qi Cheng, Ken Fukuda, Teruko Mitamura  

**Link**: [PDF](https://arxiv.org/pdf/2501.18324)  

**Abstract**: This paper presents VDAct, a dataset for a Video-grounded Dialogue on Event-driven Activities, alongside VDEval, a session-based context evaluation metric specially designed for the task. Unlike existing datasets, VDAct includes longer and more complex video sequences that depict a variety of event-driven activities that require advanced contextual understanding for accurate response generation. The dataset comprises 3,000 dialogues with over 30,000 question-and-answer pairs, derived from 1,000 videos with diverse activity scenarios. VDAct displays a notably challenging characteristic due to its broad spectrum of activity scenarios and wide range of question types. Empirical studies on state-of-the-art vision foundation models highlight their limitations in addressing certain question types on our dataset. Furthermore, VDEval, which integrates dialogue session history and video content summaries extracted from our supplementary Knowledge Graphs to evaluate individual responses, demonstrates a significantly higher correlation with human assessments on the VDAct dataset than existing evaluation metrics that rely solely on the context of single dialogue turns. 

**Abstract (ZH)**: 本文介绍了VDAct数据集，该数据集是关于事件驱动活动的视频引导对话，同时还引入了VDEval，这是一种特别为该任务设计的基于会话的上下文评估指标。与现有的数据集不同，VDAct包含更长、更复杂的视频序列，这些视频序列展示了需要高级上下文理解才能生成准确响应的各种事件驱动活动。该数据集包含3000次对话，超过30,000个问答对，来源于1000个具有多种活动场景的视频。VDAct表现出明显的挑战性特征，源于其广泛的应用场景和多种类型的问答。对最先进的视觉基础模型进行的实证研究表明，这些模型在处理我们数据集中的某些问答类型时存在局限性。此外，VDEval通过整合从我们补充的知识图谱中提取的对话会话历史和视频内容摘要来评估个体响应，其与人类评估在VDAct数据集上的相关性显著高于仅依赖单一对话回合背景的现有评估指标。 

---
# Citation Recommendation based on Argumentative Zoning of User Queries 

**Title (ZH)**: 基于用户查询论元区隔的引用推荐 

**Authors**: Shutian Ma, Chengzhi Zhang, Heng Zhang, Zheng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2501.18292)  

**Abstract**: Citation recommendation aims to locate the important papers for scholars to cite. When writing the citing sentences, the authors usually hold different citing intents, which are referred to citation function in citation analysis. Since argumentative zoning is to identify the argumentative and rhetorical structure in scientific literature, we want to use this information to improve the citation recommendation task. In this paper, a multi-task learning model is built for citation recommendation and argumentative zoning classification. We also generated an annotated corpus of the data from PubMed Central based on a new argumentative zoning schema. The experimental results show that, by considering the argumentative information in the citing sentence, citation recommendation model will get better performance. 

**Abstract (ZH)**: 引文推荐旨在为学者找到需要引用的重要论文。在撰写引文句子时，作者通常有不同的引文意图，这些意图在引文分析中被称为引文功能。鉴于论证区域划分是识别科学文献中的论证和修辞结构，我们希望通过利用这些信息来改进引文推荐任务。在本文中，我们构建了一个多任务学习模型，用于引文推荐和论证区域分类。我们还根据新的论证区域划分方案，从PubMed Central生成了一个标注数据集。实验结果表明，通过考虑引文句子中的论证信息，引文推荐模型的性能将得到提升。 

---
# Collecting Cost-Effective, High-Quality Truthfulness Assessments with LLM Summarized Evidence 

**Title (ZH)**: 使用LLM总结的证据收集高效、高质量的可信度评估 

**Authors**: Kevin Roitero, Dustin Wright, Michael Soprano, Isabelle Augenstein, Stefano Mizzaro  

**Link**: [PDF](https://arxiv.org/pdf/2501.18265)  

**Abstract**: With the degradation of guardrails against mis- and disinformation online, it is more critical than ever to be able to effectively combat it. In this paper, we explore the efficiency and effectiveness of using crowd-sourced truthfulness assessments based on condensed, large language model (LLM) generated summaries of online sources. We compare the use of generated summaries to the use of original web pages in an A/B testing setting, where we employ a large and diverse pool of crowd-workers to perform the truthfulness assessment. We evaluate the quality of assessments, the efficiency with which assessments are performed, and the behavior and engagement of participants. Our results demonstrate that the Summary modality, which relies on summarized evidence, offers no significant change in assessment accuracy over the Standard modality, while significantly increasing the speed with which assessments are performed. Workers using summarized evidence produce a significantly higher number of assessments in the same time frame, reducing the cost needed to acquire truthfulness assessments. Additionally, the Summary modality maximizes both the inter-annotator agreements as well as the reliance on and perceived usefulness of evidence, demonstrating the utility of summarized evidence without sacrificing the quality of assessments. 

**Abstract (ZH)**: 随着防护栏对虚假和误导信息的保护能力下降，有效地对抗这些信息比以往任何时候都更为重要。在本文中，我们探讨了利用基于大型语言模型（LLM）生成的在线信息浓缩摘要的众包真实度评估的有效性和效率。我们将生成的摘要与原始网页进行对比，在A/B测试环境中，采用大量多样化的众包工人进行真实度评估。我们评估了评估的质量、执行效率，以及参与者的言行和参与度。结果显示，依赖浓缩证据的摘要模态并未显著提高评估准确性，但显著提高了评估的执行速度。使用浓缩证据的工人在相同时间内能够产生更多评估，从而降低了获取真实度评估的成本。此外，摘要模态不仅最大化了注释者之间的共识，还提高了对证据的依赖性和感知有用性，证明了浓缩证据的效用，而不会牺牲评估的质量。 

---
# Statistical multi-metric evaluation and visualization of LLM system predictive performance 

**Title (ZH)**: 统计多指标评估与可视化LLM系统预测性能 

**Authors**: Samuel Ackerman, Eitan Farchi, Orna Raz, Assaf Toledo  

**Link**: [PDF](https://arxiv.org/pdf/2501.18243)  

**Abstract**: The evaluation of generative or discriminative large language model (LLM)-based systems is often a complex multi-dimensional problem. Typically, a set of system configuration alternatives are evaluated on one or more benchmark datasets, each with one or more evaluation metrics, which may differ between datasets. We often want to evaluate -- with a statistical measure of significance -- whether systems perform differently either on a given dataset according to a single metric, on aggregate across metrics on a dataset, or across datasets. Such evaluations can be done to support decision-making, such as deciding whether a particular system component change (e.g., choice of LLM or hyperparameter values) significantly improves performance over the current system configuration, or, more generally, whether a fixed set of system configurations (e.g., a leaderboard list) have significantly different performances according to metrics of interest. We present a framework implementation that automatically performs the correct statistical tests, properly aggregates the statistical results across metrics and datasets (a nontrivial task), and can visualize the results. The framework is demonstrated on the multi-lingual code generation benchmark CrossCodeEval, for several state-of-the-art LLMs. 

**Abstract (ZH)**: 基于生成式或判别式大型语言模型（LLM）的系统评估往往是多维度的复杂问题。通常，会对系统配置的不同选择进行评估，这些评估可能在单一基准数据集上使用一个或多个评估指标，或者在多个指标上评估同一个数据集，或者跨越不同的数据集进行评估。这些评估可以用于支持决策，例如，判断某个特定的系统组件更改（例如，选择不同的LLM或超参数值）是否显著地提高了当前系统配置的性能，或者更一般地，判断一系列固定系统配置（例如，排行榜列表）在感兴趣指标上的性能是否存在显著差异。我们提出了一种框架实现，该框架能够自动执行正确的统计检验，并正确地跨指标和数据集汇总统计结果（这是一个非平凡的任务），并且可以可视化结果。该框架已在多语言代码生成基准CrossCodeEval上对几种最先进的LLM进行了演示。 

---
# Scaling Inference-Efficient Language Models 

**Title (ZH)**: 高效推理的语言模型的扩展 

**Authors**: Song Bian, Minghao Yan, Shivaram Venkataraman  

**Link**: [PDF](https://arxiv.org/pdf/2501.18107)  

**Abstract**: Scaling laws are powerful tools to predict the performance of large language models. However, current scaling laws fall short of accounting for inference costs. In this work, we first show that model architecture affects inference latency, where models of the same size can have up to 3.5x difference in latency. To tackle this challenge, we modify the Chinchilla scaling laws to co-optimize the model parameter count, the number of training tokens, and the model architecture. Due to the reason that models of similar training loss exhibit gaps in downstream evaluation, we also propose a novel method to train inference-efficient models based on the revised scaling laws. We perform extensive empirical studies to fit and evaluate our inference-aware scaling laws. We vary model parameters from 80M to 1B, training tokens from 1.6B to 30B, and model shapes, training a total of 63 models. Guided by our inference-efficient scaling law and model selection method, we release the Morph-1B model, which improves inference latency by 1.8x while maintaining accuracy on downstream tasks compared to open-source models, pushing the Pareto frontier of accuracy-latency tradeoff. 

**Abstract (ZH)**: 规模律是预测大型语言模型性能的强大工具。然而，现有的规模律未能充分考虑推理成本。在本文中，我们首先展示了模型架构会影响推理延迟，即使是相同规模的模型，在延迟上也可能存在高达3.5倍的差异。为应对这一挑战，我们改进了Chinchilla规模律，同时优化模型参数数量、训练令牌数量以及模型架构。由于具有类似训练损失的模型在下游评估中表现出差异，我们还提出了一种基于修订后的规模律训练推理高效模型的新方法。我们进行了广泛的实证研究来拟合和评估我们的感知延迟规模律。我们从80M到1B变化模型参数，从1.6B到30B变化训练令牌数量，并调整了模型形状，总共训练了63个模型。在我们的感知延迟规模律和模型选择方法的指导下，我们发布了Morph-1B模型，在保持下游任务准确性的前提下，将推理延迟提升了1.8倍，推动了精度-延迟权衡 Pareto 边界的前沿。 

---
# Beyond Turn-taking: Introducing Text-based Overlap into Human-LLM Interactions 

**Title (ZH)**: 超越轮转：引入基于文本的重叠到人-大语言模型互动中 

**Authors**: JiWoo Kim, Minsuk Chang, JinYeong Bak  

**Link**: [PDF](https://arxiv.org/pdf/2501.18103)  

**Abstract**: Traditional text-based human-AI interactions often adhere to a strict turn-taking approach. In this research, we propose a novel approach that incorporates overlapping messages, mirroring natural human conversations. Through a formative study, we observed that even in text-based contexts, users instinctively engage in overlapping behaviors like "A: Today I went to-" "B: yeah." To capitalize on these insights, we developed OverlapBot, a prototype chatbot where both AI and users can initiate overlapping. Our user study revealed that OverlapBot was perceived as more communicative and immersive than traditional turn-taking chatbot, fostering faster and more natural interactions. Our findings contribute to the understanding of design space for overlapping interactions. We also provide recommendations for implementing overlap-capable AI interactions to enhance the fluidity and engagement of text-based conversations. 

**Abstract (ZH)**: 传统的基于文本的人机交互通常遵循严格的轮流对话模式。本研究提出了一种新的方法，该方法结合了重叠消息，模仿自然的人类对话。通过一种形成性研究，我们观察到，即使在基于文本的环境中，用户也会本能地表现出重叠行为，如“A: 今天我去-”“B: 嗯。”为了利用这些见解，我们开发了OverlapBot，这是一种原型聊天机器人，其中AI和用户都可以发起重叠对话。用户研究表明，与传统的轮流对话聊天机器人相比，OverlapBot 被视为更具沟通性和沉浸感，促进了更自然和快速的交互。我们的发现为重叠交互的设计空间提供了新的认识。我们还提供了实施具备重叠能力的人工智能交互的建议，以增强基于文本对话的流畅性和参与性。 

---
# Learning to Plan & Reason for Evaluation with Thinking-LLM-as-a-Judge 

**Title (ZH)**: 学习规划与推理以Thinking-LLM为法官进行评估 

**Authors**: Swarnadeep Saha, Xian Li, Marjan Ghazvininejad, Jason Weston, Tianlu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.18099)  

**Abstract**: LLM-as-a-Judge models generate chain-of-thought (CoT) sequences intended to capture the step-bystep reasoning process that underlies the final evaluation of a response. However, due to the lack of human annotated CoTs for evaluation, the required components and structure of effective reasoning traces remain understudied. Consequently, previous approaches often (1) constrain reasoning traces to hand-designed components, such as a list of criteria, reference answers, or verification questions and (2) structure them such that planning is intertwined with the reasoning for evaluation. In this work, we propose EvalPlanner, a preference optimization algorithm for Thinking-LLM-as-a-Judge that first generates an unconstrained evaluation plan, followed by its execution, and then the final judgment. In a self-training loop, EvalPlanner iteratively optimizes over synthetically constructed evaluation plans and executions, leading to better final verdicts. Our method achieves a new state-of-the-art performance for generative reward models on RewardBench (with a score of 93.9), despite being trained on fewer amount of, and synthetically generated, preference pairs. Additional experiments on other benchmarks like RM-Bench, JudgeBench, and FollowBenchEval further highlight the utility of both planning and reasoning for building robust LLM-as-a-Judge reasoning models. 

**Abstract (ZH)**: 大规模语言模型作为法官（LLM-as-a-Judge）模型生成逐步推理序列（CoT），旨在捕捉最终回应评估背后一步一步的推理过程。然而，由于缺乏人工注释的CoT进行评估，有效推理路径所需的组成部分和结构仍不够明了。因此，之前的许多方法往往（1）限制推理路径到手工设计的组件，如一系列标准、参考答案或验证问题，并且（2）将规划与评估推理相互交织。在本项工作中，我们提出了一种名为EvalPlanner的偏好优化算法，它首先生成一个不受约束的评估计划，然后执行该计划，最后得出最终判决。在一个自我训练循环中，EvalPlanner通过迭代优化合成构造的评估计划及其执行，从而提高最终的判决效果。尽管我们的方法仅在较少的合成偏好对数据上进行训练，但在RewardBench上取得了新的最佳性能（得分为93.9）。此外，我们在其他基准测试如RM-Bench、JudgeBench和FollowBenchEval中的实验进一步突显了规划和推理的实用性，对于构建稳健的大规模语言模型作为法官的推理模型具有重要意义。 

---
# LLMs can see and hear without any training 

**Title (ZH)**: LLMs可以在未经过任何训练的情况下看到和听到。 

说明：这个翻译尽量保持了原文的意思，同时符合学术写作的规范。不过，需要根据具体上下文进行适当调整以确保准确性和流畅性。如果“看到和听到”指的是模型处理视觉和听觉数据的能力，可以进一步明确表达，例如：

大型语言模型(LLMs)可以在未经过专门视觉或听觉训练的情况下处理和理解视觉和听觉数据。 

**Authors**: Kumar Ashutosh, Yossi Gandelsman, Xinlei Chen, Ishan Misra, Rohit Girdhar  

**Link**: [PDF](https://arxiv.org/pdf/2501.18096)  

**Abstract**: We present MILS: Multimodal Iterative LLM Solver, a surprisingly simple, training-free approach, to imbue multimodal capabilities into your favorite LLM. Leveraging their innate ability to perform multi-step reasoning, MILS prompts the LLM to generate candidate outputs, each of which are scored and fed back iteratively, eventually generating a solution to the task. This enables various applications that typically require training specialized models on task-specific data. In particular, we establish a new state-of-the-art on emergent zero-shot image, video and audio captioning. MILS seamlessly applies to media generation as well, discovering prompt rewrites to improve text-to-image generation, and even edit prompts for style transfer! Finally, being a gradient-free optimization approach, MILS can invert multimodal embeddings into text, enabling applications like cross-modal arithmetic. 

**Abstract (ZH)**: 我们提出了MILS：多模态迭代LLM求解器，这是一种出人意料地简单且无需训练的方法，用于将多模态能力注入您喜爱的LLM中。利用它们本身具备的多步骤推理能力，MILS促使LLM生成候选输出，每个输出都会被打分并反馈回去进行迭代，最终生成任务的解决方案。这种方法能够应用于通常需要在任务特定数据上训练专门模型的各种应用场景。特别是在零样本图像、视频和音频描述方面，我们取得了新的最佳结果。MILS还能无缝应用于媒体生成，发现提示重写以提高文本到图像生成的质量，并且甚至可以编辑提示进行风格转移！最后，作为一种无梯度优化方法，MILS能够将多模态嵌入反向转换为文本，从而实现跨模态算术等应用场景。 

---
# FinanceQA: A Benchmark for Evaluating Financial Analysis Capabilities of Large Language Models 

**Title (ZH)**: FinanceQA：评估大型语言模型财务分析能力的标准基准 

**Authors**: Spencer Mateega, Carlos Georgescu, Danny Tang  

**Link**: [PDF](https://arxiv.org/pdf/2501.18062)  

**Abstract**: FinanceQA is a testing suite that evaluates LLMs' performance on complex numerical financial analysis tasks that mirror real-world investment work. Despite recent advances, current LLMs fail to meet the strict accuracy requirements of financial institutions, with models failing approximately 60% of realistic tasks that mimic on-the-job analyses at hedge funds, private equity firms, investment banks, and other financial institutions. The primary challenges include hand-spreading metrics, adhering to standard accounting and corporate valuation conventions, and performing analysis under incomplete information - particularly in multi-step tasks requiring assumption generation. This performance gap highlights the disconnect between existing LLM capabilities and the demands of professional financial analysis that are inadequately tested by current testing architectures. Results show that higher-quality training data is needed to support such tasks, which we experiment with using OpenAI's fine-tuning API. FinanceQA is publicly released at [this https URL](this https URL). 

**Abstract (ZH)**: FinanceQA 是一个测试套件，用于评估大型语言模型（LLM）在复杂数值金融分析任务上的性能，这些任务能够反映现实中的投资工作场景。尽管最近取得了进展，当前的 LLM 仍未达到金融机构严格的准确要求，在对冲基金、私人 equity 公司、投资银行及其他金融机构中模拟实际工作分析的任务失败率约为 60%。主要挑战包括手工拆分指标、遵循标准会计和企业估值惯例以及在信息不完整的情况下进行分析，特别是在需要假设生成的多步任务中。这一性能差距突显了现有 LLM 能力与专业金融分析需求之间的差距，而当前的测试架构未能充分测试这些要求。实验结果表明，需要更高质量的训练数据来支持此类任务，我们利用 OpenAI 的微调 API 进行了相应的实验。FinanceQA 已在 [此链接]（[this https URL]）上公开发布。 

---
# From tools to thieves: Measuring and understanding public perceptions of AI through crowdsourced metaphors 

**Title (ZH)**: 从工具到窃贼：通过众包类比测量和理解公众对人工智能的看法 

**Authors**: Myra Cheng, Angela Y. Lee, Kristina Rapuano, Kate Niederhoffer, Alex Liebscher, Jeffrey Hancock  

**Link**: [PDF](https://arxiv.org/pdf/2501.18045)  

**Abstract**: How has the public responded to the increasing prevalence of artificial intelligence (AI)-based technologies? We investigate public perceptions of AI by collecting over 12,000 responses over 12 months from a nationally representative U.S. sample. Participants provided open-ended metaphors reflecting their mental models of AI, a methodology that overcomes the limitations of traditional self-reported measures. Using a mixed-methods approach combining quantitative clustering and qualitative coding, we identify 20 dominant metaphors shaping public understanding of AI. To analyze these metaphors systematically, we present a scalable framework integrating language modeling (LM)-based techniques to measure key dimensions of public perception: anthropomorphism (attribution of human-like qualities), warmth, and competence. We find that Americans generally view AI as warm and competent, and that over the past year, perceptions of AI's human-likeness and warmth have significantly increased ($+34\%, r = 0.80, p < 0.01; +41\%, r = 0.62, p < 0.05$). Furthermore, these implicit perceptions, along with the identified dominant metaphors, strongly predict trust in and willingness to adopt AI ($r^2 = 0.21, 0.18, p < 0.001$). We further explore how differences in metaphors and implicit perceptions--such as the higher propensity of women, older individuals, and people of color to anthropomorphize AI--shed light on demographic disparities in trust and adoption. In addition to our dataset and framework for tracking evolving public attitudes, we provide actionable insights on using metaphors for inclusive and responsible AI development. 

**Abstract (ZH)**: 公众对基于人工智能（AI）技术的日益普及有何反应？我们通过收集来自全国代表性的美国样本的超过12,000份回应，研究公众对AI的看法。参与者提供了开放式的隐喻，反映了他们对AI的心理模型，这种方法克服了传统自我报告测度的局限性。利用混合方法结合定量聚类和定性编码，我们识别出了20个主导的隐喻，这些隐喻塑造了公众对AI的理解。为了系统地分析这些隐喻，我们提供了一个可扩展的框架，结合语言模型（LM）技术来衡量公众感知的关键维度：拟人性（赋予人类特质）、温暖和能力。

研究发现，美国人普遍认为AI是温暖且有能力的，而且在过去一年，公众对AI的人类特质感知和温暖程度显著提高（分别为+34%，相关系数r = 0.80，p < 0.01；+41%，相关系数r = 0.62，p < 0.05）。此外，这些隐含的感知以及所识别出的主导隐喻强烈预测了公众对AI的信任度和采用意愿（解释变异度分别为21%，18%，p < 0.001）。我们进一步探讨了不同隐喻和隐含感知的差异，如女性、老年人和有色人种更倾向于将拟人性赋予AI，这揭示了不同社会群体在信任和采用方面的差异。除了我们的数据集和追踪公众态度变化的框架外，我们还提供了关于如何使用隐喻促进包容性和负责任的AI发展的实际见解。 

---
# DReSS: Data-driven Regularized Structured Streamlining for Large Language Models 

**Title (ZH)**: DReSS：面向数据驱动的正则化结构化流水线精简方法用于大型语言模型

在这个翻译中，我尽量保持了原文的技术术语和学术规范。"DReSS"被保留为原名，因为它是论文的缩写名称。其他部分如“数据驱动的正则化结构化流水线精简方法”和“大型语言模型”都是学术化的表达方式。 

**Authors**: Mingkuan Feng, Jinyang Wu, Shuai Zhang, Pengpeng Shao, Ruihan Jin, Zhengqi Wen, Jianhua Tao, Feihu Che  

**Link**: [PDF](https://arxiv.org/pdf/2501.17905)  

**Abstract**: Large language models (LLMs) have achieved significant progress across various domains, but their increasing scale results in high computational and memory costs. Recent studies have revealed that LLMs exhibit sparsity, providing the potential to reduce model size through pruning techniques. However, existing pruning methods typically follow a prune-then-finetune paradigm. Since the pruned components still contain valuable information, their direct removal often leads to irreversible performance degradation, imposing a substantial computational burden to recover performance during finetuning. In this paper, we propose a novel paradigm that first applies regularization, then prunes, and finally finetunes. Based on this paradigm, we introduce DReSS, a simple and effective Data-driven Regularized Structured Streamlining method for LLMs. By leveraging a small amount of data to regularize the components to be pruned, DReSS explicitly transfers the important information to the remaining parts of the model in advance. Compared to direct pruning, this can reduce the information loss caused by parameter removal, thereby enhancing its language modeling capabilities. Experimental results demonstrate that DReSS significantly outperforms existing pruning methods even under extreme pruning ratios, significantly reducing latency and increasing throughput. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在各个领域取得了显著进展，但随着模型规模的增加，其计算和内存成本也大幅提升。近期研究发现，LLMs表现出稀疏性，为通过剪枝技术减小模型大小提供了可能性。然而，现有的剪枝方法通常遵循先剪枝后微调的范式。由于被剪枝的部分仍然包含有价值的信息，直接移除这些部分往往会导致不可逆的性能下降，从而在微调过程中给恢复性能带来巨大的计算负担。本文提出了一种新的范式：先应用正则化，再进行剪枝，最后进行微调。基于这一范式，我们引入了DReSS，一种简单且有效的大规模语言模型数据驱动正则化结构化剪枝方法。通过利用少量数据对将要剪枝的组件进行正则化，DReSS可以提前将重要信息转移到模型剩余部分。与直接剪枝相比，这种方法可以减少因参数移除而造成的信息损失，从而增强其语言建模能力。实验结果表明，即使在极端剪枝比的情况下，DReSS也显著优于现有的剪枝方法，显著降低了延迟并提高了吞吐量。 

---
