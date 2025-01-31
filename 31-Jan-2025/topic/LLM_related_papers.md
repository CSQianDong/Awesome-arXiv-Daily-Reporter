# Illusions of Relevance: Using Content Injection Attacks to Deceive Retrievers, Rerankers, and LLM Judges 

**Title (ZH)**: 虚幻的相关性：使用内容注入攻击欺骗检索器、 reranking器和LLM裁判机 

**Authors**: Manveer Singh Tamber, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2501.18536)  

**Abstract**: Consider a scenario in which a user searches for information, only to encounter texts flooded with misleading or non-relevant content. This scenario exemplifies a simple yet potent vulnerability in neural Information Retrieval (IR) pipelines: content injection attacks. We find that embedding models for retrieval, rerankers, and large language model (LLM) relevance judges are vulnerable to these attacks, in which adversaries insert misleading text into passages to manipulate model judgements. We identify two primary threats: (1) inserting unrelated or harmful content within passages that still appear deceptively "relevant", and (2) inserting entire queries or key query terms into passages to boost their perceived relevance. While the second tactic has been explored in prior research, we present, to our knowledge, the first empirical analysis of the first threat, demonstrating how state-of-the-art models can be easily misled. Our study systematically examines the factors that influence an attack's success, such as the placement of injected content and the balance between relevant and non-relevant material. Additionally, we explore various defense strategies, including adversarial passage classifiers, retriever fine-tuning to discount manipulated content, and prompting LLM judges to adopt a more cautious approach. However, we find that these countermeasures often involve trade-offs, sacrificing effectiveness for attack robustness and sometimes penalizing legitimate documents in the process. Our findings highlight the need for stronger defenses against these evolving adversarial strategies to maintain the trustworthiness of IR systems. We release our code and scripts to facilitate further research. 

**Abstract (ZH)**: 用户在搜索信息时，往往会遇到充斥着误导性或不相关内容的文本。这种场景揭示了神经信息检索（IR）管道中的一个简单而强大的漏洞：内容注入攻击。我们发现，用于检索、重排和大型语言模型（LLM）相关性判断的嵌入模型都容易受到这些攻击的影响，攻击者会向文段中插入误导性文本以操控模型的判断结果。我们识别出两类主要的威胁：（1）在看似相关但实际上是不相关的文段中插入无关或有害内容；（2）将整个查询或关键查询术语插入文段以提高其感知相关性。虽然第二种战术在以往的研究中已有探索，但据我们所知，本研究首次进行了第一种威胁的实证分析，展示了最先进的模型如何容易被误导。我们系统地研究了影响攻击成功率的各种因素，如插入内容的位置以及相关和不相关材料之间的平衡。此外，我们还探讨了多种防御策略，包括对抗文段分类器、对检索器进行微调以忽略操纵内容，以及提示LLM判断者采取更加谨慎的态度。然而，我们发现这些反制措施常常存在权衡，可能会牺牲有效性以增强攻击抵御能力，并且有时会对合法文档造成惩罚。我们的研究结果强调了需要更强的防御措施来应对不断演变的对抗策略，以维护信息检索系统的可信度。我们发布了代码和脚本，以促进进一步的研究。 

---
# Collecting Cost-Effective, High-Quality Truthfulness Assessments with LLM Summarized Evidence 

**Title (ZH)**: 使用LLM总结的证据收集高效、高质量的可信度评估 

**Authors**: Kevin Roitero, Dustin Wright, Michael Soprano, Isabelle Augenstein, Stefano Mizzaro  

**Link**: [PDF](https://arxiv.org/pdf/2501.18265)  

**Abstract**: With the degradation of guardrails against mis- and disinformation online, it is more critical than ever to be able to effectively combat it. In this paper, we explore the efficiency and effectiveness of using crowd-sourced truthfulness assessments based on condensed, large language model (LLM) generated summaries of online sources. We compare the use of generated summaries to the use of original web pages in an A/B testing setting, where we employ a large and diverse pool of crowd-workers to perform the truthfulness assessment. We evaluate the quality of assessments, the efficiency with which assessments are performed, and the behavior and engagement of participants. Our results demonstrate that the Summary modality, which relies on summarized evidence, offers no significant change in assessment accuracy over the Standard modality, while significantly increasing the speed with which assessments are performed. Workers using summarized evidence produce a significantly higher number of assessments in the same time frame, reducing the cost needed to acquire truthfulness assessments. Additionally, the Summary modality maximizes both the inter-annotator agreements as well as the reliance on and perceived usefulness of evidence, demonstrating the utility of summarized evidence without sacrificing the quality of assessments. 

**Abstract (ZH)**: 随着防护栏对虚假和误导信息的保护能力下降，有效地对抗这些信息比以往任何时候都更为重要。在本文中，我们探讨了利用基于大型语言模型（LLM）生成的在线信息浓缩摘要的众包真实度评估的有效性和效率。我们将生成的摘要与原始网页进行对比，在A/B测试环境中，采用大量多样化的众包工人进行真实度评估。我们评估了评估的质量、执行效率，以及参与者的言行和参与度。结果显示，依赖浓缩证据的摘要模态并未显著提高评估准确性，但显著提高了评估的执行速度。使用浓缩证据的工人在相同时间内能够产生更多评估，从而降低了获取真实度评估的成本。此外，摘要模态不仅最大化了注释者之间的共识，还提高了对证据的依赖性和感知有用性，证明了浓缩证据的效用，而不会牺牲评估的质量。 

---
# Investigating Tax Evasion Emergence Using Dual Large Language Model and Deep Reinforcement Learning Powered Agent-based Simulation 

**Title (ZH)**: 使用双大型语言模型和深度强化学习驱动的基于代理的仿真研究逃税行为的 emergence 

**Authors**: Teddy Lazebnik, Labib Shami  

**Link**: [PDF](https://arxiv.org/pdf/2501.18177)  

**Abstract**: Tax evasion, usually the largest component of an informal economy, is a persistent challenge over history with significant socio-economic implications. Many socio-economic studies investigate its dynamics, including influencing factors, the role and influence of taxation policies, and the prediction of the tax evasion volume over time. These studies assumed such behavior is given, as observed in the real world, neglecting the "big bang" of such activity in a population. To this end, computational economy studies adopted developments in computer simulations, in general, and recent innovations in artificial intelligence (AI), in particular, to simulate and study informal economy appearance in various socio-economic settings. This study presents a novel computational framework to examine the dynamics of tax evasion and the emergence of informal economic activity. Employing an agent-based simulation powered by Large Language Models and Deep Reinforcement Learning, the framework is uniquely designed to allow informal economic behaviors to emerge organically, without presupposing their existence or explicitly signaling agents about the possibility of evasion. This provides a rigorous approach for exploring the socio-economic determinants of compliance behavior. The experimental design, comprising model validation and exploratory phases, demonstrates the framework's robustness in replicating theoretical economic behaviors. Findings indicate that individual personality traits, external narratives, enforcement probabilities, and the perceived efficiency of public goods provision significantly influence both the timing and extent of informal economic activity. The results underscore that efficient public goods provision and robust enforcement mechanisms are complementary; neither alone is sufficient to curtail informal activity effectively. 

**Abstract (ZH)**: 税收 evasion 通常是非正式经济中最大的组成部分，一直是历史上一个持久的挑战，具有显著的经济社会影响。许多经济社会研究探讨了税收 evasion 的动态，包括影响因素、税收政策的作用和影响，以及税收 evasion 规模的预测。这些研究假设这种行为是给定的，基于现实世界中的观察，忽略了这种活动在人群中的“突变”。为此，计算经济学研究采用了计算机模拟的一般发展，尤其是最近人工智能（AI）的创新成果，来模拟和研究在不同经济社会环境中的非正式经济活动出现。

本研究提出了一种新的计算框架，以探讨税收 evasion 的动态及其非正式经济活动的产生。通过基于大型语言模型和深度强化学习的代理基础仿真，该框架独特地设计了无需假设其存在或明确指示代理存在逃税可能性的方法，从而使其有机地产生非正式经济行为。这种方法为探索合规行为的经济社会决定因素提供了严谨的方法。实验设计包括模型验证和探索阶段，展示了该框架在重现理论经济行为方面的稳健性。研究结果表明，个体人格特质、外部叙述、实施可能性以及公共物品提供效率显著影响非正式经济活动的时间和规模。研究结果强调了高效的公共物品提供和严格的执法机制之间的互补性；两者单独均不足以有效遏制非正式活动。 

---
# RL-based Query Rewriting with Distilled LLM for online E-Commerce Systems 

**Title (ZH)**: 基于RL的查询重写方法：结合精炼的LLM在线电子商务系统中的应用 

**Authors**: Duy A. Nguyen, Rishi Kesav Mohan, Van Yang, Pritom Saha Akash, Kevin Chen-Chuan Chang  

**Link**: [PDF](https://arxiv.org/pdf/2501.18056)  

**Abstract**: Query rewriting (QR) is a critical technique in e-commerce search, addressing the lexical gap between user queries and product descriptions to enhance search performance. Existing QR approaches typically fall into two categories: discriminative models and generative methods leveraging large language models (LLMs). Discriminative models often struggle with natural language understanding and offer limited flexibility in rewriting, while generative LLMs, despite producing high-quality rewrites, face high inference latency and cost in online settings. These limitations force offline deployment, making them vulnerable to issues like information staleness and semantic drift. To overcome these challenges, we propose a novel hybrid pipeline for QR that balances efficiency and effectiveness. Our approach combines offline knowledge distillation to create a lightweight but efficient student model with online reinforcement learning (RL) to refine query rewriting dynamically using real-time feedback. A key innovation is the use of LLMs as simulated human feedback, enabling scalable reward signals and cost-effective evaluation without manual annotations. Experimental results on Amazon ESCI dataset demonstrate significant improvements in query relevance, diversity, and adaptability, as well as positive feedback from the LLM simulation. This work contributes to advancing LLM capabilities for domain-specific applications, offering a robust solution for dynamic and complex e-commerce search environments. 

**Abstract (ZH)**: 查询重写（Query Rewriting, QR）是电子商务搜索中的一个关键技术，用于弥合用户查询与产品描述之间的词汇差距，从而提高搜索性能。现有的QR方法通常可以分为两类：判别模型和利用大规模语言模型（LLMs）的生成方法。判别模型在自然语言理解方面往往表现不足，灵活性有限；而生成的LLMs虽然可以生成高质量的重写，但在在线环境中面临高昂的推理延迟和成本问题。这些限制迫使它们进行离线部署，使它们容易受到信息过时和语义漂移等问题的影响。为了解决这些问题，我们提出了一种新的混合管道方法，以平衡效率和有效性。我们的方法结合了离线知识蒸馏来创建一个轻量级但高效的student模型，并在线使用强化学习（RL）根据实时反馈动态改进查询重写。一个关键的创新在于使用LLMs作为模拟的人类反馈，这使得奖励信号的标定更加可扩展，同时也提供了一种经济有效的评估方法，无需人工注释。我们在Amazon ESCI数据集上的实验结果表明，在查询相关性、多样性和适应性方面取得了显著的改进，并且LLM模拟得到了积极的反馈。这项工作有助于推进LLMs在特定领域应用中的能力，并为动态和复杂的电子商务搜索环境提供了一个稳健的解决方案。 

---
# Can Generative LLMs Create Query Variants for Test Collections? An Exploratory Study 

**Title (ZH)**: 生成式大型语言模型能否为测试集合创建查询变体？一项探索性研究 

**Authors**: Marwah Alaofi, Luke Gallagher, Mark Sanderson, Falk Scholer, Paul Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2501.17981)  

**Abstract**: This paper explores the utility of a Large Language Model (LLM) to automatically generate queries and query variants from a description of an information need. Given a set of information needs described as backstories, we explore how similar the queries generated by the LLM are to those generated by humans. We quantify the similarity using different metrics and examine how the use of each set would contribute to document pooling when building test collections. Our results show potential in using LLMs to generate query variants. While they may not fully capture the wide variety of human-generated variants, they generate similar sets of relevant documents, reaching up to 71.1% overlap at a pool depth of 100. 

**Abstract (ZH)**: 本文探讨了大型语言模型（LLM）在从信息需求描述中自动生成查询及其变体方面的效用。给定一组以背景故事形式描述的信息需求，我们研究了LLM生成的查询与人类生成的查询之间的相似性。我们使用不同的度量标准量化这种相似性，并探讨每种方法在构建测试集合时对文档汇集的贡献。研究结果表明，使用LLM生成查询变体具有潜在价值。虽然它们可能无法完全捕捉到人类生成的广泛变体，但它们能够生成相似的相关文档集合，在100个文档池深度时重叠率达到71.1%。 

---
# LLMs can be Fooled into Labelling a Document as Relevant (best caf\'e near me; this paper is perfectly relevant) 

**Title (ZH)**: 大型语言模型可以被欺骗，将一份文档标记为相关（如：“最近的最好咖啡馆在哪里；本文完全相关”） 

**Authors**: Marwah Alaofi, Paul Thomas, Falk Scholer, Mark Sanderson  

**Link**: [PDF](https://arxiv.org/pdf/2501.17969)  

**Abstract**: LLMs are increasingly being used to assess the relevance of information objects. This work reports on experiments to study the labelling of short texts (i.e., passages) for relevance, using multiple open-source and proprietary LLMs. While the overall agreement of some LLMs with human judgements is comparable to human-to-human agreement measured in previous research, LLMs are more likely to label passages as relevant compared to human judges, indicating that LLM labels denoting non-relevance are more reliable than those indicating relevance.
This observation prompts us to further examine cases where human judges and LLMs disagree, particularly when the human judge labels the passage as non-relevant and the LLM labels it as relevant. Results show a tendency for many LLMs to label passages that include the original query terms as relevant. We, therefore, conduct experiments to inject query words into random and irrelevant passages, not unlike the way we inserted the query "best café near me" into this paper. The results show that LLMs are highly influenced by the presence of query words in the passages under assessment, even if the wider passage has no relevance to the query. This tendency of LLMs to be fooled by the mere presence of query words demonstrates a weakness in our current measures of LLM labelling: relying on overall agreement misses important patterns of failures. There is a real risk of bias in LLM-generated relevance labels and, therefore, a risk of bias in rankers trained on those labels.
We also investigate the effects of deliberately manipulating LLMs by instructing them to label passages as relevant, similar to the instruction "this paper is perfectly relevant" inserted above. We find that such manipulation influences the performance of some LLMs, highlighting the critical need to consider potential vulnerabilities when deploying LLMs in real-world applications. 

**Abstract (ZH)**: 大型语言模型（LLMs）越来越多地被用来评估信息对象的相关性。本研究报告了使用多种开源和专有LLMs对简短文本（即段落）进行相关性标注的实验。虽然某些LLMs与人类判断的一致性程度与以往研究中测量的人类之间的一致性相当，但LLMs将段落标记为相关性的频率高于人类评判者，表明LLMs表示不相关性的标签比表示相关性的标签更可靠。

这一观察促使我们进一步研究人类评判者和LLMs存在分歧的案例，特别是当人类评判者将段落标记为不相关，而LLMs将其标记为相关时。结果表明，许多LLMs倾向于将包含原始查询词的段落标记为相关。因此，我们进行了实验，在随机且无关的段落中注入查询词，类似于我们在本文中插入查询词“最佳附近的咖啡馆”那样。结果表明，评估过程中段落中查询词的存在对LLMs有很大影响，即使更广泛的内容与查询无关。LLMs被查询词的存在误导的趋势表明我们当前的LLM标签指标存在缺陷：仅依赖整体一致性忽略了重要的失败模式。LLMs生成的相关性标签存在偏差风险，这进而导致使用这些标签训练的排名器也存在偏差风险。

我们还研究了故意操控LLMs的效果，例如指示它们将段落标记为相关，类似于上面插入的“本文完全相关”指令。我们发现这种操控影响了一些LLMs的性能，这突显了在实际应用中部署LLMs时需要考虑潜在脆弱性的关键需求。 

---
# RbFT: Robust Fine-tuning for Retrieval-Augmented Generation against Retrieval Defects 

**Title (ZH)**: RbFT：针对检索缺陷的鲁棒微调以增强生成检索方法 

**Authors**: Yiteng Tu, Weihang Su, Yujia Zhou, Yiqun Liu, Qingyao Ai  

**Link**: [PDF](https://arxiv.org/pdf/2501.18365)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models (LLMs) by integrating external knowledge retrieved from a knowledge base. However, its effectiveness is fundamentally constrained by the reliability of both the retriever and the knowledge base. In real-world scenarios, imperfections in these components often lead to the retrieval of noisy, irrelevant, or misleading counterfactual information, ultimately undermining the trustworthiness of RAG systems. To address this challenge, we propose Robust Fine-Tuning (RbFT), a method designed to enhance the resilience of LLMs against retrieval defects through two targeted fine-tuning tasks. Experimental results demonstrate that RbFT significantly improves the robustness of RAG systems across diverse retrieval conditions, surpassing existing methods while maintaining high inference efficiency and compatibility with other robustness techniques. 

**Abstract (ZH)**: 检索增强生成（RAG）通过将来自知识库的外部知识整合到大型语言模型（LLMs）中来增强其性能。然而，其有效性从根本上受到检索器和知识库可靠性的影响。在现实世界情境中，这些组件中的不足往往导致检索出噪声、无关或误导性的反事实信息，从而削弱了RAG系统的可信度。为了应对这一挑战，我们提出了鲁棒微调（RbFT）方法，这是一种旨在通过两个针对性的微调任务来增强LLMs对检索缺陷的抵抗力的方法。实验结果表明，RbFT 显著提高了RAG系统在各种检索条件下的鲁棒性，同时保持了高推理效率，并且与现有的鲁棒性技术兼容。 

---
# Thoughts Are All Over the Place: On the Underthinking of o1-Like LLMs 

**Title (ZH)**: 思绪纷繁杂乱：关于o1-类语言模型的过度简略思考 

**Authors**: Yue Wang, Qiuzhi Liu, Jiahao Xu, Tian Liang, Xingyu Chen, Zhiwei He, Linfeng Song, Dian Yu, Juntao Li, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2501.18585)  

**Abstract**: Large language models (LLMs) such as OpenAI's o1 have demonstrated remarkable abilities in complex reasoning tasks by scaling test-time compute and exhibiting human-like deep thinking. However, we identify a phenomenon we term underthinking, where o1-like LLMs frequently switch between different reasoning thoughts without sufficiently exploring promising paths to reach a correct solution. This behavior leads to inadequate depth of reasoning and decreased performance, particularly on challenging mathematical problems. To systematically analyze this issue, we conduct experiments on three challenging test sets and two representative open-source o1-like models, revealing that frequent thought switching correlates with incorrect responses. We introduce a novel metric to quantify underthinking by measuring token efficiency in incorrect answers. To address underthinking, we propose a decoding strategy with thought switching penalty TIP that discourages premature transitions between thoughts, encouraging deeper exploration of each reasoning path. Experimental results demonstrate that our approach improves accuracy across challenging datasets without requiring model fine-tuning. Our findings contribute to understanding reasoning inefficiencies in o1-like LLMs and offer a practical solution to enhance their problem-solving capabilities. 

**Abstract (ZH)**: 大型语言模型（LLMs）如OpenAI的o1在通过扩展测试时计算能力和展现类人深度思维的能力上，已经展现出了在复杂推理任务中的显著能力。然而，我们发现了一种我们称之为“欠思考”的现象，即o1类似的LLMs在解决问题时频繁在不同的推理思路之间切换，而未能充分探索有希望的路径以达到正确解。这种行为导致了推理深度不足和性能下降，尤其是在解决复杂的数学问题时。为了系统地分析这一问题，我们在三个具有挑战性的测试集中对两种代表性的开源o1类似模型进行了实验，结果显示频繁的思维切换与错误响应有关。我们引入了一个新的度量标准来量化“欠思考”，通过测量在错误答案中关键词的有效性来进行评估。为解决“欠思考”问题，我们提出了一种带有思维切换惩罚（TIP）的解码策略，该策略抑制过早的思维转换，鼓励深入探索每条推理路径。实验结果表明，我们的方法可以在无需对模型进行微调的情况下提高复杂数据集的准确性。我们的研究成果有助于理解o1类似LLMs在推理效率方面的问题，并提供一种实用的解决方案，以提升其问题解决能力。 

---
# Differentially Private Steering for Large Language Model Alignment 

**Title (ZH)**: 大型语言模型对齐的差异隐私指引 

**Authors**: Anmol Goel, Yaxi Hu, Iryna Gurevych, Amartya Sanyal  

**Link**: [PDF](https://arxiv.org/pdf/2501.18532)  

**Abstract**: Aligning Large Language Models (LLMs) with human values and away from undesirable behaviors (such as hallucination) has become increasingly important. Recently, steering LLMs towards a desired behavior via activation editing has emerged as an effective method to mitigate harmful generations at inference-time. Activation editing modifies LLM representations by preserving information from positive demonstrations (e.g., truthful) and minimising information from negative demonstrations (e.g., hallucinations). When these demonstrations come from a private dataset, the aligned LLM may leak private information contained in those private samples. In this work, we present the first study of aligning LLM behavior with private datasets. Our work proposes the \textit{\underline{P}rivate \underline{S}teering for LLM \underline{A}lignment (PSA)} algorithm to edit LLM activations with differential privacy (DP) guarantees. We conduct extensive experiments on seven different benchmarks with open-source LLMs of different sizes (0.5B to 7B) and model families (LlaMa, Qwen, Mistral and Gemma). Our results show that PSA achieves DP guarantees for LLM alignment with minimal loss in performance, including alignment metrics, open-ended text generation quality, and general-purpose reasoning. We also develop the first Membership Inference Attack (MIA) for evaluating and auditing the empirical privacy for the problem of LLM steering via activation editing. Our attack is tailored for activation editing and relies solely on the generated texts without their associated probabilities. Our experiments support the theoretical guarantees by showing improved guarantees for our \textit{PSA} algorithm compared to several existing non-private techniques. 

**Abstract (ZH)**: 将大型语言模型（LLMs）与人类价值观对齐，并远离不良行为（如幻觉）变得越来越重要。最近，通过激活编辑引导LLMs朝着期望的行为发展，已成为在推理时减轻有害生成的有效方法。激活编辑通过保留正面示例（例如真实的）的信息并最小化负面示例（例如幻觉）的信息来修改LLM表示。当这些示例来自私人数据集时，对齐后的LLM可能会泄露私人样本中包含的私人信息。本文首次探讨了将LLM行为与私人数据集对齐的问题。我们的工作提出了\textit{\underline{P}rivate \underline{S}teering for L\underline{L}anguage \underline{M}odel \underline{A}lignment (PSA)}算法，该算法在差分隐私（DP）保证下编辑LLM激活。我们使用不同大小（0.5B到7B）和不同模型家族（LlaMa、Qwen、Mistral和Gemma）的开源LLM在七个不同的基准上进行了广泛的实验。我们的结果表明，PSA在保持性能（包括对齐指标、开放式文本生成质量和通用推理能力）几乎无损失的情况下实现了DP保证。我们还开发了第一个评估和审计通过激活编辑引导LLM问题中的实际隐私的成员推理攻击（MIA）。我们的攻击专门针对激活编辑，仅依赖于生成的文字而不需要其相关的概率。我们的实验通过显示与几种现有非隐私技术相比\textit{PSA}算法的理论保证改善，支持了理论保证。 

---
# CALM: Unleashing the Cross-Lingual Self-Aligning Ability of Language Model Question Answering 

**Title (ZH)**: CALM：释放语言模型问答中的跨语言自我对齐能力 

**Authors**: Yumeng Wang, Zhiyuan Fan, Qingyun Wang, May Fung, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2501.18457)  

**Abstract**: Large Language Models (LLMs) are pretrained on extensive multilingual corpora to acquire both language-specific cultural knowledge and general knowledge. Ideally, while LLMs should provide consistent responses to culture-independent questions across languages, we observe significant performance disparities. To address this, we explore the Cross-Lingual Self-Aligning ability of Language Models (CALM) to align knowledge across languages. Specifically, for a given question, we sample multiple responses across different languages, and select the most self-consistent response as the target, leaving the remaining responses as negative examples. We then employ direct preference optimization (DPO) to align the model's knowledge across different languages. Evaluations on the MEDQA and X-CSQA datasets demonstrate CALM's effectiveness in enhancing cross-lingual knowledge question answering, both in zero-shot and retrieval augmented settings. We also found that increasing the number of languages involved in CALM training leads to even higher accuracy and consistency. We offer a qualitative analysis of how cross-lingual consistency can enhance knowledge alignment and explore the method's generalizability. The source code and data of this paper are available on GitHub. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过大量多语言语料库进行预训练，以获取特定文化和通用知识。理想情况下，LLMs 应该能在不同语言中提供一致的答案以回答与文化无关的问题，但我们在实践中观察到显著的性能差异。为了解决这一问题，我们探索了语言模型的跨语言自我对齐能力（Cross-Lingual Self-Aligning，简称CALM），以实现跨语言知识的对齐。具体来说，对于给定的问题，我们会在不同的语言中采样多个响应，并选择最自洽的响应作为目标，其余响应作为负样本。然后，我们使用直接偏好优化（DPO）来跨语言对齐模型的知识。在 MEDQA 和 X-CSQA 数据集上的评估表明，CALM 在零样本和检索增强设置下都有效提升了跨语言知识问答。我们还发现，在CALM 训练过程中涉及的语言数量越多，可以获得更高的准确度和一致性。我们对跨语言一致性如何增强知识对齐进行了定性分析，并探讨了该方法的通用性。本文的源代码和数据已发布在 GitHub 上。 

---
# Jailbreaking LLMs' Safeguard with Universal Magic Words for Text Embedding Models 

**Title (ZH)**: 使用通用魔法词突破LLMs的安全防护以供文本嵌入模型使用 

**Authors**: Haoyu Liang, Youran Sun, Yunfeng Cai, Jun Zhu, Bo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.18280)  

**Abstract**: The security issue of large language models (LLMs) has gained significant attention recently, with various defense mechanisms developed to prevent harmful outputs, among which safeguards based on text embedding models serve as a fundamental defense. Through testing, we discover that the distribution of text embedding model outputs is significantly biased with a large mean. Inspired by this observation, we propose novel efficient methods to search for universal magic words that can attack text embedding models. The universal magic words as suffixes can move the embedding of any text towards the bias direction, therefore manipulate the similarity of any text pair and mislead safeguards. By appending magic words to user prompts and requiring LLMs to end answers with magic words, attackers can jailbreak the safeguard. To eradicate this security risk, we also propose defense mechanisms against such attacks, which can correct the biased distribution of text embeddings in a train-free manner. 

**Abstract (ZH)**: 大型语言模型（LLMs）的安全问题最近引起了广泛关注，开发出了各种防御机制以防止产生有害输出，其中基于文本嵌入模型的安全保障成为基础性的防御手段。通过测试，我们发现文本嵌入模型的输出分布存在显著偏差，且具有较大的均值。基于这一观察，我们提出了新的高效方法来寻找通用的“魔法词”，这些“魔法词”能够攻击文本嵌入模型。作为后缀的“魔法词”可以将任何文本的嵌入向偏差方向移动，从而操控任何文本对之间的相似度，并误导安全保障。通过在用户提示中附加“魔法词”，并在要求LLMs以“魔法词”结尾的条件下生成答案，攻击者可以突破安全保障。为消除这一安全风险，我们还提出了一种无需训练的防御机制，该机制能够以无监督的方式纠正文本嵌入的偏差分布。 

---
# Contextually Structured Token Dependency Encoding for Large Language Models 

**Title (ZH)**: 面向上下文的结构化词元依赖编码在大规模语言模型中的应用 

**Authors**: James Blades, Frederick Somerfield, William Langley, Susan Everingham, Maurice Witherington  

**Link**: [PDF](https://arxiv.org/pdf/2501.18205)  

**Abstract**: Token representation strategies within large-scale neural architectures often rely on contextually refined embeddings, yet conventional approaches seldom encode structured relationships explicitly within token interactions. Self-attention mechanisms effectively capture dynamic contextual dependencies, but their reliance on learned weight distributions limits the preservation of long-range hierarchical structures in generated sequences. Dependency-aware token encoding introduces a structured approach to embedding initialization, ensuring that relational constraints are embedded within token representations rather than inferred solely through attention dynamics. The proposed encoding mechanism refines token interactions through dependency-weighted attention computations, ensuring that syntactic and semantic dependencies are retained across multiple processing layers. Empirical evaluations indicate reductions in perplexity across diverse linguistic benchmarks, suggesting improvements in contextual coherence and predictive consistency in autoregressive text generation. Computational efficiency assessments reveal a moderate increase in memory consumption and training time, attributed to additional matrix computations within the encoding module, yet scalability remains feasible within conventional transformer architectures. Structured encoding enhances lexical variation and dependency retention, reinforcing linguistic coherence without requiring external syntactic annotations or auxiliary training objectives. Statistical comparisons highlight improvements in dependency alignment, particularly in longer sequences where conventional self-attention models exhibit degradation in hierarchical consistency. Sentence length distributions indicate a reduction in abrupt phrase transitions, further supporting the hypothesis that explicit dependency encoding facilitates more structured phrase generation. 

**Abstract (ZH)**: 大规模神经架构中的令牌表示策略通常依赖于上下文精炼的嵌入表示，但传统方法很少在令牌交互中显式地编码结构化的关系。自我注意机制能够有效捕获动态上下文依赖关系，但它们对学习权重分布的高度依赖限制了在生成序列中长距离层次结构结构的保留。依赖关系感知的令牌编码引入了一种结构化的嵌入初始化方法，确保关系约束在令牌表示中嵌入，而不仅仅是通过注意动态推断。所提出的编码机制通过依赖加权注意力计算细化令牌交互，确保在多个处理层中保留语法和语义依赖关系。实证评估表明，在多种语言基准上困惑度的降低，这表明在自回归文本生成中上下文连贯性和预测一致性有所提高。计算效率评估显示内存消耗和训练时间有适度增加，这归因于编码模块内的额外矩阵计算，但结构化编码在传统变压器架构中的可扩展性仍然可行。结构化编码增强了词汇变异性和依赖关系保留，无需外部句法注释或辅助训练目标，即可增强语言连贯性。统计比较突显了在较长序列中的依赖对齐改进，特别是在传统自我注意模型表现退化的场合。句子长度分布表明语句不自然转变的减少，进一步支持显式依赖编码有助于更结构化短语生成的假设。 

---
# Self-supervised Quantized Representation for Seamlessly Integrating Knowledge Graphs with Large Language Models 

**Title (ZH)**: 自我监督量化表示以无缝整合知识图谱与大规模语言模型 

**Authors**: Qika Lin, Tianzhe Zhao, Kai He, Zhen Peng, Fangzhi Xu, Ling Huang, Jingying Ma, Mengling Feng  

**Link**: [PDF](https://arxiv.org/pdf/2501.18119)  

**Abstract**: Due to the presence of the natural gap between Knowledge Graph (KG) structures and the natural language, the effective integration of holistic structural information of KGs with Large Language Models (LLMs) has emerged as a significant question. To this end, we propose a two-stage framework to learn and apply quantized codes for each entity, aiming for the seamless integration of KGs with LLMs. Firstly, a self-supervised quantized representation (SSQR) method is proposed to compress both KG structural and semantic knowledge into discrete codes (\ie, tokens) that align the format of language sentences. We further design KG instruction-following data by viewing these learned codes as features to directly input to LLMs, thereby achieving seamless integration. The experiment results demonstrate that SSQR outperforms existing unsupervised quantized methods, producing more distinguishable codes. Further, the fine-tuned LLaMA2 and LLaMA3.1 also have superior performance on KG link prediction and triple classification tasks, utilizing only 16 tokens per entity instead of thousands in conventional prompting methods. 

**Abstract (ZH)**: 由于知识图谱（KG）结构与自然语言之间存在天然差距，如何有效地将KG的全面结构信息与大型语言模型（LLMs）结合起来成为一个重要的问题。为了解决这一问题，我们提出了一种两阶段框架，旨在学习和应用每个实体的量化代码，实现KG与LLMs的无缝集成。首先，我们提出了自监督量化表示（SSQR）方法，将KG的结构知识和语义知识压缩成离散的代码（即，标记），以匹配语言句子的格式。在此基础上，我们进一步设计了KG指示遵循数据，将这些学习到的代码作为特征直接输入到LLMs中，从而实现无缝集成。实验结果表明，SSQR方法在现有无监督量化方法中表现更优，生成了更为可区分的代码。此外，微调的LLaMA2和LLaMA3.1在KG链接预测和三元组分类任务中也表现出色，仅使用每实体16个标记，而传统的提示方法中则需要成千上万的标记。 

---
# Panacea: Mitigating Harmful Fine-tuning for Large Language Models via Post-fine-tuning Perturbation 

**Title (ZH)**: 泛药方：通过后微调扰动缓解大型语言模型的有害微调问题 

**Authors**: Yibo Wang, Tiansheng Huang, Li Shen, Huanjin Yao, Haotian Luo, Rui Liu, Naiqiang Tan, Jiaxing Huang, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2501.18100)  

**Abstract**: Harmful fine-tuning attack introduces significant security risks to the fine-tuning services. Mainstream defenses aim to vaccinate the model such that the later harmful fine-tuning attack is less effective. However, our evaluation results show that such defenses are fragile -- with a few fine-tuning steps, the model still can learn the harmful knowledge. To this end, we do further experiment and find that an embarrassingly simple solution -- adding purely random perturbations to the fine-tuned model, can recover the model from harmful behavior, though it leads to a degradation in the model's fine-tuning performance. To address the degradation of fine-tuning performance, we further propose Panacea, which optimizes an adaptive perturbation that will be applied to the model after fine-tuning. Panacea maintains model's safety alignment performance without compromising downstream fine-tuning performance. Comprehensive experiments are conducted on different harmful ratios, fine-tuning tasks and mainstream LLMs, where the average harmful scores are reduced by up-to 21.5%, while maintaining fine-tuning performance. As a by-product, we analyze the optimized perturbation and show that different layers in various LLMs have distinct safety coefficients. Source code available at this https URL 

**Abstract (ZH)**: 有害微调攻击对微调服务带来了显著的安全风险。主流防御措施旨在“接种”模型，使得后续的有害微调攻击效果降低。然而，我们的评估结果表明，这些防御措施是脆弱的——只需几次微调步骤，模型仍然可以学习到有害知识。为了解决这一问题，我们进一步进行了实验，并发现一个极为简单的解决方案——向微调后的模型添加纯粹随机扰动，可以在一定程度上恢复模型的行为，尽管这会导致微调性能的下降。为了应对微调性能的下降，我们进一步提出了Panacea，它优化了一个自适应的扰动，该扰动将在微调后应用于模型。Panacea在保持模型安全对齐性能的同时，也不牺牲下游微调性能。我们在不同有害比例、微调任务和主流大语言模型（LLM）上进行了全面的实验，结果显示有害评分平均降低了21.5%，而微调性能得以保持。作为副产品，我们分析了优化的扰动，并展示了各种LLM的不同层具有不同的安全系数。源代码可在以下链接获取：this https URL 

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
# State Stream Transformer (SST) : Emergent Metacognitive Behaviours Through Latent State Persistence 

**Title (ZH)**: 状态流变换器（SST）：通过潜在状态持久化实现 emergent 元认知行为 

**Authors**: Thea Aviss  

**Link**: [PDF](https://arxiv.org/pdf/2501.18356)  

**Abstract**: We introduce the State Stream Transformer (SST), a novel LLM architecture that reveals emergent reasoning behaviours and capabilities latent in pretrained weights through addressing a fundamental limitation in traditional transformer models: the lack of latent computational continuity across autoregressive generations in the state space. SST introduces a sliding window latent state (FFN) cache with weighted decay that maintains and evolves persistent latent processes throughout autoregressive generations. Through controlled experiments comparing base and SST architectures using the same frozen weights, we demonstrate that this architectural modification alone enables enhanced reasoning capabilities which appear best explained by some form of potential higher-order processing, as evidenced by emergent metacognitive behaviours. These behaviours persist under controlled conditions designed to eliminate confounding factors such as stochastic variation or learned response patterns. Analysis of latent state distributions and processing dynamics provides evidence that it is solely the 'state stream' that is responsible for these phenomena. In quantitative evaluations, the SST achieves substantial performance improvements over the base model on two reasoning benchmarks, reaching 89.01\% accuracy on GSM-8K (0-shot) and 91.04\% on ARC Challenge (0-shot CoT). These findings indicate that persistent computation in the latent state space enables fundamentally different information processing and internal reasoning strategies, with implications for our understanding of artificial intelligence systems. 

**Abstract (ZH)**: 我们引入了State Stream Transformer (SST)，这是一种新型的大型语言模型（LLM）架构，通过解决传统变压器模型的基本限制——状态空间中自回归生成过程中缺乏潜在的连续计算能力——揭示了预训练权重中潜藏的新兴推理行为和能力。SST 引入了一个具有加权衰减的滑动窗口潜在状态（FFN）缓存，以保持并发展自回归生成过程中的持久潜在过程。通过使用相同的冻结权重进行控制实验，我们将基本架构和SST架构进行比较，证明这种架构修改本身就使模型具备了增强的推理能力，这些能力似乎是某种形式的潜在高阶处理所解释的，这些行为在旨在消除混淆因素（如随机波动或学习到的响应模式）的设计条件下仍然持续存在。对潜在状态分布和处理动态的分析显示，正是“状态流”导致了这些现象。在定量评估中，SST 在两个推理基准测试中比基线模型取得了显著的性能提升，在 GSM-8K（零样本）测试中达到了 89.01% 的准确率，在 ARC 挑战赛（零样本推理）中达到了 91.04%。这些发现表明，潜在状态空间中的持续计算能够实现根本不同的信息处理和内部推理策略，这对理解人工智能系统具有重要意义。 

---
# Statistical multi-metric evaluation and visualization of LLM system predictive performance 

**Title (ZH)**: 统计多指标评估与可视化LLM系统预测性能 

**Authors**: Samuel Ackerman, Eitan Farchi, Orna Raz, Assaf Toledo  

**Link**: [PDF](https://arxiv.org/pdf/2501.18243)  

**Abstract**: The evaluation of generative or discriminative large language model (LLM)-based systems is often a complex multi-dimensional problem. Typically, a set of system configuration alternatives are evaluated on one or more benchmark datasets, each with one or more evaluation metrics, which may differ between datasets. We often want to evaluate -- with a statistical measure of significance -- whether systems perform differently either on a given dataset according to a single metric, on aggregate across metrics on a dataset, or across datasets. Such evaluations can be done to support decision-making, such as deciding whether a particular system component change (e.g., choice of LLM or hyperparameter values) significantly improves performance over the current system configuration, or, more generally, whether a fixed set of system configurations (e.g., a leaderboard list) have significantly different performances according to metrics of interest. We present a framework implementation that automatically performs the correct statistical tests, properly aggregates the statistical results across metrics and datasets (a nontrivial task), and can visualize the results. The framework is demonstrated on the multi-lingual code generation benchmark CrossCodeEval, for several state-of-the-art LLMs. 

**Abstract (ZH)**: 基于生成式或判别式大型语言模型（LLM）的系统评估往往是多维度的复杂问题。通常，会对系统配置的不同选择进行评估，这些评估可能在单一基准数据集上使用一个或多个评估指标，或者在多个指标上评估同一个数据集，或者跨越不同的数据集进行评估。这些评估可以用于支持决策，例如，判断某个特定的系统组件更改（例如，选择不同的LLM或超参数值）是否显著地提高了当前系统配置的性能，或者更一般地，判断一系列固定系统配置（例如，排行榜列表）在感兴趣指标上的性能是否存在显著差异。我们提出了一种框架实现，该框架能够自动执行正确的统计检验，并正确地跨指标和数据集汇总统计结果（这是一个非平凡的任务），并且可以可视化结果。该框架已在多语言代码生成基准CrossCodeEval上对几种最先进的LLM进行了演示。 

---
# Learning to Plan & Reason for Evaluation with Thinking-LLM-as-a-Judge 

**Title (ZH)**: 学习规划与推理以 Thinking-LLM-as-a-Judge 进行评估 

**Authors**: Swarnadeep Saha, Xian Li, Marjan Ghazvininejad, Jason Weston, Tianlu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.18099)  

**Abstract**: LLM-as-a-Judge models generate chain-of-thought (CoT) sequences intended to capture the step-bystep reasoning process that underlies the final evaluation of a response. However, due to the lack of human annotated CoTs for evaluation, the required components and structure of effective reasoning traces remain understudied. Consequently, previous approaches often (1) constrain reasoning traces to hand-designed components, such as a list of criteria, reference answers, or verification questions and (2) structure them such that planning is intertwined with the reasoning for evaluation. In this work, we propose EvalPlanner, a preference optimization algorithm for Thinking-LLM-as-a-Judge that first generates an unconstrained evaluation plan, followed by its execution, and then the final judgment. In a self-training loop, EvalPlanner iteratively optimizes over synthetically constructed evaluation plans and executions, leading to better final verdicts. Our method achieves a new state-of-the-art performance for generative reward models on RewardBench (with a score of 93.9), despite being trained on fewer amount of, and synthetically generated, preference pairs. Additional experiments on other benchmarks like RM-Bench, JudgeBench, and FollowBenchEval further highlight the utility of both planning and reasoning for building robust LLM-as-a-Judge reasoning models. 

**Abstract (ZH)**: 大规模语言模型作为法官（LLM-as-a-Judge）模型生成包含推理过程的链式思考（CoT）序列，旨在捕捉最终回答评估背后的逐步推理过程。然而，由于缺乏经过人工标注的CoT进行评估，有效的推理痕迹所必需的成分和结构仍然研究不足。因此，以往的方法往往（1）限制推理痕迹到手设计的成分，如标准列表、参考答案或验证问题，（2）将规划与评估推理交织在一起。在本文中，我们提出了EvalPlanner，一种用于Thinking-LLM-as-a-Judge的偏好优化算法，该算法首先生成不加约束的评估计划，然后执行，最后得出最终判决。在自我训练循环中，EvalPlanner逐步优化合成构建的评估计划与执行，从而获得更好的最终裁决结果。尽管仅在较少的真实偏好配对和合成生成的偏好配对进行训练，我们的方法仍然在RewardBench上取得了新的最佳性能（得分为93.9）。此外，在其他基准测试，如RM-Bench、JudgeBench和FollowBenchEval的实验进一步突显了规划和推理在构建稳健的LLM-as-a-Judge推理模型中的作用。 

---
# LLMs can see and hear without any training 

**Title (ZH)**: 大语言模型无需训练即可具备视觉和听觉能力 

**Authors**: Kumar Ashutosh, Yossi Gandelsman, Xinlei Chen, Ishan Misra, Rohit Girdhar  

**Link**: [PDF](https://arxiv.org/pdf/2501.18096)  

**Abstract**: We present MILS: Multimodal Iterative LLM Solver, a surprisingly simple, training-free approach, to imbue multimodal capabilities into your favorite LLM. Leveraging their innate ability to perform multi-step reasoning, MILS prompts the LLM to generate candidate outputs, each of which are scored and fed back iteratively, eventually generating a solution to the task. This enables various applications that typically require training specialized models on task-specific data. In particular, we establish a new state-of-the-art on emergent zero-shot image, video and audio captioning. MILS seamlessly applies to media generation as well, discovering prompt rewrites to improve text-to-image generation, and even edit prompts for style transfer! Finally, being a gradient-free optimization approach, MILS can invert multimodal embeddings into text, enabling applications like cross-modal arithmetic. 

**Abstract (ZH)**: 我们介绍了MILS：一种多模态迭代LLM求解器，这是一种出人意料地简单且无需训练的方法，将多模态能力注入您最喜欢的LLM。利用其天生的多步骤推理能力，MILS 提示LLM生成候选输出，每个输出都会被打分并反馈给下一个迭代过程，最终生成任务的解决方案。这使得许多通常需要在特定任务数据上训练专门模型的应用成为可能。具体而言，我们在这个领域取得了新的最佳成果，特别是在新兴的零样本图像、视频和音频描述方面。MILS 无缝应用于媒体生成，通过发现提示重写来改进文本到图像生成，并且甚至可以编辑提示进行风格转换！最后，作为无梯度优化方法，MILS 可以将多模态嵌入反转为文本，从而支持跨模态算术等应用。 

---
# DReSS: Data-driven Regularized Structured Streamlining for Large Language Models 

**Title (ZH)**: DReSS：面向数据驱动的正则化结构化流水线精简方法用于大型语言模型

在这个翻译中，我尽量保持了原文的技术术语和学术规范。"DReSS"被保留为原名，因为它是论文的缩写名称。其他部分如“数据驱动的正则化结构化流水线精简方法”和“大型语言模型”都是学术化的表达方式。 

**Authors**: Mingkuan Feng, Jinyang Wu, Shuai Zhang, Pengpeng Shao, Ruihan Jin, Zhengqi Wen, Jianhua Tao, Feihu Che  

**Link**: [PDF](https://arxiv.org/pdf/2501.17905)  

**Abstract**: Large language models (LLMs) have achieved significant progress across various domains, but their increasing scale results in high computational and memory costs. Recent studies have revealed that LLMs exhibit sparsity, providing the potential to reduce model size through pruning techniques. However, existing pruning methods typically follow a prune-then-finetune paradigm. Since the pruned components still contain valuable information, their direct removal often leads to irreversible performance degradation, imposing a substantial computational burden to recover performance during finetuning. In this paper, we propose a novel paradigm that first applies regularization, then prunes, and finally finetunes. Based on this paradigm, we introduce DReSS, a simple and effective Data-driven Regularized Structured Streamlining method for LLMs. By leveraging a small amount of data to regularize the components to be pruned, DReSS explicitly transfers the important information to the remaining parts of the model in advance. Compared to direct pruning, this can reduce the information loss caused by parameter removal, thereby enhancing its language modeling capabilities. Experimental results demonstrate that DReSS significantly outperforms existing pruning methods even under extreme pruning ratios, significantly reducing latency and increasing throughput. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在各个领域取得了显著进展，但随着模型规模的增加，其计算和内存成本也大幅提升。近期研究发现，LLMs表现出稀疏性，为通过剪枝技术减小模型大小提供了可能性。然而，现有的剪枝方法通常遵循先剪枝后微调的范式。由于被剪枝的部分仍然包含有价值的信息，直接移除这些部分往往会导致不可逆的性能下降，从而在微调过程中给恢复性能带来巨大的计算负担。本文提出了一种新的范式：先应用正则化，再进行剪枝，最后进行微调。基于这一范式，我们引入了DReSS，一种简单且有效的大规模语言模型数据驱动正则化结构化剪枝方法。通过利用少量数据对将要剪枝的组件进行正则化，DReSS可以提前将重要信息转移到模型剩余部分。与直接剪枝相比，这种方法可以减少因参数移除而造成的信息损失，从而增强其语言建模能力。实验结果表明，即使在极端剪枝比的情况下，DReSS也显著优于现有的剪枝方法，显著降低了延迟并提高了吞吐量。 

---
# Leveraging LLM Agents for Automated Optimization Modeling for SASP Problems: A Graph-RAG based Approach 

**Title (ZH)**: 利用大语言模型代理进行SASP问题的自动化优化建模：基于Graph-RAG的方法 

**Authors**: Tianpeng Pan, Wenqiang Pu, Licheng Zhao, Rui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2501.18320)  

**Abstract**: Automated optimization modeling (AOM) has evoked considerable interest with the rapid evolution of large language models (LLMs). Existing approaches predominantly rely on prompt engineering, utilizing meticulously designed expert response chains or structured guidance. However, prompt-based techniques have failed to perform well in the sensor array signal processing (SASP) area due the lack of specific domain knowledge. To address this issue, we propose an automated modeling approach based on retrieval-augmented generation (RAG) technique, which consists of two principal components: a multi-agent (MA) structure and a graph-based RAG (Graph-RAG) process. The MA structure is tailored for the architectural AOM process, with each agent being designed based on principles of human modeling procedure. The Graph-RAG process serves to match user query with specific SASP modeling knowledge, thereby enhancing the modeling result. Results on ten classical signal processing problems demonstrate that the proposed approach (termed as MAG-RAG) outperforms several AOM benchmarks. 

**Abstract (ZH)**: 自动化优化建模（AOM）随着大型语言模型（LLMs）的迅速发展引起了广泛的关注。现有的方法主要依赖于提示工程，利用精心设计的专家响应链或结构化的指导。然而，在传感器阵列信号处理（SASP）领域，提示基础的方法由于缺乏特定领域的知识而表现不佳。为了解决这一问题，我们提出了一种基于检索增强生成（RAG）技术的自动化建模方法，该方法主要包括两个主要组成部分：一个多代理（MA）结构和基于图的RAG（Graph-RAG）过程。多代理结构专门针对架构化的AOM流程，其中每个代理都是基于人类建模过程的原则设计的。基于图的RAG过程用于匹配用户查询与特定的SASP建模知识，从而提高建模结果。在十个经典信号处理问题上的实验结果显示，所提出的方法（称为MAG-RAG）在几个AOM基准中表现更优。 

---
# Normative Evaluation of Large Language Models with Everyday Moral Dilemmas 

**Title (ZH)**: 使用日常道德困境对大型语言模型进行规范性评估 

**Authors**: Pratik S. Sachdeva, Tom van Nuenen  

**Link**: [PDF](https://arxiv.org/pdf/2501.18081)  

**Abstract**: The rapid adoption of large language models (LLMs) has spurred extensive research into their encoded moral norms and decision-making processes. Much of this research relies on prompting LLMs with survey-style questions to assess how well models are aligned with certain demographic groups, moral beliefs, or political ideologies. While informative, the adherence of these approaches to relatively superficial constructs tends to oversimplify the complexity and nuance underlying everyday moral dilemmas. We argue that auditing LLMs along more detailed axes of human interaction is of paramount importance to better assess the degree to which they may impact human beliefs and actions. To this end, we evaluate LLMs on complex, everyday moral dilemmas sourced from the "Am I the Asshole" (AITA) community on Reddit, where users seek moral judgments on everyday conflicts from other community members. We prompted seven LLMs to assign blame and provide explanations for over 10,000 AITA moral dilemmas. We then compared the LLMs' judgments and explanations to those of Redditors and to each other, aiming to uncover patterns in their moral reasoning. Our results demonstrate that large language models exhibit distinct patterns of moral judgment, varying substantially from human evaluations on the AITA subreddit. LLMs demonstrate moderate to high self-consistency but low inter-model agreement. Further analysis of model explanations reveals distinct patterns in how models invoke various moral principles. These findings highlight the complexity of implementing consistent moral reasoning in artificial systems and the need for careful evaluation of how different models approach ethical judgment. As LLMs continue to be used in roles requiring ethical decision-making such as therapists and companions, careful evaluation is crucial to mitigate potential biases and limitations. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的快速普及已经推动了对它们编码道德规范和决策过程的广泛研究。这些研究中的许多都依赖于用调查式问题提示LLMs，以评估模型与特定人口群体、道德信念或政治意识形态的对齐程度。尽管这些方法具有信息价值，但它们对相对表面的构建物的依从性往往会简化日常生活中的道德困境的复杂性和细微之处。我们主张，沿更详细的人类互动轴审计LLMs对于更好地评估它们可能对人类信念和行为的影响至关重要。为了实现这一目标，我们评估了来自“Am I the Asshole”（AITA）社区（Reddit用户在此寻求其他社区成员对日常冲突的道德判断）的复杂且常见的道德困境中的LLMs。我们提示了七种LLMs为超过10,000个AITA道德困境分配责任并提供解释。然后我们将LLMs的判断和解释与 redditors 的判断以及彼此之间的判断进行了比较，旨在揭示它们在道德推理中的模式。我们的结果显示，大规模语言模型在处理AITA子网页上的道德判断时表现出不同的模式，与人类评价相差甚远。LLMs在自我一致性方面表现出中等到高的水平，但在模型之间的一致性较低。进一步分析模型解释揭示了模型在运用各种道德原则时存在的不同模式。这些发现突显了在人工系统中实现一致道德推理的复杂性，并强调了仔细评估不同模型在伦理判断方面的方法的必要性。随着LLMs在需要伦理决策角色，如治疗师和伴侣等方面的应用，需要仔细评估以减轻潜在的偏见和限制。 

---
# Think Smarter not Harder: Adaptive Reasoning with Inference Aware Optimization 

**Title (ZH)**: 更聪明地思考，而不是更加努力地思考：基于推理意识的适应性推理优化 

**Authors**: Zishun Yu, Tengyu Xu, Di Jin, Karthik Abinav Sankararaman, Yun He, Wenxuan Zhou, Zhouhao Zeng, Eryk Helenowski, Chen Zhu, Sinong Wang, Hao Ma, Han Fang  

**Link**: [PDF](https://arxiv.org/pdf/2501.17974)  

**Abstract**: Solving mathematics problems has been an intriguing capability of large language models, and many efforts have been made to improve reasoning by extending reasoning length, such as through self-correction and extensive long chain-of-thoughts. While promising in problem-solving, advanced long reasoning chain models exhibit an undesired single-modal behavior, where trivial questions require unnecessarily tedious long chains of thought. In this work, we propose a way to allow models to be aware of inference budgets by formulating it as utility maximization with respect to an inference budget constraint, hence naming our algorithm Inference Budget-Constrained Policy Optimization (IBPO). In a nutshell, models fine-tuned through IBPO learn to ``understand'' the difficulty of queries and allocate inference budgets to harder ones. With different inference budgets, our best models are able to have a $4.14$\% and $5.74$\% absolute improvement ($8.08$\% and $11.2$\% relative improvement) on MATH500 using $2.16$x and $4.32$x inference budgets respectively, relative to LLaMA3.1 8B Instruct. These improvements are approximately $2$x those of self-consistency under the same budgets. 

**Abstract (ZH)**: 解决数学问题一直是大型语言模型的一种令人着迷的能力，许多努力已经致力于通过扩展推理长度来提升推理能力，例如通过自我修正和广泛的长链条逻辑推理。虽然在解决问题方面表现出色，但先进的长推理链模型却表现出一种不希望看到的单一模态行为，即简单的问题需要不必要的冗长链条的推理过程。在此项工作中，我们提出了一种方法，通过将推理预算视为约束下的效用最大化问题，从而使模型具备意识推理预算的能力，因此将我们的算法命名为推理预算约束策略优化（IBPO）。简而言之，通过IBPO微调的模型学会了“理解”问题的难度，并将推理预算分配给更难的问题。通过不同的推理预算，我们的最佳模型在使用2.16倍和4.32倍推理预算的情况下，相对于LLaMA3.1 8B Instruct，在MATH500上分别实现了4.14%和5.74%的绝对改进（相对改进分别为8.08%和11.2%）。这些改进大致是相同预算下自我一致性策略的两倍。 

---
# DeltaLLM: Compress LLMs with Low-Rank Deltas between Shared Weights 

**Title (ZH)**: DeltaLLM：通过共享权重之间的低秩增量压缩大语言模型 

**Authors**: Liana Mikaelyan, Ayyoob Imani, Mathew Salvaris, Parth Pathak, Mohsen Fayyaz  

**Link**: [PDF](https://arxiv.org/pdf/2501.18596)  

**Abstract**: We introduce DeltaLLM, a new post-training compression technique to reduce the memory footprint of LLMs. We propose an alternative way of structuring LLMs with weight sharing between layers in subsequent Transformer blocks, along with additional low-rank difference matrices between them. For training, we adopt the progressing module replacement method and show that the lightweight training of the low-rank modules with approximately 30M-40M tokens is sufficient to achieve performance on par with LLMs of comparable sizes trained from scratch. We release the resultant models, DeltaLLAMA and DeltaPHI, with a 12% parameter reduction, retaining 90% of the performance of the base Llama and Phi models on common knowledge and reasoning benchmarks. Our method also outperforms compression techniques JointDrop, LaCo, ShortGPT and SliceGPT with the same number of parameters removed. For example, DeltaPhi 2.9B with a 24% reduction achieves similar average zero-shot accuracies as recovery fine-tuned SlicedPhi 3.3B with a 12% reduction, despite being approximately 400M parameters smaller with no fine-tuning applied. This work provides new insights into LLM architecture design and compression methods when storage space is critical. 

**Abstract (ZH)**: 我们提出了DeltaLLM，这是一种新的后训练压缩技术，用于减少大规模语言模型（LLM）的内存占用。我们提出了一种替代的LLM结构化方法，在后续的Transformer块之间实现权重共享，并且在它们之间附加了低秩差异矩阵。在训练过程中，我们采用了逐步模块替换方法，并展示出使用约30M-40M个令牌训练的低秩模块的轻量化训练足以达到与从头训练的相似大小的LLM相当的性能。我们发布了DeltaLLAMA和DeltaPHI两个模型，这两个模型的参数减少了12%，并且在常见的知识和推理基准测试中保留了基线Llama和Phi模型90%的性能。我们的方法在参数减少相同数量的情况下也优于JointDrop、LaCo、ShortGPT和SliceGPT等压缩技术。例如，DeltaPhi 2.9B参数减少24%，在未进行微调的情况下，其平均零样本准确性与恢复微调的SlicePhi 3.3B（参数减少12%）相当，尽管前者比后者大约小400M个参数。这项工作为在存储空间受限的情况下提供了关于LLM架构设计和压缩方法的新见解。 

---
# Can we Retrieve Everything All at Once? ARM: An Alignment-Oriented LLM-based Retrieval Method 

**Title (ZH)**: 当然可以。以下是翻译后的标题和内容，符合学术规范：

标题：一次性检索一切可能吗？ARM：一种基于LLM的对齐导向检索方法

内容摘要：本文探讨了一种基于大规模语言模型（LLM）的对齐导向检索方法，提出了一个问题：我们能否一次性检索到所有相关信息？提出的ARM方法旨在通过有效对齐查询与文档之间的关系，提高检索的准确性和效率。 

**Authors**: Peter Baile Chen, Yi Zhang, Michael Cafarella, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2501.18539)  

**Abstract**: Real-world open-domain questions can be complicated, particularly when answering them involves information from multiple information sources. LLMs have demonstrated impressive performance in decomposing complex tasks into simpler steps, and previous work has used it for better retrieval in support of complex questions. However, LLM's decomposition of questions is unaware of what data is available and how data is organized, often leading to a sub-optimal retrieval performance. Recent effort in agentic RAG proposes to perform retrieval in an iterative fashion, where a followup query is derived as an action based on previous rounds of retrieval. While this provides one way of interacting with the data collection, agentic RAG's exploration of data is inefficient because successive queries depend on previous results rather than being guided by the organization of available data in the collection. To address this problem, we propose an LLM-based retrieval method -- ARM, that aims to better align the question with the organization of the data collection by exploring relationships among data objects beyond matching the utterance of the query, thus leading to a retrieve-all-at-once solution for complex queries. We evaluated ARM on two datasets, Bird and OTT-QA. On Bird, it outperforms standard RAG with query decomposition by up to 5.2 pt in execution accuracy and agentic RAG (ReAct) by up to 15.9 pt. On OTT-QA, it achieves up to 5.5 pt and 19.3 pt higher F1 match scores compared to these approaches. 

**Abstract (ZH)**: 现实世界中的开放领域问题可能相当复杂，尤其是在回答这些问题时需要从多个信息源中获取信息。大规模语言模型（LLMs）在将复杂任务分解为更简单步骤方面表现出了令人印象深刻的性能，之前的许多工作也利用这一点来提高复杂问题检索的支持效果。然而，LLM对问题的分解并没有意识到可用数据及其组织方式，常常会导致检索性能不佳。最近在代理型检索-回答框架（RAG）方面的工作提出了一种迭代的检索方式，其中后续查询是基于上一轮检索结果而产生的动作。虽然这种方法提供了一种与数据集交互的方式，但代理型RAG探索数据的方式效率较低，因为后续查询依赖于上一轮的结果而非数据集可用数据的组织方式。为了解决这一问题，我们提出了一种基于LLM的检索方法——ARM，该方法旨在通过探索数据对象之间的关系来更好地将问题与数据集的组织方式相匹配，从而为复杂问题提供一次性检索所有信息的解决方案。我们在两个数据集——Bird和OTT-QA上对ARM进行了评估。在Bird数据集上，ARM在执行准确性上比标准RAG（含查询分解）高出最多5.2个百分点，在与代理型RAG（ReAct）相比时高出最多15.9个百分点。在OTT-QA数据集上，ARM分别在F1匹配分数上比这些方法高出5.5个百分点和19.3个百分点。 

---
# CLEAR: Cue Learning using Evolution for Accurate Recognition Applied to Sustainability Data Extraction 

**Title (ZH)**: CLEAR：面向可持续性数据提取的进化学习线索学习及其准确识别应用 

**Authors**: Peter J. Bentley, Soo Ling Lim, Fuyuki Ishikawa  

**Link**: [PDF](https://arxiv.org/pdf/2501.18504)  

**Abstract**: Large Language Model (LLM) image recognition is a powerful tool for extracting data from images, but accuracy depends on providing sufficient cues in the prompt - requiring a domain expert for specialized tasks. We introduce Cue Learning using Evolution for Accurate Recognition (CLEAR), which uses a combination of LLMs and evolutionary computation to generate and optimize cues such that recognition of specialized features in images is improved. It achieves this by auto-generating a novel domain-specific representation and then using it to optimize suitable textual cues with a genetic algorithm. We apply CLEAR to the real-world task of identifying sustainability data from interior and exterior images of buildings. We investigate the effects of using a variable-length representation compared to fixed-length and show how LLM consistency can be improved by refactoring from categorical to real-valued estimates. We show that CLEAR enables higher accuracy compared to expert human recognition and human-authored prompts in every task with error rates improved by up to two orders of magnitude and an ablation study evincing solution concision. 

**Abstract (ZH)**: 大规模语言模型（LLM）图片识别是一种强大的工具，可用于从图像中提取数据，但其准确性取决于在提示中提供足够的线索，这需要特定领域的专家来完成特殊任务。本文介绍了结合进化计算的准确识别（Cue Learning using Evolution for Accurate Recognition, CLEAR）方法，该方法利用了一种组合L大型语言模型和进化计算的方法来生成和优化线索，从而提高图像中特有特征的识别能力。它通过自动生成一种特定领域的新型表示，然后使用遗传算法优化合适的文本线索来实现这一点。我们把CLEAR应用于从建筑物内外图像中识别可持续性数据的实际任务。我们研究了使用可变长度表示与固定长度表示相比的影响，并展示了通过从分类估计重构为实数值估计如何提高LLM的一致性。我们证明了CLEAR在所有任务中相较于专家人工识别和人工撰写的提示具有更高的准确性，错误率最多可降低两个数量级，并通过消融研究展示了解决方案的简洁性。 

---
# GuardReasoner: Towards Reasoning-based LLM Safeguards 

**Title (ZH)**: GuardReasoner：基于推理的LLM安全保障研究 

**Authors**: Yue Liu, Hongcheng Gao, Shengfang Zhai, Jun Xia, Tianyi Wu, Zhiwei Xue, Yulin Chen, Kenji Kawaguchi, Jiaheng Zhang, Bryan Hooi  

**Link**: [PDF](https://arxiv.org/pdf/2501.18492)  

**Abstract**: As LLMs increasingly impact safety-critical applications, ensuring their safety using guardrails remains a key challenge. This paper proposes GuardReasoner, a new safeguard for LLMs, by guiding the guard model to learn to reason. Concretely, we first create the GuardReasonerTrain dataset, which consists of 127K samples with 460K detailed reasoning steps. Then, we introduce reasoning SFT to unlock the reasoning capability of guard models. In addition, we present hard sample DPO to further strengthen their reasoning ability. In this manner, GuardReasoner achieves better performance, explainability, and generalizability. Extensive experiments and analyses on 13 benchmarks of 3 guardrail tasks demonstrate its superiority. Remarkably, GuardReasoner 8B surpasses GPT-4o+CoT by 5.74% and LLaMA Guard 3 8B by 20.84% F1 score on average. We release the training data, code, and models with different scales (1B, 3B, 8B) of GuardReasoner : this https URL. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在安全关键应用中的影响日益增大，确保它们的安全性仍然是一个关键挑战。本文提出了一种新的大型语言模型安全措施GuardReasoner，通过引导保护模型学习推理能力来解决这一问题。具体而言，我们首先创建了GuardReasonerTrain数据集，该数据集包含127,000个样本和460,000个详细的推理步骤。然后，我们引入了推理样本微调（reasoning SFT），以解锁保护模型的推理能力。此外，我们还提出了困难样本DPO（Difficult Sample DPO）以进一步加强它们的推理能力。通过这种方式，GuardReasoner实现了更好的性能、可解释性和泛化能力。我们在13项基准测试中的3项保护措施任务中进行了广泛的实验和分析，展示了其优越性。尤为值得一提的是，GuardReasoner 8B在平均F1分数上分别超过了GPT-4o+CoT和LLaMA Guard 3 8B达5.74%和20.84%。我们已发布GuardReasoner的训练数据、代码以及不同规模（1B、3B、8B）的模型：[此链接]。

请注意，[此链接]需要替换为实际的链接地址。 

---
# Scaling Inference-Efficient Language Models 

**Title (ZH)**: 高效推理的语言模型的扩展研究 

**Authors**: Song Bian, Minghao Yan, Shivaram Venkataraman  

**Link**: [PDF](https://arxiv.org/pdf/2501.18107)  

**Abstract**: Scaling laws are powerful tools to predict the performance of large language models. However, current scaling laws fall short of accounting for inference costs. In this work, we first show that model architecture affects inference latency, where models of the same size can have up to 3.5x difference in latency. To tackle this challenge, we modify the Chinchilla scaling laws to co-optimize the model parameter count, the number of training tokens, and the model architecture. Due to the reason that models of similar training loss exhibit gaps in downstream evaluation, we also propose a novel method to train inference-efficient models based on the revised scaling laws. We perform extensive empirical studies to fit and evaluate our inference-aware scaling laws. We vary model parameters from 80M to 1B, training tokens from 1.6B to 30B, and model shapes, training a total of 63 models. Guided by our inference-efficient scaling law and model selection method, we release the Morph-1B model, which improves inference latency by 1.8x while maintaining accuracy on downstream tasks compared to open-source models, pushing the Pareto frontier of accuracy-latency tradeoff. 

**Abstract (ZH)**: 规模律是预测大型语言模型性能的强大工具。然而，现有的规模律未能充分考虑推理成本。在这项工作中，我们首先展示了模型架构会影响推理延迟，即使模型大小相同，其推理延迟也可能相差3.5倍。为应对这一挑战，我们修改了Chinchilla规模律，以优化模型参数数量、训练令牌数量和模型架构。由于相似训练损失的模型在下游评估中显示出性能差异，我们还提出了一种基于修订后的规模律训练高效推理模型的新方法。我们进行了广泛的实证研究，以拟合和评估我们的推理感知规模律。我们从80M到1B更改模型参数，从1.6B到30B更改训练令牌数量，并调整了模型形状，总共训练了63个模型。根据我们的高效推理规模律和模型选择方法的指导，我们发布了Morph-1B模型，该模型相比开源模型在保持下游任务准确性的同时，将推理延迟减少了1.8倍，推动了精度-延迟折衷的帕累托前沿。 

---
