# UniGuardian: A Unified Defense for Detecting Prompt Injection, Backdoor Attacks and Adversarial Attacks in Large Language Models 

**Title (ZH)**: UniGuardian：统一防御方法以检测大型语言模型中的提示注入攻击、后门攻击和 adversarial 攻击 

**Authors**: Huawei Lin, Yingjie Lao, Tong Geng, Tan Yu, Weijie Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.13141)  

**Abstract**: Large Language Models (LLMs) are vulnerable to attacks like prompt injection, backdoor attacks, and adversarial attacks, which manipulate prompts or models to generate harmful outputs. In this paper, departing from traditional deep learning attack paradigms, we explore their intrinsic relationship and collectively term them Prompt Trigger Attacks (PTA). This raises a key question: Can we determine if a prompt is benign or poisoned? To address this, we propose UniGuardian, the first unified defense mechanism designed to detect prompt injection, backdoor attacks, and adversarial attacks in LLMs. Additionally, we introduce a single-forward strategy to optimize the detection pipeline, enabling simultaneous attack detection and text generation within a single forward pass. Our experiments confirm that UniGuardian accurately and efficiently identifies malicious prompts in LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）对诸如提示注入、后门攻击和对抗性攻击等攻击具有脆弱性，这些攻击通过操控提示或模型来生成有害输出。在本文中，我们从传统的深度学习攻击范式出发，探讨了这些攻击的内在关系，并将它们统称为提示触发攻击（PTA）。这引发了一个关键问题：我们能否确定一个提示是良性还是受污染的？为了解决这个问题，我们提出了UniGuardian，这是第一个统一的防御机制，旨在检测LLMs中的提示注入、后门攻击和对抗性攻击。此外，我们还提出了一种单向前策略来优化检测管道，从而使攻击检测和文本生成能够在单一前向传递中同时进行。我们的实验结果证实，UniGuardian能够准确且高效地识别LLMs中的恶意提示。 

---
# Facilitating Long Context Understanding via Supervised Chain-of-Thought Reasoning 

**Title (ZH)**: 通过监督链式思维推理促进长上下文理解 

**Authors**: Jingyang Lin, Andy Wong, Tian Xia, Shenghua He, Hui Wei, Mei Han, Jiebo Luo  

**Link**: [PDF](https://arxiv.org/pdf/2502.13127)  

**Abstract**: Recent advances in Large Language Models (LLMs) have enabled them to process increasingly longer sequences, ranging from 2K to 2M tokens and even beyond. However, simply extending the input sequence length does not necessarily lead to effective long-context understanding. In this study, we integrate Chain-of-Thought (CoT) reasoning into LLMs in a supervised manner to facilitate effective long-context understanding. To achieve this, we introduce LongFinanceQA, a synthetic dataset in the financial domain designed to improve long-context reasoning. Unlike existing long-context synthetic data, LongFinanceQA includes intermediate CoT reasoning before the final conclusion, which encourages LLMs to perform explicit reasoning, improving accuracy and interpretability in long-context understanding. To generate synthetic CoT reasoning, we propose Property-driven Agentic Inference (PAI), an agentic framework that simulates human-like reasoning steps, including property extraction, retrieval, and summarization. We evaluate PAI's reasoning capabilities by assessing GPT-4o-mini w/ PAI on the Loong benchmark, outperforming standard GPT-4o-mini by 20.0%. Furthermore, we fine-tune LLaMA-3.1-8B-Instruct on LongFinanceQA, achieving a 24.6% gain on Loong's financial subset. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）的进步使其能够处理越来越长的序列，范围从2K到2M tokens甚至更多。然而，仅仅延长输入序列的长度并不一定能够提高长上下文理解的有效性。在这项研究中，我们以监督的方式将思维链（Chain-of-Thought，CoT）推理集成到LLMs中，以促进有效的长上下文理解。为了实现这一目标，我们引入了LongFinanceQA，这是一个在金融领域设计的合成数据集，旨在提高长上下文推理能力。与现有的长上下文合成数据不同，LongFinanceQA 包括在最终结论之前的中间CoT推理，这鼓励LLMs进行显式的推理，从而提高长上下文理解的准确性和可解释性。为了生成合成的CoT推理，我们提出了基于属性驱动的代理推理（Property-driven Agentic Inference，PAI）框架，这是一个模拟人类推理步骤的框架，包括属性提取、检索和总结。我们通过在Loong基准上评估具有PAI的GPT-4o-mini的推理能力，发现其性能优于标准的GPT-4o-mini，提高幅度为20.0%。此外，我们对LLaMA-3.1-8B-Instruct进行微调，使其在LongFinanceQA上的表现提升了24.6%，并对Loong的金融子集产生了积极影响。 

---
# RuozhiBench: Evaluating LLMs with Logical Fallacies and Misleading Premises 

**Title (ZH)**: 鲁兹比查恩：使用逻辑谬误和误导性前提评估语言模型 

**Authors**: Zenan Zhai, Hao Li, Xudong Han, Zhenxuan Zhang, Yixuan Zhang, Timothy Baldwin, Haonan Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.13125)  

**Abstract**: Recent advances in large language models (LLMs) have shown that they can answer questions requiring complex reasoning. However, their ability to identify and respond to text containing logical fallacies or deliberately misleading premises remains less studied. To address this gap, we introduce RuozhiBench, a bilingual dataset comprising 677 carefully curated questions that contain various forms of deceptive reasoning, meticulously crafted through extensive human effort and expert review. In a comprehensive evaluation of 17 LLMs from 5 Series over RuozhiBench using both open-ended and two-choice formats, we conduct extensive analyses on evaluation protocols and result patterns. Despite their high scores on conventional benchmarks, these models showed limited ability to detect and reason correctly about logical fallacies, with even the best-performing model, Claude-3-haiku, achieving only 62% accuracy compared to the human of more than 90%. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）的最新进展表明，它们能够回答需要复杂推理的问题。然而，它们识别和响应包含逻辑谬误或故意误导前提的文本的能力仍较少受到研究关注。为填补这一空白，我们引入了RuozhiBench，这是一个双语数据集，包含677个精心筛选的问题，这些问题是通过大量的人工努力和专家评审精心设计的，涵盖了各种形式的误导性推理。在使用开放问题和两选一格式对来自5个系列的17个LLM在RuozhiBench上的全面评估中，我们对评估协议和结果模式进行了广泛分析。尽管这些模型在传统基准测试中的得分很高，但在检测和正确推理逻辑谬误方面的能力仍然有限，即使是表现最好的模型Claude-3-haiku，其准确率也只有62%，而人类的准确率超过90%。 

---
# NaturalReasoning: Reasoning in the Wild with 2.8M Challenging Questions 

**Title (ZH)**: 自然推理：面对2.8M个具有挑战性的问答场景中的推理 

**Authors**: Weizhe Yuan, Jane Yu, Song Jiang, Karthik Padthe, Yang Li, Dong Wang, Ilia Kulikov, Kyunghyun Cho, Yuandong Tian, Jason E Weston, Xian Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.13124)  

**Abstract**: Scaling reasoning capabilities beyond traditional domains such as math and coding is hindered by the lack of diverse and high-quality questions. To overcome this limitation, we introduce a scalable approach for generating diverse and challenging reasoning questions, accompanied by reference answers. We present NaturalReasoning, a comprehensive dataset comprising 2.8 million questions that span multiple domains, including STEM fields (e.g., Physics, Computer Science), Economics, Social Sciences, and more. We demonstrate the utility of the questions in NaturalReasoning through knowledge distillation experiments which show that NaturalReasoning can effectively elicit and transfer reasoning capabilities from a strong teacher model. Furthermore, we demonstrate that NaturalReasoning is also effective for unsupervised self-training using external reward models or self-rewarding. 

**Abstract (ZH)**: 将传统领域如数学和编程之外的推理能力进行扩展受到多样化和高质量问题缺乏的限制。为克服这一局限，我们引入了一种可扩展的方法来生成多样化和富有挑战性的推理问题，并附带参考答案。我们提出了NaturalReasoning这一综合数据集，该数据集包含280万道问题，涵盖了多个领域，包括STEM领域（例如，物理、计算机科学）、经济学和社会科学等。我们通过知识萃取实验展示了NaturalReasoning中问题的应用价值，证明NaturalReasoning能够有效地激发和转移一个强大教师模型的推理能力。此外，我们还展示了NaturalReasoning在未监督自我训练中也具有有效性，无论是使用外部奖励模型还是自我奖励机制。 

---
# Adapting Psycholinguistic Research for LLMs: Gender-inclusive Language in a Coreference Context 

**Title (ZH)**: 将心理语言学研究适应于大规模语言模型：核心语义中性别包容性语言的研究 

**Authors**: Marion Bartl, Thomas Brendan Murphy, Susan Leavy  

**Link**: [PDF](https://arxiv.org/pdf/2502.13120)  

**Abstract**: Gender-inclusive language is often used with the aim of ensuring that all individuals, regardless of gender, can be associated with certain concepts. While psycholinguistic studies have examined its effects in relation to human cognition, it remains unclear how Large Language Models (LLMs) process gender-inclusive language. Given that commercial LLMs are gaining an increasingly strong foothold in everyday applications, it is crucial to examine whether LLMs in fact interpret gender-inclusive language neutrally, because the language they generate has the potential to influence the language of their users. This study examines whether LLM-generated coreferent terms align with a given gender expression or reflect model biases. Adapting psycholinguistic methods from French to English and German, we find that in English, LLMs generally maintain the antecedent's gender but exhibit underlying masculine bias. In German, this bias is much stronger, overriding all tested gender-neutralization strategies. 

**Abstract (ZH)**: 为了确保所有个体，无论性别，都能与特定概念相关联，常常使用性别包容性语言。尽管心理学语言学研究已经在人类认知方面考察了其影响，但对于大型语言模型（LLMs）如何处理性别包容性语言这一问题仍然不清楚。鉴于商用LLMs在日常应用中的角色日益重要，有必要考察这类模型是否实际上以中性的方式解释性别包容性语言，因为它们生成的语言可能会对用户语言产生影响。本研究旨在探讨LLM生成的核心参照词是否与给定的性别表达一致，或者是否反映了模型的偏见。采用从法语和德语心理学语言学方法改编的方法，我们发现，在英语中，LLM通常保持先行词的性别，但表现出潜在的男性偏见。在德语中，这种偏见更为强烈，甚至可以克服所有测试的性别中性化策略。 

---
# STEER-ME: Assessing the Microeconomic Reasoning of Large Language Models 

**Title (ZH)**: STEER-ME: 评估大型语言模型的微观经济推理能力 

**Authors**: Narun Raman, Taylor Lundy, Thiago Amin, Jesse Perla, Kevin-Leyton Brown  

**Link**: [PDF](https://arxiv.org/pdf/2502.13119)  

**Abstract**: How should one judge whether a given large language model (LLM) can reliably perform economic reasoning? Most existing LLM benchmarks focus on specific applications and fail to present the model with a rich variety of economic tasks. A notable exception is Raman et al. [2024], who offer an approach for comprehensively benchmarking strategic decision-making; however, this approach fails to address the non-strategic settings prevalent in microeconomics, such as supply-and-demand analysis. We address this gap by taxonomizing microeconomic reasoning into $58$ distinct elements, focusing on the logic of supply and demand, each grounded in up to $10$ distinct domains, $5$ perspectives, and $3$ types. The generation of benchmark data across this combinatorial space is powered by a novel LLM-assisted data generation protocol that we dub auto-STEER, which generates a set of questions by adapting handwritten templates to target new domains and perspectives. Because it offers an automated way of generating fresh questions, auto-STEER mitigates the risk that LLMs will be trained to over-fit evaluation benchmarks; we thus hope that it will serve as a useful tool both for evaluating and fine-tuning models for years to come. We demonstrate the usefulness of our benchmark via a case study on $27$ LLMs, ranging from small open-source models to the current state of the art. We examined each model's ability to solve microeconomic problems across our whole taxonomy and present the results across a range of prompting strategies and scoring metrics. 

**Abstract (ZH)**: 如何判断一个给定的大语言模型（LLM）能否可靠地进行经济推理？目前大多数现有的LLM基准主要集中在特定的应用上，并未提供一个多样化的经济任务集。一个显著的例外是Raman等人（2024），他们提供了一种全面评估战略决策的方法；然而，该方法未能涵盖微观经济学中常见的非战略环境，如供需分析。为填补这一空白，我们将微观经济推理分类为58个不同的元素，重点关注供需逻辑，这些元素涵盖了多达10个不同的领域、5个视角和3种类型。我们在这一组合空间中生成基准数据的方法是一种新颖的基于LLM的数据生成协议，我们称之为auto-STEER，该协议通过适应手写的模板来生成针对新领域和视角的问题集。由于它可以自动化地生成新的问题，auto-STEER减少了LLM过度拟合评估基准的风险；我们希望它能在未来数年中作为评估和微调模型的有用工具。我们通过一项涵盖27个不同模型（从小的开源模型到当前最先进的模型）的应用案例研究，展示了我们基准的实用性。我们评估了每个模型解决我们整个分类体系下的微观经济问题的能力，并通过各种提示策略和评分指标呈现了结果。 

---
# The influence of motion features in temporal perception 

**Title (ZH)**: 运动特征对时间知觉的影响 

**Authors**: Rosa Illan Castillo, Javier Valenzuela  

**Link**: [PDF](https://arxiv.org/pdf/2502.13114)  

**Abstract**: This paper examines the role of manner-of-motion verbs in shaping subjective temporal perception and emotional resonance. Through four complementary studies, we explore how these verbs influence the conceptualization of time, examining their use in literal and metaphorical (temporal) contexts. Our findings reveal that faster verbs (e.g., fly, zoom) evoke dynamic and engaging temporal experiences, often linked to positive emotions and greater agency. In contrast, slower verbs (e.g., crawl, drag) convey passivity, monotony, and negative emotions, reflecting tedious or constrained experiences of time. These effects are amplified in metaphorical contexts, where manner verbs encode emotional and experiential nuances that transcend their literal meanings. We also find that participants prefer manner verbs over path verbs (e.g., go, pass) in emotionally charged temporal contexts, as manner verbs capture the experiential and emotional qualities of time more effectively. These findings highlight the interplay between language, motion, and emotion in shaping temporal perception, offering insights into how linguistic framing influences subjective experiences of time. 

**Abstract (ZH)**: 本文探讨了运动动词在塑造主观时间感知和情感共振中的作用。通过四项互补的研究，我们探讨了这些动词如何影响时间概念的理解，考察了它们在字面意义上和比喻时间语境中的使用情况。研究发现，使用快速动词（如fly、zoom）可以唤起动态且引人入胜的时间体验，往往与积极情绪和更强的自主感相关。相反，使用缓慢动词（如crawl、drag）则传达出被动、单调和负面情绪，反映了时间和经历的单调或受限的体验。这些效果在比喻语境中尤为显著，因为运动动词传达的情感和体验细微差别超出了其字面意义。我们还发现，在情感紧张的时间语境中，参与者更偏好使用运动动词而非路径动词（如go、pass），因为运动动词更有效地捕捉了时间的体验和情感特质。这些发现突显了语言、运动和情感在塑造时间感知中的相互作用，提供了关于语言框架如何影响人们对时间的主观体验的见解。 

---
# Improving Clinical Question Answering with Multi-Task Learning: A Joint Approach for Answer Extraction and Medical Categorization 

**Title (ZH)**: 利用多任务学习提高临床问题回答：一种结合答案提取和医疗分类的联合方法 

**Authors**: Priyaranjan Pattnayak, Hitesh Laxmichand Patel, Amit Agarwal, Bhargava Kumar, Srikant Panda, Tejaswini Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2502.13108)  

**Abstract**: Clinical Question Answering (CQA) plays a crucial role in medical decision-making, enabling physicians to extract relevant information from Electronic Medical Records (EMRs). While transformer-based models such as BERT, BioBERT, and ClinicalBERT have demonstrated state-of-the-art performance in CQA, existing models lack the ability to categorize extracted answers, which is critical for structured retrieval, content filtering, and medical decision support.
To address this limitation, we introduce a Multi-Task Learning (MTL) framework that jointly trains CQA models for both answer extraction and medical categorization. In addition to predicting answer spans, our model classifies responses into five standardized medical categories: Diagnosis, Medication, Symptoms, Procedure, and Lab Reports. This categorization enables more structured and interpretable outputs, making clinical QA models more useful in real-world healthcare settings.
We evaluate our approach on emrQA, a large-scale dataset for medical question answering. Results show that MTL improves F1-score by 2.2% compared to standard fine-tuning, while achieving 90.7% accuracy in answer categorization. These findings suggest that MTL not only enhances CQA performance but also introduces an effective mechanism for categorization and structured medical information retrieval. 

**Abstract (ZH)**: 临床问题回答（CQA）在医疗决策中发挥着重要作用，使医生能够从电子病历（EMRs）中提取相关信息。尽管基于变换器的模型如BERT、BioBERT和ClinicalBERT已经在CQA任务中展示了最先进的性能，但现有的模型缺乏对提取答案进行分类的能力，这在结构化检索、内容过滤和医疗决策支持方面是至关重要的。

为了解决这一局限性，我们引入了一个多任务学习（MTL）框架，该框架同时训练CQA模型进行答案提取和医学分类。除了预测答案跨度外，我们的模型还将响应分类为五个标准化的医学类别：诊断、药物、症状、程序和实验室报告。这种分类使临床QA模型的输出更加结构化和可解释，从而在实际医疗保健环境中更具实用价值。

我们在emrQA数据集上评估了我们的方法，这是一个用于医疗问答的大规模数据集。结果显示，MTL在F1分数上提高了2.2%，并且在答案分类准确性上达到了90.7%。这些发现表明，MTL不仅增强了CQA性能，还引入了一种有效的分类机制和结构化医疗信息检索方法。 

---
# Text2World: Benchmarking Large Language Models for Symbolic World Model Generation 

**Title (ZH)**: Text2World：大型语言模型符号世界模型生成基准测试 

**Authors**: Mengkang Hu, Tianxing Chen, Yude Zou, Yuheng Lei, Qiguang Chen, Ming Li, Hongyuan Zhang, Wenqi Shao, Ping Luo  

**Link**: [PDF](https://arxiv.org/pdf/2502.13092)  

**Abstract**: Recently, there has been growing interest in leveraging large language models (LLMs) to generate symbolic world models from textual descriptions. Although LLMs have been extensively explored in the context of world modeling, prior studies encountered several challenges, including evaluation randomness, dependence on indirect metrics, and a limited domain scope. To address these limitations, we introduce a novel benchmark, Text2World, based on planning domain definition language (PDDL), featuring hundreds of diverse domains and employing multi-criteria, execution-based metrics for a more robust evaluation. We benchmark current LLMs using Text2World and find that reasoning models trained with large-scale reinforcement learning outperform others. However, even the best-performing model still demonstrates limited capabilities in world modeling. Building on these insights, we examine several promising strategies to enhance the world modeling capabilities of LLMs, including test-time scaling, agent training, and more. We hope that Text2World can serve as a crucial resource, laying the groundwork for future research in leveraging LLMs as world models. The project page is available at this https URL. 

**Abstract (ZH)**: 近年来，人们越来越关注利用大规模语言模型（LLMs）从文本描述中生成符号型世界模型。虽然LLMs已经在世界建模的背景下得到了广泛探索，但先前的研究遇到了一些挑战，包括评估的随机性、依赖于间接指标以及研究领域范围有限。为应对这些局限性，我们提出了一种基于规划定义语言（PDDL）的新基准——Text2World，该基准包含数百个多样化领域，并采用多准则、基于执行的评估指标，从而提供更稳健的评估。我们使用Text2World对当前的LLMs进行了基准测试，发现使用大规模强化学习训练的推理模型的表现优于其他模型。然而，即使是表现最佳的模型在世界建模方面仍显示出有限的能力。基于这些发现，我们探讨了几种有潜力提高LLMs世界建模能力的策略，包括测试时扩展、智能体训练等。我们希望Text2World能够作为重要资源，为今后利用LLMs作为世界模型的研究奠定基础。项目页面可以通过以下链接访问：this https URL。 

---
# KAPPA: A Generic Patent Analysis Framework with Keyphrase-Based Portraits 

**Title (ZH)**: KAPPA：一种基于关键词画像的通用专利分析框架 

**Authors**: Xin Xia, Yujin Wang, Jun Zhou, Guisheng Zhong, Linning Cai, Chen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13076)  

**Abstract**: Patent analysis highly relies on concise and interpretable document representations, referred to as patent portraits. Keyphrases, both present and absent, are ideal candidates for patent portraits due to their brevity, representativeness, and clarity. In this paper, we introduce KAPPA, an integrated framework designed to construct keyphrase-based patent portraits and enhance patent analysis. KAPPA operates in two phases: patent portrait construction and portrait-based analysis. To ensure effective portrait construction, we propose a semantic-calibrated keyphrase generation paradigm that integrates pre-trained language models with a prompt-based hierarchical decoding strategy to leverage the multi-level structural characteristics of patents. For portrait-based analysis, we develop a comprehensive framework that employs keyphrase-based patent portraits to enable efficient and accurate patent analysis. Extensive experiments on benchmark datasets of keyphrase generation, the proposed model achieves significant improvements compared to state-of-the-art baselines. Further experiments conducted on real-world patent applications demonstrate that our keyphrase-based portraits effectively capture domain-specific knowledge and enrich semantic representation for patent analysis tasks. 

**Abstract (ZH)**: 专利分析高度依赖简洁且易于解释的文档表示，称为专利肖像。现有关键词（包括存在和不存在的关键词）因其简明性、代表性及清晰性，是理想的专利肖像候选者。本文介绍了一种名为KAPPA的集成框架，该框架旨在构建基于关键词的专利肖像并增强专利分析能力。KAPPA采用两个阶段来运行：专利肖像构建阶段和基于肖像的分析阶段。为了确保有效的肖像构建，我们提出了一种语义校准的关键词生成范式，该范式结合预训练语言模型和基于提示的层次解码策略，利用专利的多级结构特性。在基于肖像的分析阶段，我们开发了一个综合框架，利用基于关键词的专利肖像以实现高效且准确的专利分析。通过对关键短语生成基准数据集的大量实验，所提出模型在各项指标上显著优于现有的基线模型。在对实际专利申请的进一步实验中，展示了基于关键词的专利肖像能够有效捕捉特定领域的知识，并丰富专利分析任务的语义表示。 

---
# Cramming 1568 Tokens into a Single Vector and Back Again: Exploring the Limits of Embedding Space Capacity 

**Title (ZH)**: 将1568个词元压缩进单一向量再重新解读：探索嵌入空间容量的极限 

**Authors**: Yuri Kuratov, Mikhail Arkhipov, Aydar Bulatov, Mikhail Burtsev  

**Link**: [PDF](https://arxiv.org/pdf/2502.13063)  

**Abstract**: A range of recent works addresses the problem of compression of sequence of tokens into a shorter sequence of real-valued vectors to be used as inputs instead of token embeddings or key-value cache. These approaches allow to reduce the amount of compute in existing language models. Despite relying on powerful models as encoders, the maximum attainable lossless compression ratio is typically not higher than x10. This fact is highly intriguing because, in theory, the maximum information capacity of large real-valued vectors is far beyond the presented rates even for 16-bit precision and a modest vector size. In this work, we explore the limits of compression by replacing the encoder with a per-sample optimization procedure. We show that vectors with compression ratios up to x1500 exist, which highlights two orders of magnitude gap between existing and practically attainable solutions. Furthermore, we empirically show that the compression limits are determined not by the length of the input but by the amount of uncertainty to be reduced, namely, the cross-entropy loss on this sequence without any conditioning. The obtained limits highlight the substantial gap between the theoretical capacity of input embeddings and their practical utilization, suggesting significant room for optimization in model design. 

**Abstract (ZH)**: 近年来，有一系列研究致力于将序列的标记压缩为较短的实值向量序列，以替代标记嵌入或键值缓存作为输入。这些方法允许在现有的语言模型中减少计算量。尽管这些方法依赖于强大的模型作为编码器，但能够实现无损压缩的最大比率通常不超过x10。这一事实非常引人注目，因为在理论上，即使在16位精度和适度大小的情况下，大型实值向量的最大信息容量远超出当前的压缩率。在本文中，我们探讨了通过用每样本优化过程替代编码器来压缩的极限。我们展示了压缩比高达x1500的向量存在，这突显出现有解决方案与可实现的解决方案之间的两个数量级差距。此外，我们实验证明压缩极限并不由输入的长度决定，而是由需要减少的不确定性决定，即在没有任何条件的情况下，该序列的交叉熵损失。所获得的极限突显了输入嵌入的理论容量与其实际利用之间的巨大差距，表明在模型设计中存在优化的巨大空间。 

---
# Improved Fine-Tuning of Large Multimodal Models for Hateful Meme Detection 

**Title (ZH)**: 改进的大规模多模态模型细调方法在仇恨亚文化识别中的应用 

**Authors**: Jingbiao Mei, Jinghong Chen, Guangyu Yang, Weizhe Lin, Bill Byrne  

**Link**: [PDF](https://arxiv.org/pdf/2502.13061)  

**Abstract**: Hateful memes have become a significant concern on the Internet, necessitating robust automated detection systems. While large multimodal models have shown strong generalization across various tasks, they exhibit poor generalization to hateful meme detection due to the dynamic nature of memes tied to emerging social trends and breaking news. Recent work further highlights the limitations of conventional supervised fine-tuning for large multimodal models in this context. To address these challenges, we propose Large Multimodal Model Retrieval-Guided Contrastive Learning (LMM-RGCL), a novel two-stage fine-tuning framework designed to improve both in-domain accuracy and cross-domain generalization. Experimental results on six widely used meme classification datasets demonstrate that LMM-RGCL achieves state-of-the-art performance, outperforming agent-based systems such as VPD-PALI-X-55B. Furthermore, our method effectively generalizes to out-of-domain memes under low-resource settings, surpassing models like GPT-4o. 

**Abstract (ZH)**: 仇恨帖梗在网络上传播已成为一个重要关切，需要建立稳健的自动化检测系统。虽然大型多模态模型在各种任务上表现出较强的泛化能力，但在仇恨帖梗检测方面却表现出较差的泛化能力，这主要是因为帖子梗与不断出现的社会趋势和突发新闻密切相关。近期的研究进一步指出了在这种背景下，传统监督微调方法对于大型多模态模型的局限性。为应对这些挑战，我们提出了一种新的两阶段微调框架——大型多模态模型检索引导对比学习（LMM-RGCL），旨在提高领域内准确性和跨领域泛化能力。我们在六个广泛使用的帖梗分类数据集上的实验结果表明，LMM-RGCL 赢得了最先进的性能，并在性能方面超越了基于代理的方法，如 VPD-PALI-X-55B。此外，我们的方法在资源有限的情况下能够有效泛化到领域外的帖梗上，超越了如 GPT-4o 等模型。 

---
# SimpleVQA: Multimodal Factuality Evaluation for Multimodal Large Language Models 

**Title (ZH)**: SimpleVQA：多模态事实性评估方法用于多模态大型语言模型 

**Authors**: Xianfu Cheng, Wei Zhang, Shiwei Zhang, Jian Yang, Xiangyuan Guan, Xianjie Wu, Xiang Li, Ge Zhang, Jiaheng Liu, Yuying Mai, Yutao Zeng, Zhoufutu Wen, Ke Jin, Baorui Wang, Weixiao Zhou, Yunhong Lu, Tongliang Li, Wenhao Huang, Zhoujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.13059)  

**Abstract**: The increasing application of multi-modal large language models (MLLMs) across various sectors have spotlighted the essence of their output reliability and accuracy, particularly their ability to produce content grounded in factual information (e.g. common and domain-specific knowledge). In this work, we introduce SimpleVQA, the first comprehensive multi-modal benchmark to evaluate the factuality ability of MLLMs to answer natural language short questions. SimpleVQA is characterized by six key features: it covers multiple tasks and multiple scenarios, ensures high quality and challenging queries, maintains static and timeless reference answers, and is straightforward to evaluate. Our approach involves categorizing visual question-answering items into 9 different tasks around objective events or common knowledge and situating these within 9 topics. Rigorous quality control processes are implemented to guarantee high-quality, concise, and clear answers, facilitating evaluation with minimal variance via an LLM-as-a-judge scoring system. Using SimpleVQA, we perform a comprehensive assessment of leading 18 MLLMs and 8 text-only LLMs, delving into their image comprehension and text generation abilities by identifying and analyzing error cases. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在各个领域的广泛应用凸显了其输出可靠性和准确性的本质，尤其是它们生成基于事实信息的内容（包括通用和领域特定知识）的能力。本文介绍了SimpleVQA，这是首个全面评估MLLMs事实准确性能力的多模态基准测试，通过自然语言简短问题来评估其回答能力。SimpleVQA具有六大关键特征：覆盖多个任务和场景、确保高质量和具有挑战性的查询、保持静态和永恒参考答案以及易于评估。我们的方法是将视觉问答项目分类为9个与客观事件或常见知识相关的任务，并将其置于9个主题之下。实施了严格的质量控制流程，以保证高质量、简洁和清晰的答案，通过LLM作为法官的评分系统实现最小偏差的评估。使用SimpleVQA，我们对18个领先的MLLMs和8个仅基于文本的LLM进行了全面评估，深入探讨了它们的图像理解和文本生成能力，并对错误案例进行了识别和分析。 

---
# AEIA-MN: Evaluating the Robustness of Multimodal LLM-Powered Mobile Agents Against Active Environmental Injection Attacks 

**Title (ZH)**: AEIA-MN：评估多模态LLM驱动的移动代理在活跃环境注入攻击下的鲁棒性 

**Authors**: Yurun Chen, Xueyu Hu, Keting Yin, Juncheng Li, Shengyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13053)  

**Abstract**: As researchers continuously optimize AI agents to perform tasks more effectively within operating systems, they often neglect to address the critical need for enabling these agents to identify "impostors" within the system. Through an analysis of the agents' operating environment, we identified a potential threat: attackers can disguise their attack methods as environmental elements, injecting active disturbances into the agents' execution process, thereby disrupting their decision-making. We define this type of attack as Active Environment Injection Attack (AEIA). Based on this, we propose AEIA-MN, an active environment injection attack scheme that exploits interaction vulnerabilities in the mobile operating system to evaluate the robustness of MLLM-based agents against such threats. Experimental results show that even advanced MLLMs are highly vulnerable to this attack, achieving a maximum attack success rate of 93% in the AndroidWorld benchmark. 

**Abstract (ZH)**: 随着研究者不断优化AI代理以在操作系统中更有效地执行任务，他们常常忽视了使这些代理能够识别系统中的“冒充者”的关键需求。通过对代理的操作环境进行分析，我们发现了一个潜在威胁：攻击者可以伪装其攻击手段为环境元素，并向代理的执行过程注入活跃的干扰，从而扰乱其决策过程。我们定义此类攻击为活跃环境注入攻击（AEIA）。基于此，我们提出了一种AEIA-MN方案，该方案利用移动操作系统中的交互漏洞来评估基于MLLM的代理在面对此类威胁时的鲁棒性。实验结果显示，即使是先进的MLLM，在AndroidWorld基准测试中也高度容易受到此类攻击的影响，攻击成功率达到93%。 

---
# Do we still need Human Annotators? Prompting Large Language Models for Aspect Sentiment Quad Prediction 

**Title (ZH)**: 我们还需要人类注释员吗？通过提示大语言模型进行 aspect 情感四元组预测 

**Authors**: Nils Constantin Hellwig, Jakob Fehle, Udo Kruschwitz, Christian Wolff  

**Link**: [PDF](https://arxiv.org/pdf/2502.13044)  

**Abstract**: Aspect sentiment quadruple prediction (ASQP) facilitates a detailed understanding of opinions expressed in a text by identifying the opinion term, aspect term, aspect category and sentiment polarity for each opinion. However, annotating a full set of training examples to fine-tune models for ASQP is a resource-intensive process. In this study, we explore the capabilities of large language models (LLMs) for zero- and few-shot learning on the ASQP task across five diverse datasets. We report F1 scores slightly below those obtained with state-of-the-art fine-tuned models but exceeding previously reported zero- and few-shot performance. In the 40-shot setting on the Rest16 restaurant domain dataset, LLMs achieved an F1 score of 52.46, compared to 60.39 by the best-performing fine-tuned method MVP. Additionally, we report the performance of LLMs in target aspect sentiment detection (TASD), where the F1 scores were also close to fine-tuned models, achieving 66.03 on Rest16 in the 40-shot setting, compared to 72.76 with MVP. While human annotators remain essential for achieving optimal performance, LLMs can reduce the need for extensive manual annotation in ASQP tasks. 

**Abstract (ZH)**: Aspect情感四元组预测（ASQP）通过识别每个观点中的意见词、方面词、方面类别和情感极性，有助于对文本中表达的观点进行详细的理解。然而，为了 fine-tune ASQP 任务的模型，标注完整的一组训练示例是一个资源密集型的过程。在本研究中，我们探讨了大规模语言模型（LLMs）在跨五个不同数据集上的 ASQP 任务中进行零-shot 和少-shot 学习的能力。我们报告的 F1 分数略低于最先进的 fine-tuned 模型，但超过了先前报道的零-shot 和少-shot 性能。在针对 Restaurant16 餐馆领域数据集的 40-shot 设置中，LLMs 达到了 52.46 的 F1 分数，而最佳 fine-tuned 方法 MVP 达到了 60.39。此外，我们在目标方面情感检测（TASD）任务中报告了 LLMs 的性能，F1 分数也接近 fine-tuned 模型，在 40-shot 设置下，LLMs 达到了 66.03，而 MVP 达到了 72.76。虽然人类注释者在获得最佳性能方面仍然是必不可少的，但 LLMs 可以减少 ASQP 任务中大量手动注释的需求。 

---
# Natural Language Generation from Visual Sequences: Challenges and Future Directions 

**Title (ZH)**: 视觉序列的自然语言生成：挑战与未来方向 

**Authors**: Aditya K Surikuchi, Raquel Fernández, Sandro Pezzelle  

**Link**: [PDF](https://arxiv.org/pdf/2502.13034)  

**Abstract**: The ability to use natural language to talk about visual content is at the core of human intelligence and a crucial feature of any artificial intelligence system. Various studies have focused on generating text for single images. In contrast, comparatively little attention has been paid to exhaustively analyzing and advancing work on multiple-image vision-to-text settings. In this position paper, we claim that any task dealing with temporally ordered sequences of multiple images or frames is an instance of a broader, more general problem involving the understanding of intricate relationships between the visual content and the corresponding text. We comprehensively analyze five tasks that are instances of this problem and argue that they pose a common set of challenges and share similarities in terms of modeling and evaluation approaches. Based on the insights from these various aspects and stages of multi-image-to-text generation, we highlight several open questions and suggest future research directions. We believe that these directions can advance the understanding of complex phenomena in this domain and the development of better models. 

**Abstract (ZH)**: 能够使用自然语言描述视觉内容是人类智能的核心，并且是任何人工智能系统的关键特征。各类研究主要集中于为单张图片生成文本。相比之下，对多张图像的视觉到文本转换进行详尽分析和推进的工作则较少受到关注。在此观点性论文中，我们主张任何涉及时间上有序的多张图像或帧的任务其实是更广泛、更通用问题的一个特例，该问题涉及到理解视觉内容和相应文本之间复杂关系的理解。我们全面分析了五个此类问题的特例，并论证了它们共同面临的一系列挑战，并且在建模和评估方法上具有相似性。基于这些多图像到文本生成各个方面的洞见，我们突出强调了几个开放问题，并提出了未来的研究方向。我们认为，这些方向可以促进对该领域复杂现象的理解，并推动更优秀模型的发展。 

---
# HPSS: Heuristic Prompting Strategy Search for LLM Evaluators 

**Title (ZH)**: HPSS：LLM评估器的启发式提示策略搜索 

**Authors**: Bosi Wen, Pei Ke, Yufei Sun, Cunxiang Wang, Xiaotao Gu, Jinfeng Zhou, Jie Tang, Hongning Wang, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13031)  

**Abstract**: Since the adoption of large language models (LLMs) for text evaluation has become increasingly prevalent in the field of natural language processing (NLP), a series of existing works attempt to optimize the prompts for LLM evaluators to improve their alignment with human judgment. However, their efforts are limited to optimizing individual factors of evaluation prompts, such as evaluation criteria or output formats, neglecting the combinatorial impact of multiple factors, which leads to insufficient optimization of the evaluation pipeline. Nevertheless, identifying well-behaved prompting strategies for adjusting multiple factors requires extensive enumeration. To this end, we comprehensively integrate 8 key factors for evaluation prompts and propose a novel automatic prompting strategy optimization method called Heuristic Prompting Strategy Search (HPSS). Inspired by the genetic algorithm, HPSS conducts an iterative search to find well-behaved prompting strategies for LLM evaluators. A heuristic function is employed to guide the search process, enhancing the performance of our algorithm. Extensive experiments across four evaluation tasks demonstrate the effectiveness of HPSS, consistently outperforming both human-designed evaluation prompts and existing automatic prompt optimization methods. 

**Abstract (ZH)**: 自从大规模语言模型（LLMs）在自然语言处理（NLP）领域被广泛应用于文本评估以来，一系列现有工作试图优化评估提示，以提高LLM评估者与人类判断的一致性。然而，他们的努力仅限于优化评估提示的个别因素，例如评估标准或输出格式，而忽略了多个因素组合影响，这导致评估管道的优化不足。尽管如此，识别能够调整多个因素的可行提示策略需要进行广泛的枚举。为此，我们全面整合了8个关键的评估提示因素，并提出了一种新颖的自动提示策略优化方法——启发式提示策略搜索（HPSS）。受到遗传算法的启发，HPSS进行迭代搜索，以找到适用于LLM评估者的可行提示策略。采用启发式函数指导搜索过程，提高算法性能。在四个评估任务上的大量实验表明，HPSS的有效性优于人工设计的评估提示和现有的自动提示优化方法。 

---
# Whose story is it? Personalizing story generation by inferring author styles 

**Title (ZH)**: 《谁的故事？通过推断作者风格实现个性化故事生成》 

**Authors**: Nischal Ashok Kumar, Chau Minh Pham, Mohit Iyyer, Andrew Lan  

**Link**: [PDF](https://arxiv.org/pdf/2502.13028)  

**Abstract**: Personalization has become essential for improving user experience in interactive writing and educational applications, yet its potential in story generation remains largely unexplored. In this work, we propose a novel two-stage pipeline for personalized story generation. Our approach first infers an author's implicit story-writing characteristics from their past work and organizes them into an Author Writing Sheet, inspired by narrative theory. The second stage uses this sheet to simulate the author's persona through tailored persona descriptions and personalized story writing rules. To enable and validate our approach, we construct Mythos, a dataset of 590 stories from 64 authors across five distinct sources that reflect diverse story-writing settings. A head-to-head comparison with a non-personalized baseline demonstrates our pipeline's effectiveness in generating high-quality personalized stories. Our personalized stories achieve a 75 percent win rate (versus 14 percent for the baseline and 11 percent ties) in capturing authors' writing style based on their past works. Human evaluation highlights the high quality of our Author Writing Sheet and provides valuable insights into the personalized story generation task. Notable takeaways are that writings from certain sources, such as Reddit, are easier to personalize than others, like AO3, while narrative aspects, like Creativity and Language Use, are easier to personalize than others, like Plot. 

**Abstract (ZH)**: 个性化已成为提升交互式写作和教育应用中用户经验的关键因素，但在故事生成方面的潜力尚未得到充分探索。本文提出了一种新颖的两阶段管道，用于个性化故事生成。我们首先从作者的过往作品中推断出其隐含的故事创作特征，并将其组织成一张作者写作表单，这一做法借鉴了叙事理论。第二阶段则利用该表单通过定制化的角色描述和个性化的故事情节生成规则来模拟作者的性格。为了支持和验证我们的方法，我们构建了一个包含590个来自64位作者，涵盖五大不同来源的故事的数据集，这些作者的故事展现了多样的故事创作环境。通过与非个性化基线模型的直接对比，显示了我们管道在生成高质量个性化故事方面的有效性。基于过往作品，我们的个性化故事获得了75%的成功率（而基线模型仅为14%，平局率为11%）。人类评估强调了作者写作表单的高质量和价值，并对个性化故事生成任务提供了宝贵的见解。值得注意的是，某些来源的写作（如Reddit）比其他来源（如AO3）更容易实现个性化，而在叙事方面，如创造力和语言使用方面，更容易实现个性化，而剧情方面的个性化则较为困难。 

---
# Oreo: A Plug-in Context Reconstructor to Enhance Retrieval-Augmented Generation 

**Title (ZH)**: Oreo：一种插件式上下文重建器，用于增强检索增强生成 

**Authors**: Sha Li, Naren Ramarkrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2502.13019)  

**Abstract**: Despite the remarkable capabilities of Large Language Models (LLMs) in various NLP tasks, they remain vulnerable to hallucinations due to their limited parametric knowledge and lack of domain-specific expertise. Retrieval-Augmented Generation (RAG) addresses this challenge by incorporating external document retrieval to augment the knowledge base of LLMs. In this approach, RAG retrieves document chunks from an external corpus in response to a query, which are then used as context for the downstream language model to generate an answer. However, these retrieved knowledge sources often include irrelevant or erroneous information, undermining the effectiveness of RAG in downstream tasks. To overcome this limitation, we introduce a compact, efficient, and pluggable module designed to refine external knowledge sources before feeding them to the generator. The module reconstructs retrieved content by extracting the most relevant and supportive information and reorganising it into a concise, query-specific format. Through a three-stage training paradigm - comprising supervised fine-tuning, contrastive multi-task learning, and reinforcement learning-based alignment - it prioritises critical knowledge and aligns it with the generator's preferences. This method enables LLMs to produce outputs that are more accurate, reliable, and contextually appropriate. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在各种自然语言处理（NLP）任务中展现出显著的能力，但由于其有限的参数知识和缺乏特定领域的专业知识，它们仍然容易发生幻觉。检索增强生成（RAG）通过引入外部文档检索来增强LLMs的知识库，从而解决了这一挑战。在这种方法中，RAG收到查询后会从外部语料库中检索文档片段，然后将这些片段用作下游语言模型生成答案的上下文。然而，这些检索到的知识来源往往包含不相关或错误的信息，削弱了RAG在下游任务中的有效性。为克服这一局限性，我们提出了一种紧凑、高效且可插拔的模块，该模块在将外部知识源输入生成器之前对其进行精炼。该模块通过提取最相关和支持性的信息并重新组织成简洁、查询特定的格式来重构检索内容。通过一个三阶段的训练范式——包括监督微调、对比多任务学习和基于强化学习的对齐——该方法优先处理关键知识，并与生成器的偏好进行对齐。这种方法使LLMs能够产生更准确、可靠且上下文适当的输出。 

---
# Adaptive Knowledge Graphs Enhance Medical Question Answering: Bridging the Gap Between LLMs and Evolving Medical Knowledge 

**Title (ZH)**: 自适应知识图谱增强医学问答：弥合大规模语言模型与 evolving 医学知识之间的差距 

**Authors**: Mohammad Reza Rezaei, Reza Saadati Fard, Jayson Parker, Rahul G. Krishnan, Milad Lankarany  

**Link**: [PDF](https://arxiv.org/pdf/2502.13010)  

**Abstract**: Large Language Models (LLMs) have significantly advanced medical question-answering by leveraging extensive clinical data and medical literature. However, the rapid evolution of medical knowledge and the labor-intensive process of manually updating domain-specific resources pose challenges to the reliability of these systems. To address this, we introduce Adaptive Medical Graph-RAG (AMG-RAG), a comprehensive framework that automates the construction and continuous updating of medical knowledge graphs, integrates reasoning, and retrieves current external evidence, such as PubMed and WikiSearch. By dynamically linking new findings and complex medical concepts, AMG-RAG not only improves accuracy but also enhances interpretability in medical queries.
Evaluations on the MEDQA and MEDMCQA benchmarks demonstrate the effectiveness of AMG-RAG, achieving an F1 score of 74.1 percent on MEDQA and an accuracy of 66.34 percent on MEDMCQA, outperforming both comparable models and those 10 to 100 times larger. Notably, these improvements are achieved without increasing computational overhead, highlighting the critical role of automated knowledge graph generation and external evidence retrieval in delivering up-to-date, trustworthy medical insights. 

**Abstract (ZH)**: 大规模语言模型（LLMs）通过利用丰富的临床数据和医学文献，显著推进了医学问题解答领域的发展。然而，医学知识的快速演变和手动更新特定领域资源的劳动密集型过程对这些系统的可靠性构成了挑战。为了解决这一问题，我们提出了自适应医学图-RAG（AMG-RAG）框架，该框架能够自动构建和持续更新医学知识图谱，整合推理，并集成当前外部证据的检索，例如PubMed和WikiSearch。通过动态链接新发现和复杂的医学概念，AMG-RAG 不仅提高了准确性，还增强了医学查询的可解释性。

在MEDQA和MEDMCQA基准测试中的评估表明，AMG-RAG 的有效性。其在MEDQA上的F1分数为74.1%，在MEDMCQA上的准确率为66.34%，均优于可比模型，且优于规模大10到100倍的模型。值得注意的是，这些改进是在不增加计算开销的情况下实现的，这突显了自动化知识图谱生成和外部证据检索在提供最新、可靠医学见解方面的关键作用。 

---
# Language Barriers: Evaluating Cross-Lingual Performance of CNN and Transformer Architectures for Speech Quality Estimation 

**Title (ZH)**: 语言障碍：评估CNN和Transformer架构在跨语言声音质量估计中的性能 

**Authors**: Wafaa Wardah, Tuğçe Melike Koçak Büyüktaş, Kirill Shchegelskiy, Sebastian Möller, Robert P. Spang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13004)  

**Abstract**: Objective speech quality models aim to predict human-perceived speech quality using automated methods. However, cross-lingual generalization remains a major challenge, as Mean Opinion Scores (MOS) vary across languages due to linguistic, perceptual, and dataset-specific differences. A model trained primarily on English data may struggle to generalize to languages with different phonetic, tonal, and prosodic characteristics, leading to inconsistencies in objective assessments. This study investigates the cross-lingual performance of two speech quality models: NISQA, a CNN-based model, and a Transformer-based Audio Spectrogram Transformer (AST) model. Both models were trained exclusively on English datasets containing over 49,000 speech samples and subsequently evaluated on speech in German, French, Mandarin, Swedish, and Dutch. We analyze model performance using Pearson Correlation Coefficient (PCC) and Root Mean Square Error (RMSE) across five speech quality dimensions: coloration, discontinuity, loudness, noise, and MOS. Our findings show that while AST achieves a more stable cross-lingual performance, both models exhibit noticeable biases. Notably, Mandarin speech quality predictions correlate highly with human MOS scores, whereas Swedish and Dutch present greater prediction challenges. Discontinuities remain difficult to model across all languages. These results highlight the need for more balanced multilingual datasets and architecture-specific adaptations to improve cross-lingual generalization. 

**Abstract (ZH)**: 客观语音质量模型旨在使用自动化方法预测人类感知的语音质量。然而，跨语言泛化仍然是一个重大挑战，因为平均意见评分（Modified Overall Mean Opinion Score, MOS）因语言、感知和数据集特定差异而异。主要基于英语数据训练的模型可能难以泛化到具有不同音素、声调和语调特征的语言，导致客观评估存在不一致性。本研究调查了两种语音质量模型在跨语言表现上的差异：一种是基于卷积神经网络（CNN）的模型 NISQA，另一种是基于变换器的音频光谱变换（Audio Spectrogram Transformer, AST）模型。这两种模型均仅在包含超过 49,000 个语音样本的英语数据集上进行训练，并随后在德语、法语、普通话、瑞典语和荷兰语的语音上进行评估。我们使用皮尔逊相关系数（PCC）和均方根误差（RMSE）分析了两种模型在五种语音质量维度上的表现：色彩、不连续性、响度、噪声和 MOS。研究发现，尽管 AST 在跨语言性能上更为稳定，但两种模型都表现出明显的偏差。特别地，普通话的语音质量预测与人类的 MOS 评分高度相关，而瑞典语和荷兰语则面临更大的预测挑战。不连续性在所有语言中都难以建模。这些结果表明，需要更平衡的多语言数据集以及特定架构的适配来提高跨语言的泛化能力。 

---
# Eager Updates For Overlapped Communication and Computation in DiLoCo 

**Title (ZH)**: DiLoCo中重叠通信与计算的前瞻更新方法 

**Authors**: Satyen Kale, Arthur Douillard, Yanislav Donchev  

**Link**: [PDF](https://arxiv.org/pdf/2502.12996)  

**Abstract**: Distributed optimization methods such as DiLoCo have been shown to be effective in training very large models across multiple distributed workers, such as datacenters. These methods split updates into two parts: an inner optimization phase, where the workers independently execute multiple optimization steps on their own local data, and an outer optimization step, where the inner updates are synchronized. While such approaches require orders of magnitude less communication than standard data-parallel training, in settings where the workers are datacenters, even the limited communication requirements of these approaches can still cause significant slow downs due to the blocking necessary at each outer optimization step. In this paper, we investigate techniques to mitigate this issue by overlapping communication with computation in a manner that allows the outer optimization step to fully overlap with the inner optimization phase. We show that a particular variant, dubbed eager updates, provides competitive performance with standard DiLoCo in settings with low bandwidth between workers. 

**Abstract (ZH)**: 分布式优化方法，如DiLoCo已被证明在多台分布式工作器（如数据中心）上训练非常大的模型时是有效的。这些方法将更新分成两个阶段：内部优化阶段，工人独立在其本地数据上执行多个优化步骤；外部优化阶段，将内部更新同步。尽管这样的方法所需的通信量比标准数据并行训练少好几个数量级，但在工人是数据中心的情况下，由于在每个外部优化步骤中的阻塞性通信需求，仍然会导致显著的性能下降。在本文中，我们研究了通过在计算与通信之间重叠的方式来减轻这一问题的技术，使得外部优化阶段能够完全与内部优化阶段重叠。我们证明了一种特定的变体——称为急切更新——在工人之间带宽较低的情况下，提供了与标准DiLoCo相当的性能。 

---
# B-cos LM: Efficiently Transforming Pre-trained Language Models for Improved Explainability 

**Title (ZH)**: B-cos LM：高效转换预训练语言模型以提高可解释性 

**Authors**: Yifan Wang, Sukrut Rao, Ji-Ung Lee, Mayank Jobanputra, Vera Demberg  

**Link**: [PDF](https://arxiv.org/pdf/2502.12992)  

**Abstract**: Post-hoc explanation methods for black-box models often struggle with faithfulness and human interpretability due to the lack of explainability in current neural models. Meanwhile, B-cos networks have been introduced to improve model explainability through architectural and computational adaptations, but their application has so far been limited to computer vision models and their associated training pipelines. In this work, we introduce B-cos LMs, i.e., B-cos networks empowered for NLP tasks. Our approach directly transforms pre-trained language models into B-cos LMs by combining B-cos conversion and task fine-tuning, improving efficiency compared to previous B-cos methods. Our automatic and human evaluation results demonstrate that B-cos LMs produce more faithful and human interpretable explanations than post hoc methods, while maintaining task performance comparable to conventional fine-tuning. Our in-depth analysis explores how B-cos LMs differ from conventionally fine-tuned models in their learning processes and explanation patterns. Finally, we provide practical guidelines for effectively building B-cos LMs based on our findings. Our code is available at this https URL. 

**Abstract (ZH)**: 黑盒模型的后验解释方法往往由于当前神经网络模型缺乏可解释性而难以保证忠实性和人类可理解性。与此同时，B-cos网络已经被引入以通过架构和计算上的改进来提高模型的可解释性，但其应用目前仅限于计算机视觉模型及其相关的训练管线。在本工作中，我们引入了B-cos LMs，即为NLP任务赋能的B-cos网络。我们的方法通过结合B-cos转换和任务微调，直接将预训练语言模型转换为B-cos LMs，相较于之前的B-cos方法，提高了效率。我们的自动评估和人工评估结果表明，B-cos LMs能够产生比后验方法更忠实且更具有人类可理解性的解释，同时保持与传统微调相当的任务性能。我们深入分析了B-cos LMs在学习过程和解释模式上如何不同于传统微调模型。最后，我们根据研究结果提供了构建B-cos LMs的实用指南。我们的代码可以在以下链接获取：this https URL。 

---
# Beyond Profile: From Surface-Level Facts to Deep Persona Simulation in LLMs 

**Title (ZH)**: 超越档案：从表面级事实到深度个性模拟在大语言模型中的实现 

**Authors**: Zixiao Wang, Duzhen Zhang, Ishita Agrawal, Shen Gao, Le Song, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.12988)  

**Abstract**: Previous approaches to persona simulation large language models (LLMs) have typically relied on learning basic biographical information, or using limited role-play dialogue datasets to capture a character's responses. However, a holistic representation of an individual goes beyond surface-level facts or conversations to deeper thoughts and thinking. In this work, we introduce CharacterBot, a model designed to replicate both the linguistic patterns and distinctive thought processes of a character. Using Lu Xun, a renowned Chinese writer, as a case study, we propose four training tasks derived from his 17 essay collections. These include a pre-training task focused on mastering external linguistic structures and knowledge, as well as three fine-tuning tasks: multiple-choice question answering, generative question answering, and style transfer, each aligning the LLM with Lu Xun's internal ideation and writing style. To optimize learning across these tasks, we introduce a CharLoRA parameter updating mechanism, where a general linguistic style expert collaborates with other task-specific experts to better study both the language style and the understanding of deeper thoughts. We evaluate CharacterBot on three tasks for linguistic accuracy and opinion comprehension, demonstrating that it significantly outperforms the baselines on our adapted metrics. We hope that this work inspires future research on deep character persona simulation LLM. 

**Abstract (ZH)**: 以下是经过学术规范翻译后的版本：

以往对人物模拟大规模语言模型（LLMs）的研究通常依赖于学习基本的生物信息，或使用有限的角色扮演对话数据集来捕捉角色的回应。然而，一个人的整体表现超越了表面事实或对话，深入到了更深层次的想法和思维过程。在本工作中，我们介绍了一种名为CharacterBot的模型，旨在复制角色的言语模式和独特的思维过程。以中国著名作家鲁迅为例，我们提出了四种训练任务，来源于他的17篇散文集。这些任务包括一种预训练任务，旨在掌握外部语言结构和知识，以及三种微调任务：多项选择题问答、生成性问答和风格转换，每种任务都与鲁迅的内心思考和写作风格保持一致。为了优化这些任务的学习，我们引入了一种CharLoRA参数更新机制，其中一种通用的语言风格专家与其他任务特定专家协同工作，以更好地研究语言风格和更深层次思维的理解。我们在语言准确性和观点理解三个任务上对CharacterBot进行了评估，结果显示，CharacterBot在我们调整后的评价标准上显著优于基线模型。我们希望这项工作能够启发未来对深层次人物角色模拟LLM的研究。 

---
# Sailor2: Sailing in South-East Asia with Inclusive Multilingual LLMs 

**Title (ZH)**: Sailor2：利用包容性多语言大语言模型探索东南亚地区 

**Authors**: Longxu Dou, Qian Liu, Fan Zhou, Changyu Chen, Zili Wang, Ziqi Jin, Zichen Liu, Tongyao Zhu, Cunxiao Du, Penghui Yang, Haonan Wang, Jiaheng Liu, Yongchi Zhao, Xiachong Feng, Xin Mao, Man Tsung Yeung, Kunat Pipatanakul, Fajri Koto, Min Si Thu, Hynek Kydlíček, Zeyi Liu, Qunshu Lin, Sittipong Sripaisarnmongkol, Kridtaphad Sae-Khow, Nirattisai Thongchim, Taechawat Konkaew, Narong Borijindargoon, Anh Dao, Matichon Maneegard, Phakphum Artkaew, Zheng-Xin Yong, Quan Nguyen, Wannaphong Phatthiyaphaibun, Hoang H. Tran, Mike Zhang, Shiqi Chen, Tianyu Pang, Chao Du, Xinyi Wan, Wei Lu, Min Lin  

**Link**: [PDF](https://arxiv.org/pdf/2502.12982)  

**Abstract**: Sailor2 is a family of cutting-edge multilingual language models for South-East Asian (SEA) languages, available in 1B, 8B, and 20B sizes to suit diverse applications. Building on Qwen2.5, Sailor2 undergoes continuous pre-training on 500B tokens (400B SEA-specific and 100B replay tokens) to support 13 SEA languages while retaining proficiency in Chinese and English. Sailor2-20B model achieves a 50-50 win rate against GPT-4o across SEA languages. We also deliver a comprehensive cookbook on how to develop the multilingual model in an efficient manner, including five key aspects: data curation, pre-training, post-training, model customization and evaluation. We hope that Sailor2 model (Apache 2.0 license) will drive language development in the SEA region, and Sailor2 cookbook will inspire researchers to build more inclusive LLMs for other under-served languages. 

**Abstract (ZH)**: Sailor2 是一系列针对东南亚（SEA）语言的先进多语言语言模型，提供1亿、8亿和20亿参数规模的版本，以适应各种应用需求。在Qwen2.5的基础上，Sailor2 经过连续预训练，包含500B词 Token（400B专用于SEA的语言 Token 和100B回放 Token），支持13种东南亚语言，同时保留对中文和英语的精通程度。Sailor2-20亿参数模型在东南亚语言方面与GPT-4o 的对战中实现了50-50的胜率。我们还提供了一份全面的指南，介绍如何高效开发多语言模型，包括五个关键方面：数据整理、预训练、后训练、模型定制和评估。我们希望Sailor2模型（采用Apache 2.0许可证）能够推动东南亚地区的语言发展，并希望Sailor2指南能够激励研究人员构建更多面向未服务语言的包容性大规模语言模型。 

---
# Reasoning-to-Defend: Safety-Aware Reasoning Can Defend Large Language Models from Jailbreaking 

**Title (ZH)**: 安全意识推理的防御作用：基于推理的安全意识能够防御大型语言模型免受 Jailbreaking 攻击 

**Authors**: Junda Zhu, Lingyong Yan, Shuaiqiang Wang, Dawei Yin, Lei Sha  

**Link**: [PDF](https://arxiv.org/pdf/2502.12970)  

**Abstract**: The reasoning abilities of Large Language Models (LLMs) have demonstrated remarkable advancement and exceptional performance across diverse domains. However, leveraging these reasoning capabilities to enhance LLM safety against adversarial attacks and jailbreak queries remains largely unexplored. To bridge this gap, we propose Reasoning-to-Defend (R2D), a novel training paradigm that integrates safety reflections of queries and responses into LLMs' generation process, unlocking a safety-aware reasoning mechanism. This approach enables self-evaluation at each reasoning step to create safety pivot tokens as indicators of the response's safety status. Furthermore, in order to improve the learning efficiency of pivot token prediction, we propose Contrastive Pivot Optimization(CPO), which enhances the model's ability to perceive the safety status of dialogues. Through this mechanism, LLMs dynamically adjust their response strategies during reasoning, significantly enhancing their defense capabilities against jailbreak attacks. Extensive experimental results demonstrate that R2D effectively mitigates various attacks and improves overall safety, highlighting the substantial potential of safety-aware reasoning in strengthening LLMs' robustness against jailbreaks. 

**Abstract (ZH)**: 大语言模型（LLMs）的推理能力在多个领域中取得了显著的进步和出色的表现。然而，如何利用这些推理能力来增强LLMs的安全性，特别是在对抗攻击和防止 jailbreak 查询的情况下，仍然较少被研究。为了弥合这一差距，我们提出了一种名为 Reasoning-to-Defend（R2D）的新颖训练范式，该范式将查询和响应的安全反思机制整合到LLMs的生成过程中，解锁了一种安全感知的推理机制。该方法使LLMs能够在每次推理步骤中进行自我评估，生成作为响应安全性状态指示的安全枢纽token。此外，为了提高枢纽token预测的学习效率，我们提出了一种对比性枢纽优化（CPO）方法，该方法增强了模型感知对话安全性状态的能力。通过这种机制，LLMs在推理过程中能够动态调整其响应策略，显著增强了其对jailbreak攻击的防御能力。广泛实验证明，R2D能够有效缓解各种攻击并提高整体安全性，突显了安全感知推理在增强LLMs抵御jailbreak攻击的稳健性方面的巨大潜力。 

---
# A Survey of Text Classification Under Class Distribution Shift 

**Title (ZH)**: 文本分类中类分布变迁的综述 

**Authors**: Adriana Valentina Costache, Silviu Florin Gheorghe, Eduard Gabriel Poesina, Paul Irofti, Radu Tudor Ionescu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12965)  

**Abstract**: The basic underlying assumption of machine learning (ML) models is that the training and test data are sampled from the same distribution. However, in daily practice, this assumption is often broken, i.e.~the distribution of the test data changes over time, which hinders the application of conventional ML models. One domain where the distribution shift naturally occurs is text classification, since people always find new topics to discuss. To this end, we survey research articles studying open-set text classification and related tasks. We divide the methods in this area based on the constraints that define the kind of distribution shift and the corresponding problem formulation, i.e.~learning with the Universum, zero-shot learning, and open-set learning. We next discuss the predominant mitigation approaches for each problem setup. Finally, we identify several future work directions, aiming to push the boundaries beyond the state of the art. Interestingly, we find that continual learning can solve many of the issues caused by the shifting class distribution. We maintain a list of relevant papers at this https URL. 

**Abstract (ZH)**: 机器学习（ML）模型的基本假设是训练数据和测试数据来自相同的分布。然而，在实际应用中，这一假设经常被打破，即测试数据的分布随时间变化，这阻碍了传统ML模型的应用。一种自然会出现分布变化的领域是文本分类，因为人们总是在不断发现新的话题。为此，我们调研了研究开放集文本分类及相关任务的文章。我们将该领域的研究方法基于定义分布变化类型的约束及其相应的问题表述进行分类，即基于Universum的学习、零样本学习和开放集学习。接下来，我们将讨论每种问题设置下的主要缓解方法。最后，我们确定了一些未来研究方向，旨在超越当前的技术水平。有趣的是，我们发现持续学习可以解决由类别分布变化引起的大多数问题。我们在此维护了一份相关论文列表：[请输入完整URL]。 

---
# Trust Me, I'm Wrong: High-Certainty Hallucinations in LLMs 

**Title (ZH)**: 《信赖我，我错了：大语言模型中的高确定性幻觉》 

**Authors**: Adi Simhi, Itay Itzhak, Fazl Barez, Gabriel Stanovsky, Yonatan Belinkov  

**Link**: [PDF](https://arxiv.org/pdf/2502.12964)  

**Abstract**: Large Language Models (LLMs) often generate outputs that lack grounding in real-world facts, a phenomenon known as hallucinations. Prior research has associated hallucinations with model uncertainty, leveraging this relationship for hallucination detection and mitigation. In this paper, we challenge the underlying assumption that all hallucinations are associated with uncertainty. Using knowledge detection and uncertainty measurement methods, we demonstrate that models can hallucinate with high certainty even when they have the correct knowledge. We further show that high-certainty hallucinations are consistent across models and datasets, distinctive enough to be singled out, and challenge existing mitigation methods. Our findings reveal an overlooked aspect of hallucinations, emphasizing the need to understand their origins and improve mitigation strategies to enhance LLM safety. The code is available at this https URL . 

**Abstract (ZH)**: 大型语言模型（LLMs）往往生成缺乏现实世界事实支持的输出，这一现象称为幻觉。先前的研究将幻觉与模型的不确定性联系起来，并利用这种关系进行幻觉检测和减轻。在本文中，我们挑战了所有幻觉都与不确定性相关的这一假设。通过知识检测和不确定性测量方法，我们证明即使模型拥有正确的知识，它们仍可能以高确定性产生幻觉。我们进一步表明，高确定性幻觉在不同模型和数据集中是一致的，具有足够的独特性，可以被识别，并挑战现有的减轻方法。我们的研究揭示了幻觉的一种未被忽视的方面，强调了理解其起源并改进减轻策略以提高LLM安全性的必要性。代码可在以下链接获取：this https URL。 

---
# Infinite Retrieval: Attention Enhanced LLMs in Long-Context Processing 

**Title (ZH)**: 无限检索：注意力增强的大语言模型在长上下文处理中的应用 

**Authors**: Xiaoju Ye, Zhichun Wang, Jingyuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12962)  

**Abstract**: Limited by the context window size of Large Language Models(LLMs), handling various tasks with input tokens exceeding the upper limit has been challenging, whether it is a simple direct retrieval task or a complex multi-hop reasoning task. Although various methods have been proposed to enhance the long-context processing capabilities of LLMs, they either incur substantial post-training costs, or require additional tool modules(e.g.,RAG), or have not shown significant improvement in realistic tasks. Our work observes the correlation between the attention distribution and generated answers across each layer, and establishes the attention allocation aligns with retrieval-augmented capabilities through experiments. Drawing on the above insights, we propose a novel method InfiniRetri that leverages the LLMs's own attention information to enable accurate retrieval across inputs of infinitely length. Our evaluations indicate that InfiniRetri achieves 100% accuracy in the Needle-In-a-Haystack(NIH) test over 1M tokens using a 0.5B parameter model, surpassing other method or larger models and setting a new state-of-the-art(SOTA). Moreover, our method achieves significant performance improvements on real-world benchmarks, with a maximum 288% improvement. In addition, InfiniRetri can be applied to any Transformer-based LLMs without additional training and substantially reduces inference latency and compute overhead in long texts. In summary, our comprehensive studies show InfiniRetri's potential for practical applications and creates a paradigm for retrievaling information using LLMs own capabilities under infinite-length tokens. Code will be released in link. 

**Abstract (ZH)**: 受限于大型语言模型（LLMs）的上下文窗口大小，当输入令牌超过上限时，处理各种任务（无论是简单的直接检索任务还是复杂的多跳推理任务）都具有挑战性。尽管提出了一系列增强LLMs长上下文处理能力的方法，但这些方法要么需要大量的后续训练成本，要么需要额外的工具模块（例如RAG），要么在实际任务中没有显示出显著改进。我们的工作观察了每层之间注意力分布与生成答案之间的关联，并通过实验建立了注意力分配与增强检索能力相一致。基于上述洞察，我们提出了一种名为InfiniRetri的新方法，利用LLMs自身的注意力信息，使系统能够准确检索无限长度的输入。评估结果显示，使用一个500M参数模型，InfiniRetri在100万令牌的Needle-In-a-Haystack（NIH）测试中达到了100%的准确率，超过了其他方法或更大规模的模型，并创下了新的最佳性能。此外，我们的方法在现实世界基准测试中实现了显著的性能提升，最高可达288%的改进。另外，InfiniRetri可以在不进行额外训练的情况下应用于任何基于Transformer的LLMs，并在长文本推理中显著减少了推理延迟和计算开销。总之，我们的全面研究表明InfiniRetri具有在实践中应用的潜力，并为使用LLMs自身能力检索无限长度文本信息建立了全新范式。代码将在链接中发布。 

---
# AlignFreeze: Navigating the Impact of Realignment on the Layers of Multilingual Models Across Diverse Languages 

**Title (ZH)**: AlignFreeze：重新对齐对多语言模型各层跨多种语言影响的导航研究 

**Authors**: Steve Bakos, Félix Gaschi, David Guzmán, Riddhi More, Kelly Chutong Li, En-Shiun Annie Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.12959)  

**Abstract**: Realignment techniques are often employed to enhance cross-lingual transfer in multilingual language models, still, they can sometimes degrade performance in languages that differ significantly from the fine-tuned source language. This paper introduces AlignFreeze, a method that freezes either the layers' lower half or upper half during realignment. Through controlled experiments on 4 tasks, 3 models, and in 35 languages, we find that realignment affects all the layers but can be the most detrimental to the lower ones. Freezing the lower layers can prevent performance degradation. Particularly, AlignFreeze improves Part-of-Speech (PoS) tagging performances in languages where full realignment fails: with XLM-R, it provides improvements of more than one standard deviation in accuracy in seven more languages than full realignment. 

**Abstract (ZH)**: 以下是对该论文内容或标题的翻译，符合学术规范：

在多语言语言模型中，重新对齐技术常被用来增强跨语言迁移能力，但有时在与微调源语言差异较大的语言中，重新对齐也可能导致性能下降。本文提出了一种名为AlignFreeze的方法，该方法在重新对齐过程中冻结层的下半部分或上半部分。通过在4个任务、3个模型和35种语言上进行受控实验，我们发现重新对齐会影响所有层，但对较低部分的影响尤为显著。冻结较低部分的层可以防止性能下降。特别是，与完全重新对齐相比，AlignFreeze在完全重新对齐失败的语言中提高了词性标注（PoS tagging）性能：使用XLM-R时，在7种更多语言中提高了超过一个标准差的准确率。 

---
# Task-Informed Anti-Curriculum by Masking Improves Downstream Performance on Text 

**Title (ZH)**: 任务导向的掩码反课程学习提高文本下游性能 

**Authors**: Andrei Jarca, Florinel Alin Croitoru, Radu Tudor Ionescu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12953)  

**Abstract**: Masked language modeling has become a widely adopted unsupervised technique to pre-train language models. However, the process of selecting tokens for masking is random, and the percentage of masked tokens is typically fixed for the entire training process. In this paper, we propose to adjust the masking ratio and to decide which tokens to mask based on a novel task-informed anti-curriculum learning scheme. First, we harness task-specific knowledge about useful and harmful tokens in order to determine which tokens to mask. Second, we propose a cyclic decaying masking ratio, which corresponds to an anti-curriculum schedule (from hard to easy). We exemplify our novel task-informed anti-curriculum by masking (TIACBM) approach across three diverse downstream tasks: sentiment analysis, text classification by topic, and authorship attribution. Our findings suggest that TIACBM enhances the ability of the model to focus on key task-relevant features, contributing to statistically significant performance gains across tasks. We release our code at this https URL. 

**Abstract (ZH)**: 掩码语言建模已成为一种广为采用的无监督技术，用于预训练语言模型。然而，选择用于掩码的词令牌的过程是随机的，掩码词令牌的比例在整个训练过程中通常是固定的。本文中，我们提出调整掩码比例，并基于一种新型任务导向的反阶梯学习方案来决定哪些词令牌需要被掩码。首先，我们利用特定任务的知识来确定哪些词令牌是有用的，哪些是无害的，从而决定哪些词令牌需要被掩码。其次，我们提出了一种递减的掩码比例循环方案，这对应于一种从难到易的反阶梯学习进度表。我们通过情感分析、主题分类和作者身份归因这三种不同的下游任务，展示了我们新型任务导向的反阶梯学习方法（TIACBM）的有效性。我们的研究结果表明，TIACBM有助于模型聚焦于关键任务相关的特征，从而在各个任务中取得统计显著性上的性能提升。我们已在以下链接发布了我们的代码：this https URL。 

---
# Every Expert Matters: Towards Effective Knowledge Distillation for Mixture-of-Experts Language Models 

**Title (ZH)**: 每位专家都重要：向着有效的混合专家语言模型知识蒸馏方向努力 

**Authors**: Gyeongman Kim, Gyouk Chu, Eunho Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12947)  

**Abstract**: With the emergence of Mixture-of-Experts (MoE), the efficient scaling of model size has accelerated the development of large language models in recent years. However, their high memory requirements prevent their use in resource-constrained environments. While knowledge distillation (KD) has been a proven method for model compression, its application to MoE teacher models remains underexplored. Through our investigation, we discover that non-activated experts in MoE models possess valuable knowledge that benefits student models. We further demonstrate that existing KD methods are not optimal for compressing MoE models, as they fail to leverage this knowledge effectively. To address this, we propose two intuitive MoE-specific KD methods for the first time: Knowledge Augmentation (KA) and Student-Aware Router (SAR), both designed to effectively extract knowledge from all experts. Specifically, KA augments knowledge by sampling experts multiple times, while SAR uses all experts and adjusts the expert weights through router training to provide optimal knowledge. Extensive experiments show that our methods outperform conventional KD methods, demonstrating their effectiveness for MoE teacher models. 

**Abstract (ZH)**: 随着Mixture-of-Experts（MoE）的兴起，模型规模的有效扩大加速了近年来大型语言模型的发展。然而，它们对内存的高需求限制了它们在资源受限环境中的应用。虽然知识蒸馏（KD）已被证实是一种有效的模型压缩方法，但其在MoE教师模型中的应用仍处于探索阶段。通过我们的研究，我们发现MoE模型中未被激活的专家也蕴含着有益于学生模型的知识。进一步研究表明，现有的KD方法对于压缩MoE模型并不理想，因为它们未能有效地利用这些知识。为解决这一问题，我们首次提出了两种直观的MoE特定KD方法：知识扩展（KA）和学生导向路由器（SAR），旨在有效地从所有专家中提取知识。具体来说，KA通过多次采样专家来进行知识扩展，而SAR利用所有专家，并通过路由训练调整专家权重以提供最优知识。广泛的实验表明，我们的方法优于传统的KD方法，证明了它们在MoE教师模型中的有效性。 

---
# LLMPopcorn: An Empirical Study of LLMs as Assistants for Popular Micro-video Generation 

**Title (ZH)**: LLMPopcorn：大型语言模型作为流行微视频生成助手的实证研究 

**Authors**: Junchen Fu, Xuri Ge, Kaiwen Zheng, Ioannis Arapakis, Xin Xin, Joemon M. Jose  

**Link**: [PDF](https://arxiv.org/pdf/2502.12945)  

**Abstract**: Popular Micro-videos, dominant on platforms like TikTok and YouTube, hold significant commercial value. The rise of high-quality AI-generated content has spurred interest in AI-driven micro-video creation. However, despite the advanced capabilities of large language models (LLMs) like ChatGPT and DeepSeek in text generation and reasoning, their potential to assist the creation of popular micro-videos remains largely unexplored.
In this paper, we conduct an empirical study on LLM-assisted popular micro-video generation (LLMPopcorn). Specifically, we investigate the following research questions: (i) How can LLMs be effectively utilized to assist popular micro-video generation? (ii) To what extent can prompt-based enhancements optimize the LLM-generated content for higher popularity? (iii) How well do various LLMs and video generators perform in the popular micro-video generation task? By exploring these questions, we show that advanced LLMs like DeepSeek-V3 enable micro-video generation to achieve popularity comparable to human-created content. Prompt enhancements further boost popularity, and benchmarking highlights DeepSeek-V3 and DeepSeek-R1 among LLMs, while LTX-Video and HunyuanVideo lead in video generation. This pioneering work advances AI-assisted micro-video creation, uncovering new research opportunities. We will release the code and datasets to support future studies. 

**Abstract (ZH)**: 流行的微视频，在TikTok和YouTube等平台上占据主导地位，具有显著的商业价值。高质量的人工智能生成内容的兴起激发了对基于人工智能的微视频创作的兴趣。然而，尽管类似ChatGPT和DeepSeek这样的大规模语言模型（LLMs）在文本生成和推理方面具有先进的能力，它们在辅助流行微视频创作方面的潜力尚未得到充分探索。

在本论文中，我们将进行一项关于LLM辅助流行微视频生成（LLMPopcorn）的实证研究。具体而言，我们将探讨以下研究问题：（i）如何有效地利用LLM来辅助流行微视频的生成？（ii）基于提示的增强可以多大程度上优化LLM生成的内容，以提高其流行度？（iii）在流行微视频生成任务中，各种LLM和视频生成器的表现如何？通过探索这些问题，我们展示了像DeepSeek-V3这样的先进LLM可以使微视频生成达到与人类创建内容相似的流行度。提示增强进一步提升了流行度，基准测试显示DeepSeek-V3和DeepSeek-R1在LLM中表现最佳，而LTX-Video和HunyuanVideo在视频生成方面表现出色。这项开创性的工作推进了AI辅助微视频创作领域，并发现了新的研究机会。我们将发布代码和数据集以支持未来的研究。 

---
# Synthetic Data Generation for Culturally Nuanced Commonsense Reasoning in Low-Resource Languages 

**Title (ZH)**: 面向低资源语言的文化细微差分辨common-sense推理的合成数据生成 

**Authors**: Salsabila Zahirah Pranida, Rifo Ahmad Genadi, Fajri Koto  

**Link**: [PDF](https://arxiv.org/pdf/2502.12932)  

**Abstract**: Quantifying reasoning capability in low-resource languages remains a challenge in NLP due to data scarcity and limited access to annotators. While LLM-assisted dataset construction has proven useful for medium- and high-resource languages, its effectiveness in low-resource languages, particularly for commonsense reasoning, is still unclear. In this paper, we compare three dataset creation strategies: (1) LLM-assisted dataset generation, (2) machine translation, and (3) human-written data by native speakers, to build a culturally nuanced story comprehension dataset. We focus on Javanese and Sundanese, two major local languages in Indonesia, and evaluate the effectiveness of open-weight and closed-weight LLMs in assisting dataset creation through extensive manual validation. To assess the utility of synthetic data, we fine-tune language models on classification and generation tasks using this data and evaluate performance on a human-written test set. Our findings indicate that LLM-assisted data creation outperforms machine translation. 

**Abstract (ZH)**: 在自然语言处理（NLP）领域，由于数据稀缺和标注者访问有限，低资源语言的推理能力量化仍是一项挑战。虽然大规模语言模型（LLM）辅助的数据集构建对中资源和高资源语言证明了其有用性，但其在低资源语言中的有效性，特别是在常识推理方面，仍然不清楚。在这篇论文中，我们比较了三种数据集创建策略：（1）LLM辅助的数据集生成，（2）机器翻译，以及（3）本地语言母语者的手工编写数据，以构建一个文化上精细化的故事理解数据集。我们重点关注印度尼西亚的两大地方语言爪哇语和巽他语，并通过广泛的manual验证评估开放权重和封闭权重LLM在辅助数据集构建中的有效性。为了评估合成数据的实用性，我们使用这些数据对语言模型进行了分类和生成任务的微调，并在由母语者编写的测试集上评估了性能。我们的研究发现表明，LLM辅助的数据创建优于机器翻译。 

---
# Finedeep: Mitigating Sparse Activation in Dense LLMs via Multi-Layer Fine-Grained Experts 

**Title (ZH)**: FineDeeP：通过多层细粒度专家机制减轻密集大模型中的稀疏激活 

**Authors**: Leiyu Pan, Zhenpeng Su, Minxuan Lv, Yizhe Xiong, Xiangwen Zhang, Zijia Lin, Hui Chen, Jungong Han, Guiguang Ding, Cheng Luo, Di Zhang, Kun Gai, Deyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2502.12928)  

**Abstract**: Large language models have demonstrated exceptional performance across a wide range of tasks. However, dense models usually suffer from sparse activation, where many activation values tend towards zero (i.e., being inactivated). We argue that this could restrict the efficient exploration of model representation space. To mitigate this issue, we propose Finedeep, a deep-layered fine-grained expert architecture for dense models. Our framework partitions the feed-forward neural network layers of traditional dense models into small experts, arranges them across multiple sub-layers. A novel routing mechanism is proposed to determine each expert's contribution. We conduct extensive experiments across various model sizes, demonstrating that our approach significantly outperforms traditional dense architectures in terms of perplexity and benchmark performance while maintaining a comparable number of parameters and floating-point operations. Moreover, we find that Finedeep achieves optimal results when balancing depth and width, specifically by adjusting the number of expert sub-layers and the number of experts per sub-layer. Empirical results confirm that Finedeep effectively alleviates sparse activation and efficiently utilizes representation capacity in dense models. 

**Abstract (ZH)**: 大型语言模型在各种任务上展示了出色的表现。然而，密集模型通常会遭受激活稀疏性的问题，其中很多激活值趋向于零（即被激活状态）。我们认为这可能限制了模型表示空间的高效探索。为了解决这一问题，我们提出了Finedeep，一种为密集模型设计的深层细粒度专家架构。我们的框架将传统密集模型的前向神经网络层划分为小型专家，并在多个子层中进行排列。我们提出了一种新颖的路由机制来确定每个专家的贡献。我们在各种模型规模上进行了广泛的实验，结果表明，我们的方法在困惑度和基准性能方面显著优于传统的密集架构，同时保持了相近的参数数量和浮点运算次数。此外，我们发现Finedeep在深度和宽度之间取得最佳效果，具体表现在调整子层中的专家数量以及每个子层中的专家数量。实验证据表明，Finedeep有效地缓解了激活稀疏性问题，并在密集模型中高效利用了表示容量。 

---
# SEFL: Harnessing Large Language Model Agents to Improve Educational Feedback Systems 

**Title (ZH)**: SEFL：利用大型语言模型代理提升教育反馈系统 

**Authors**: Mike Zhang, Amalie Pernille Dilling, Léon Gondelman, Niels Erik Ruan Lyngdorf, Euan D. Lindsay, Johannes Bjerva  

**Link**: [PDF](https://arxiv.org/pdf/2502.12927)  

**Abstract**: Providing high-quality feedback is crucial for student success but is constrained by time, cost, and limited data availability. We introduce Synthetic Educational Feedback Loops (SEFL), a novel framework designed to deliver immediate, on-demand feedback at scale without relying on extensive, real-world student data. In SEFL, two large language models (LLMs) operate in teacher--student roles to simulate assignment completion and formative feedback, generating abundant synthetic pairs of student work and corresponding critiques. We then fine-tune smaller, more computationally efficient LLMs on these synthetic pairs, enabling them to replicate key features of high-quality, goal-oriented feedback. Unlike personalized tutoring approaches that offer multi-turn, individualized instruction, SEFL specifically focuses on replicating the teacher-->student feedback loop for diverse assignments. Through both LLM-as-a-judge and human evaluations, we demonstrate that SEFL-tuned models outperform their non-tuned counterparts in feedback quality, clarity, and timeliness. These findings reveal SEFL's potential to transform feedback processes for higher education and beyond, offering an ethical and scalable alternative to conventional manual feedback cycles. 

**Abstract (ZH)**: 提供高质量的反馈对于学生成绩至关重要，但受到时间、成本和可用数据有限的限制。我们提出了一种新的框架——合成教育反馈循环（SEFL），旨在无需依赖大量真实世界的学生数据的情况下，大规模提供即时反馈。在SEFL中，两个大型语言模型（LLMs）分别扮演教师和学生的角色，模拟作业完成和形成性反馈的过程，生成大量的合成学生作品及其相应的批注。随后，我们将这些合成数据对应用于更小、计算效率更高的LLMs进行微调，使这些模型能够复制高质量、目标导向性反馈的关键特征。与提供个性化多回合个别指导的个性化辅导方法不同，SEFL特别关注模拟教师到学生的反馈循环，适用于多种类型的作业。通过对LLMs作为裁判和人类评估的双重测试，我们证明了SEFL微调模型在反馈质量、清晰度和及时性方面优于非微调模型。这些研究发现展示了SEFL在高等教育乃至其他领域的反馈过程变革潜力，提供了一种伦理上可接受且可扩展的常规手动反馈循环的替代方案。 

---
# Conditioning LLMs to Generate Code-Switched Text: A Methodology Grounded in Naturally Occurring Data 

**Title (ZH)**: 基于自然发生数据的条件生成代码切换文本的大型语言模型方法论 

**Authors**: Maite Heredia, Gorka Labaka, Jeremy Barnes, Aitor Soroa  

**Link**: [PDF](https://arxiv.org/pdf/2502.12924)  

**Abstract**: Code-switching (CS) is still a critical challenge in Natural Language Processing (NLP). Current Large Language Models (LLMs) struggle to interpret and generate code-switched text, primarily due to the scarcity of large-scale CS datasets for training. This paper presents a novel methodology to generate CS data using LLMs, and test it on the English-Spanish language pair. We propose back-translating natural CS sentences into monolingual English, and using the resulting parallel corpus to fine-tune LLMs to turn monolingual sentences into CS. Unlike previous approaches to CS generation, our methodology uses natural CS data as a starting point, allowing models to learn its natural distribution beyond grammatical patterns. We thoroughly analyse the models' performance through a study on human preferences, a qualitative error analysis and an evaluation with popular automatic metrics. Results show that our methodology generates fluent code-switched text, expanding research opportunities in CS communication, and that traditional metrics do not correlate with human judgement when assessing the quality of the generated CS data. We release our code and generated dataset under a CC-BY-NC-SA license. 

**Abstract (ZH)**: 代码混用（Code-switching, CS）仍然是自然语言处理（NLP）中的一个关键性挑战。当前的大规模语言模型（LLMs）难以理解和生成代码混用文本，主要原因是对训练这种文本的大规模数据集稀缺。本文提出了一种新的方法，通过LLMs生成CS数据，并在英西语言对上进行了测试。我们提议将自然的CS句子反向翻译成单一语言的英语，并使用生成的平行语料库对LLMs进行微调，使其能够将单一语言句子转化为CS。与之前的CS生成方法不同，我们的方法以自然的CS数据作为起点，从而使模型能够学习其自然分布模式，而不仅仅是语法模式。我们通过一项针对人类喜好的研究、定性的错误分析以及使用流行的自动评价指标进行评估，全面分析了模型的表现。结果表明，我们的方法能够生成流畅的代码混用文本，扩大了在CS通信领域的研究机会，并且传统的评价指标与人类判断在评估生成的CS数据质量时并无相关性。我们发布了我们的代码和生成的数据集，许可证为CC-BY-NC-SA。 

---
# On-Device LLMs for Home Assistant: Dual Role in Intent Detection and Response Generation 

**Title (ZH)**: 家庭助手中的本地设备大语言模型：意图检测与响应生成的双重角色 

**Authors**: Rune Birkmose, Nathan Mørkeberg Reece, Esben Hofstedt Norvin, Johannes Bjerva, Mike Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12923)  

**Abstract**: This paper investigates whether Large Language Models (LLMs), fine-tuned on synthetic but domain-representative data, can perform the twofold task of (i) slot and intent detection and (ii) natural language response generation for a smart home assistant, while running solely on resource-limited, CPU-only edge hardware. We fine-tune LLMs to produce both JSON action calls and text responses. Our experiments show that 16-bit and 8-bit quantized variants preserve high accuracy on slot and intent detection and maintain strong semantic coherence in generated text, while the 4-bit model, while retaining generative fluency, suffers a noticeable drop in device-service classification accuracy. Further evaluations on noisy human (non-synthetic) prompts and out-of-domain intents confirm the models' generalization ability, obtaining around 80--86\% accuracy. While the average inference time is 5--6 seconds per query -- acceptable for one-shot commands but suboptimal for multi-turn dialogue -- our results affirm that an on-device LLM can effectively unify command interpretation and flexible response generation for home automation without relying on specialized hardware. 

**Abstract (ZH)**: 本论文探讨了是否可以通过在合成但领域代表性数据上微调的大规模语言模型（LLMs），在仅依赖资源有限的CPU边沿硬件上完成两项任务：（i）槽位和意图检测，以及（ii）生成自然语言响应。我们对LLMs进行微调，使其既能生成JSON动作调用，又能生成文本响应。实验结果显示，16位和8位量化版本在槽位和意图检测上保持了高精度，并在生成的文本中保持了较强的语义连贯性，而4位模型虽然在生成流畅性上有所保留，但在设备-服务分类精度上却出现了明显的下降。进一步在噪声人类（非合成）提示和领域外意图上的评估确认了这些模型的泛化能力，达到了约80-86%的准确率。虽然平均推理时间约为每查询5-6秒——对于单轮指令是可以接受的，但对于多轮对话则不够优化，但我们的结果表明，设备上的LLMs能够有效地统一命令解析和灵活响应生成，无需依赖专门的硬件。 

---
# Q-STRUM Debate: Query-Driven Contrastive Summarization for Recommendation Comparison 

**Title (ZH)**: Q-STRUM 辩论：查询驱动的对比总结推荐比较 

**Authors**: George-Kirollos Saad, Scott Sanner  

**Link**: [PDF](https://arxiv.org/pdf/2502.12921)  

**Abstract**: Query-driven recommendation with unknown items poses a challenge for users to understand why certain items are appropriate for their needs. Query-driven Contrastive Summarization (QCS) is a methodology designed to address this issue by leveraging language-based item descriptions to clarify contrasts between them. However, existing state-of-the-art contrastive summarization methods such as STRUM-LLM fall short of this goal. To overcome these limitations, we introduce Q-STRUM Debate, a novel extension of STRUM-LLM that employs debate-style prompting to generate focused and contrastive summarizations of item aspects relevant to a query. Leveraging modern large language models (LLMs) as powerful tools for generating debates, Q-STRUM Debate provides enhanced contrastive summaries. Experiments across three datasets demonstrate that Q-STRUM Debate yields significant performance improvements over existing methods on key contrastive summarization criteria, thus introducing a novel and performant debate prompting methodology for QCS. 

**Abstract (ZH)**: 基于查询的推荐系统往往面临一个难题：当系统中包含未知项目时，用户难以理解为何某些项目适合其需求。基于查询的对比总结（QCS）是一种方法论，通过利用语言描述的项目信息来阐明项目间的对比特性，以解决这一问题。然而，现有的前沿对比总结方法，如STRUM-LLM，无法完全实现这一目标。为克服这些局限性，我们提出了一种名为Q-STRUM Debate的创新扩展方法，它采用辩论式提示生成与查询相关的专注且对比鲜明的总结。借助现代大型语言模型（LLMs）作为生成辩论的强大工具，Q-STRUM Debate能够提供增强的对比总结。在三个数据集上的实验表明，Q-STRUM Debate在关键对比总结标准上的表现显著优于现有方法，从而引入了一种新颖且高效的辩论式提示方法，为QCS提供支持。 

---
# Knapsack Optimization-based Schema Linking for LLM-based Text-to-SQL Generation 

**Title (ZH)**: 基于背包优化的模式链接技术在大语言模型驱动的文本到SQL生成中的应用 

**Authors**: Zheng Yuan, Hao Chen, Zijin Hong, Qinggang Zhang, Feiran Huang, Xiao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12911)  

**Abstract**: Generating SQLs from user queries is a long-standing challenge, where the accuracy of initial schema linking significantly impacts subsequent SQL generation performance. However, current schema linking models still struggle with missing relevant schema elements or an excess of redundant ones. A crucial reason for this is that commonly used metrics, recall and precision, fail to capture relevant element missing and thus cannot reflect actual schema linking performance. Motivated by this, we propose an enhanced schema linking metric by introducing a restricted missing indicator. Accordingly, we introduce Knapsack optimization-based Schema Linking Agent (KaSLA), a plug-in schema linking agent designed to prevent the missing of relevant schema elements while minimizing the inclusion of redundant ones. KaSLA employs a hierarchical linking strategy that first identifies the optimal table linking and subsequently links columns within the selected table to reduce linking candidate space. In each linking process, it utilize a knapsack optimization approach to link potentially relevant elements while accounting for a limited tolerance of potential redundant this http URL this optimization, KaSLA-1.6B achieves superior schema linking results compared to large-scale LLMs, including deepseek-v3 with state-of-the-art (SOTA) schema linking method. Extensive experiments on Spider and BIRD benchmarks verify that KaSLA can significantly improve the SQL generation performance of SOTA text-to-SQL models by substituting their schema linking processes. 

**Abstract (ZH)**: 将用户查询转换为SQL语句是一个长期存在的挑战，其中初始模式链接的准确性显著影响后续SQL生成性能。然而，目前的模式链接模型仍然难以应对缺失的相关模式元素或冗余元素过多的问题。其中一个关键原因是，常用的召回率和精确率度量无法捕捉到缺失的相关元素，因此不能反映实际的模式链接性能。为了解决这个问题，我们提出了一种增强的模式链接度量，通过引入一个受限的缺失指示器来改进。据此，我们引入了基于背包优化的模式链接代理（KaSLA），这是一个插件式模式链接代理，旨在防止缺失相关模式元素的同时，尽量减少冗余元素的包含。KaSLA 使用一种分层连接策略，首先识别最优的表连接，然后将选定的表中的列进行连接以减少候选连接空间。在每次连接过程中，它利用背包优化方法链接潜在的相关元素，同时考虑到潜在冗余的有限容忍度。通过这种优化，KaSLA-1.6B 超过了大规模语言模型（如具有最新方法的深搜三），实现了更优的模式链接结果。大规模实验表明，KaSLA 可显著提高顶级文本到SQL模型的SQL生成性能，通过替换它们的模式链接过程。 

---
# Fraud-R1 : A Multi-Round Benchmark for Assessing the Robustness of LLM Against Augmented Fraud and Phishing Inducements 

**Title (ZH)**: Fraud-R1：评估大语言模型在增强型欺诈和钓鱼诱导面前的稳健性的多轮基准测试 

**Authors**: Shu Yang, Shenzhe Zhu, Zeyu Wu, Keyu Wang, Junchi Yao, Junchao Wu, Lijie Hu, Mengdi Li, Derek F. Wong, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12904)  

**Abstract**: We introduce Fraud-R1, a benchmark designed to evaluate LLMs' ability to defend against internet fraud and phishing in dynamic, real-world scenarios. Fraud-R1 comprises 8,564 fraud cases sourced from phishing scams, fake job postings, social media, and news, categorized into 5 major fraud types. Unlike previous benchmarks, Fraud-R1 introduces a multi-round evaluation pipeline to assess LLMs' resistance to fraud at different stages, including credibility building, urgency creation, and emotional manipulation. Furthermore, we evaluate 15 LLMs under two settings: 1. Helpful-Assistant, where the LLM provides general decision-making assistance, and 2. Role-play, where the model assumes a specific persona, widely used in real-world agent-based interactions. Our evaluation reveals the significant challenges in defending against fraud and phishing inducement, especially in role-play settings and fake job postings. Additionally, we observe a substantial performance gap between Chinese and English, underscoring the need for improved multilingual fraud detection capabilities. 

**Abstract (ZH)**: 我们将介绍一种名为 Fraud-R1 的基准测试，用以评估大型语言模型（LLM）在动态且真实的网络诈骗和钓鱼场景中抵御欺诈的能力。Fraud-R1 包含来自 phishing 欺诈、虚假招聘广告、社交媒体和新闻的 8,564 个诈骗案例，并按 5 种主要诈骗类型分类。与之前的基准测试不同，Fraud-R1 引入了多轮评估流程，以在不同的欺诈阶段评估 LLM 的抗欺诈性，包括建立可信度、制造紧迫感以及情感操控。此外，我们在两种设定下评估了 15 个 LLM：1. 在“助手机器人”模式下，LLM 提供一般的决策辅助；2. 在“角色扮演”模式下，模型假设特定的人格，这是在真实世界基于代理的交互中广泛使用的方式。我们的评估揭示了在抵御诈骗和钓鱼诱导方面的重要挑战，特别是在“角色扮演”模式和虚假招聘广告环境中。另外，我们发现中文和英文之间存在显著的性能差异，突显了改进多语言诈骗检测能力的需求。 

---
# Soundwave: Less is More for Speech-Text Alignment in LLMs 

**Title (ZH)**: Soundwave: 简约胜繁复——在大规模语言模型中实现语音-文本对齐 

**Authors**: Yuhao Zhang, Zhiheng Liu, Fan Bu, Ruiyu Zhang, Benyou Wang, Haizhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.12900)  

**Abstract**: Existing end-to-end speech large language models (LLMs) usually rely on large-scale annotated data for training, while data-efficient training has not been discussed in depth. We focus on two fundamental problems between speech and text: the representation space gap and sequence length inconsistency. We propose Soundwave, which utilizes an efficient training strategy and a novel architecture to address these issues. Results show that Soundwave outperforms the advanced Qwen2-Audio in speech translation and AIR-Bench speech tasks, using only one-fiftieth of the training data. Further analysis shows that Soundwave still retains its intelligence during conversation. The project is available at this https URL. 

**Abstract (ZH)**: 现有的端到端语音大语言模型（LLMs）通常依赖大规模标注数据进行训练，而数据高效训练尚未得到深入探讨。我们专注于语音与文本之间两个基本问题：表示空间差距和序列长度不一致。我们提出了一种名为Soundwave的方法，它利用高效的训练策略和新型架构来解决这些问题。实验结果表明，与先进的Qwen2-Audio相比，Soundwave在语音翻译和AIR-Bench语音任务上表现更优，仅使用了五分之一的训练数据。进一步的分析表明，Soundwave在对话过程中仍能保持其智能。项目详情请参见：https://your-project-url.com 

---
# None of the Others: a General Technique to Distinguish Reasoning from Memorization in Multiple-Choice LLM Evaluation Benchmarks 

**Title (ZH)**: 《无他：一种区分选择题大语言模型评估基准中推理与记忆的方法》 

**Authors**: Eva Sánchez Salido, Julio Gonzalo, Guillermo Marco  

**Link**: [PDF](https://arxiv.org/pdf/2502.12896)  

**Abstract**: In LLM evaluations, reasoning is often distinguished from recall/memorization by performing numerical variations to math-oriented questions. Here we introduce a general variation method for multiple-choice questions that completely dissociates the correct answer from previously seen tokens or concepts, requiring LLMs to understand and reason (rather than memorizing) in order to answer correctly. Using this method, we evaluate state-of-the-art proprietary and open-source LLMs on two datasets available in English and Spanish: the public MMLU benchmark and the private UNED-Access 2024 dataset. Results show that all models experience remarkable accuracy drops under our proposed variation, with an average loss of 57% on MMLU and 50% on UNED-Access 2024, ranging from 10% to 93% across models. Notably, the most accurate model in our experimentation (OpenAI-o3-mini) is not the most robust (DeepSeek-R1-70B), suggesting that the best models in standard evaluations may not be the ones with better reasoning capabilities. Also, we see larger accuracy drops in public (vs private) datasets and questions posed in their original language (vs a manual translation), which are signs of contamination and also point to a relevant role of recall/memorization in current LLMs' answers. 

**Abstract (ZH)**: 在大规模语言模型（LLM）的评估中，通过在数学倾向的问题上进行数值变化，通常将推理与回忆/记忆区分开来。在这里，我们介绍了一种适用于多项选择题的通用变化方法，这种方法完全将正确答案与之前见过的词汇或概念隔离开来，要求LLM们理解并推理（而非记忆）以正确作答。通过这种方法，我们将最先进的封闭源代码和开源LLM在两个可用英语和西班牙语的数据集上进行了评估：公共的MMLU基准和私人UNED-Access 2024数据集。结果表明，在我们提出的变式下，所有模型的准确率都有显著下降，MMLU数据集的平均降幅为57%，UNED-Access 2024数据集的平均降幅为50%，各模型之间的降幅范围从10%到93%不等。值得注意的是，在我们的实验中，最准确的模型（OpenAI-o3-mini）并不是最稳健的模型（DeepSeek-R1-70B），这表明在标准评估中表现最好的模型可能并不具备更好的推理能力。此外，我们还发现公共数据集（相对于私人数据集）和原始语言问题（相对于人工翻译的问题）的准确率下降更大，这表明当前LLM的回答中可能存在污染，也显示了记忆/回忆在这些模型中的重要作用。 

---
# Multilingual European Language Models: Benchmarking Approaches and Challenges 

**Title (ZH)**: 多语言欧洲语言模型：基准测试方法与挑战 

**Authors**: Fabio Barth, Georg Rehm  

**Link**: [PDF](https://arxiv.org/pdf/2502.12895)  

**Abstract**: The breakthrough of generative large language models (LLMs) that can solve different tasks through chat interaction has led to a significant increase in the use of general benchmarks to assess the quality or performance of these models beyond individual applications. There is also a need for better methods to evaluate and also to compare models due to the ever increasing number of new models published. However, most of the established benchmarks revolve around the English language. This paper analyses the benefits and limitations of current evaluation datasets, focusing on multilingual European benchmarks. We analyse seven multilingual benchmarks and identify four major challenges. Furthermore, we discuss potential solutions to enhance translation quality and mitigate cultural biases, including human-in-the-loop verification and iterative translation ranking. Our analysis highlights the need for culturally aware and rigorously validated benchmarks to assess the reasoning and question-answering capabilities of multilingual LLMs accurately. 

**Abstract (ZH)**: 生成式大规模语言模型（LLMs）在通过对话解决不同任务方面的突破，导致了对通用基准测试的需求显著增加，以评估这些模型的质量或性能，而不仅仅是基于单个应用。此外，由于不断有新的模型被发布，对更好的评价方法和比较方法的需求也随之增加。然而，大多数现有的基准测试主要集中在英语语言上。本文分析了当前评价数据集的优势和限制，重点关注多语言欧洲基准测试。我们分析了七个多语言基准测试，并确定了四大主要挑战。此外，我们讨论了提高翻译质量并减轻文化偏见的潜在解决方案，包括人工在环验证和迭代翻译排序。我们的分析突出了需要具备文化意识并经过严格验证的基准测试，以准确评估多语言LLMs的推理和问答能力。 

---
# H-CoT: Hijacking the Chain-of-Thought Safety Reasoning Mechanism to Jailbreak Large Reasoning Models, Including OpenAI o1/o3, DeepSeek-R1, and Gemini 2.0 Flash Thinking 

**Title (ZH)**: H-CoT：劫持链式思维安全推理机制以突破大型推理模型，包括OpenAI的o1/o3、DeepSeek-R1和Gemini 2.0 Flash Thinking 

**Authors**: Martin Kuo, Jianyi Zhang, Aolin Ding, Qinsi Wang, Louis DiValentin, Yujia Bao, Wei Wei, Da-Cheng Juan, Hai Li, Yiran Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.12893)  

**Abstract**: Large Reasoning Models (LRMs) have recently extended their powerful reasoning capabilities to safety checks-using chain-of-thought reasoning to decide whether a request should be answered. While this new approach offers a promising route for balancing model utility and safety, its robustness remains underexplored. To address this gap, we introduce Malicious-Educator, a benchmark that disguises extremely dangerous or malicious requests beneath seemingly legitimate educational prompts. Our experiments reveal severe security flaws in popular commercial-grade LRMs, including OpenAI o1/o3, DeepSeek-R1, and Gemini 2.0 Flash Thinking. For instance, although OpenAI's o1 model initially maintains a high refusal rate of about 98%, subsequent model updates significantly compromise its safety; and attackers can easily extract criminal strategies from DeepSeek-R1 and Gemini 2.0 Flash Thinking without any additional tricks. To further highlight these vulnerabilities, we propose Hijacking Chain-of-Thought (H-CoT), a universal and transferable attack method that leverages the model's own displayed intermediate reasoning to jailbreak its safety reasoning mechanism. Under H-CoT, refusal rates sharply decline-dropping from 98% to below 2%-and, in some instances, even transform initially cautious tones into ones that are willing to provide harmful content. We hope these findings underscore the urgent need for more robust safety mechanisms to preserve the benefits of advanced reasoning capabilities without compromising ethical standards. 

**Abstract (ZH)**: 大型推理模型（LRMs）最近将其强大的推理能力扩展到安全性检查中，使用链式推理来决定是否回答请求。虽然这种方法为平衡模型的实用性和安全性提供了潜力，但其鲁棒性仍被严重忽视。为应对这一差距，我们提出了一个名为“恶意教育者”（Malicious-Educator）的基准测试，该基准测试将极其危险或恶意的请求隐藏在表面上看似合法的教育提示之下。我们的实验揭示了广泛使用的商品级大型推理模型中严重的安全漏洞，包括OpenAI的o1/o3模型、DeepSeek-R1以及Gemini 2.0闪思。例如，尽管OpenAI的o1模型最初拒绝率高达约98%，但随后的模型更新严重削弱了其安全性；攻击者可以从DeepSeek-R1和Gemini 2.0闪思中轻松提取犯罪策略，无需额外的技巧。为了进一步突出这些漏洞，我们提出了“劫持链式推理”（H-CoT），这是一种普遍适用且可转移的攻击方法，利用模型自身展示的中间推理过程，打破其安全性推理机制。在H-CoT下，拒绝率急剧下降，从98%降至低于2%；在某些情况下，甚至可以将原本谨慎的态度转变为愿意提供有害内容的立场。希望这些发现能够强调建立更 robust 安全机制的迫切需求，以保护高级推理能力带来的利益，同时不妥协道德标准。 

---
# Are Multilingual Language Models an Off-ramp for Under-resourced Languages? Will we arrive at Digital Language Equality in Europe in 2030? 

**Title (ZH)**: 多语言语言模型是资源匮乏语言的出路吗？到2030年，欧洲能否实现数字语言平权？ 

**Authors**: Georg Rehm, Annika Grützner-Zahn, Fabio Barth  

**Link**: [PDF](https://arxiv.org/pdf/2502.12886)  

**Abstract**: Large language models (LLMs) demonstrate unprecedented capabilities and define the state of the art for almost all natural language processing (NLP) tasks and also for essentially all Language Technology (LT) applications. LLMs can only be trained for languages for which a sufficient amount of pre-training data is available, effectively excluding many languages that are typically characterised as under-resourced. However, there is both circumstantial and empirical evidence that multilingual LLMs, which have been trained using data sets that cover multiple languages (including under-resourced ones), do exhibit strong capabilities for some of these under-resourced languages. Eventually, this approach may have the potential to be a technological off-ramp for those under-resourced languages for which "native" LLMs, and LLM-based technologies, cannot be developed due to a lack of training data. This paper, which concentrates on European languages, examines this idea, analyses the current situation in terms of technology support and summarises related work. The article concludes by focusing on the key open questions that need to be answered for the approach to be put into practice in a systematic way. 

**Abstract (ZH)**: 大型语言模型（LLMs）展示了前所未有的能力，并几乎定义了所有自然语言处理（NLP）任务和技术语言学（LT）应用的前沿。LLMs 只能针对有足够预训练数据的语言进行训练，这实际上排除了许多通常被视为资源不足的语言。然而，存在间接证据和实证证据表明，多语言 LLMs（它们使用涵盖了多种语言的数据集进行训练，包括资源不足的语言）对于某些资源不足的语言确实展示出强大的能力。最终，这种做法可能具有技术救生机会，对于那些由于缺乏训练数据而无法开发“原生”LLMs 和基于LLM 的技术的语言来说，这可能是一条替代途径。本文专注于欧洲语言，探讨了这一理念，分析了当前的技术支持情况，并总结了相关工作。文章最后集中讨论了实施这种方法时需要回答的关键开放问题。 

---
# How desirable is alignment between LLMs and linguistically diverse human users? 

**Title (ZH)**: 语言模型与语言多样的人类用户之间的一致性有多 desirable？ 

**Authors**: Pia Knoeferle, Sebastian Möller, Dorothea Kolossa, Veronika Solopova, Georg Rehm  

**Link**: [PDF](https://arxiv.org/pdf/2502.12884)  

**Abstract**: We discuss how desirable it is that Large Language Models (LLMs) be able to adapt or align their language behavior with users who may be diverse in their language use. User diversity may come about among others due to i) age differences; ii) gender characteristics, and/or iii) multilingual experience, and associated differences in language processing and use. We consider potential consequences for usability, communication, and LLM development. 

**Abstract (ZH)**: 我们探讨大型语言模型（LLMs）能否适应或与具有语言使用多样性（包括但不限于）的用户进行调整或对齐，这在多大程度上是必要的。用户多样性可能来源于：i) 年龄差异；ii) 性别特征；iii) 多语言经验，以及由此带来的语言处理和使用上的差异。我们考虑这种适应性或对齐可能对易用性、交流以及LLM的发展产生的潜在影响。 

---
# PAFT: Prompt-Agnostic Fine-Tuning 

**Title (ZH)**: PAFT：提示无感微调 

**Authors**: Chenxing Wei, Yao Shu, Mingwen Ou, Ying Tiffany He, Fei Richard Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12859)  

**Abstract**: While Large Language Models (LLMs) adapt well to downstream tasks after fine-tuning, this adaptability often compromises prompt robustness, as even minor prompt variations can significantly degrade performance. To address this, we propose Prompt-Agnostic Fine-Tuning(PAFT), a simple yet effective approach that dynamically adjusts prompts during fine-tuning. This encourages the model to learn underlying task principles rather than overfitting to specific prompt formulations. PAFT operates in two stages: First, a diverse set of meaningful, synthetic candidate prompts is constructed. Second, during fine-tuning, prompts are randomly sampled from this set to create dynamic training inputs. Extensive experiments across diverse datasets and LLMs demonstrate that models trained with PAFT exhibit strong robustness and generalization across a wide range of prompts, including unseen ones. This enhanced robustness improves both model performance and inference speed while maintaining training efficiency. Ablation studies further confirm the effectiveness of PAFT. 

**Abstract (ZH)**: 虽然在微调后，大规模语言模型（LLMs）能够很好地适应下游任务，但这种适应性往往会牺牲提示的稳健性，因为即使是细微的提示变化也可能显著降低模型的性能。为了解决这一问题，我们提出了一种简单而有效的提示无关联微调（PAFT）方法，该方法在微调过程中动态调整提示。这种方法鼓励模型学习潜在的任务原则，而不是过度拟合特定的提示形式。PAFT 在两个阶段进行：首先，构建一个包含多样化且有意义的合成提示候选集；其次，在微调过程中，从这个提示集中随机抽取提示，以创建动态训练输入。我们在多个多样化的数据集和语言模型上的广泛实验表明，使用PAFT训练的模型在各种提示（包括未见过的提示）下展现出强大的稳健性和泛化能力，这一增强的稳健性不仅提升了模型性能，还加快了推理速度，同时保持了训练效率。进一步的消融研究表明，PAFT的有效性得到了验证。 

---
# Rejected Dialects: Biases Against African American Language in Reward Models 

**Title (ZH)**: 被拒斥的方言：奖励模型中的非洲美国语言偏见 

**Authors**: Joel Mire, Zubin Trivadi Aysola, Daniel Chechelnitsky, Nicholas Deas, Chrysoula Zerva, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2502.12858)  

**Abstract**: Preference alignment via reward models helps build safe, helpful, and reliable large language models (LLMs). However, subjectivity in preference judgments and the lack of representative sampling in preference data collection can introduce new biases, hindering reward models' fairness and equity. In this work, we introduce a framework for evaluating dialect biases in reward models and conduct a case study on biases against African American Language (AAL) through several experiments comparing reward model preferences and behavior on paired White Mainstream English (WME) and both machine-translated and human-written AAL corpora. We show that reward models are less aligned with human preferences when processing AAL texts vs. WME ones (-4\% accuracy on average), frequently disprefer AAL-aligned texts vs. WME-aligned ones, and steer conversations toward WME, even when prompted with AAL texts. Our findings provide a targeted analysis of anti-AAL biases at a relatively understudied stage in LLM development, highlighting representational harms and ethical questions about the desired behavior of LLMs concerning AAL. 

**Abstract (ZH)**: 偏好对齐通过奖励模型有助于构建安全、有用且可靠的大型语言模型（LLMs）。然而，在偏好判断中的主观性以及偏好数据收集中缺乏代表性样本采集，可能会引入新的偏差，从而阻碍奖励模型的公平性和平等性。在此项工作中，我们提出了一种评估奖励模型中方言偏差的框架，并通过一系列实验，对比了奖励模型在处理非洲裔美国人语言（AAL）文本和标准白人英语（WME）文本时的偏好和行为，进行了一项案例研究。结果显示，当处理AAL文本而非WME文本时，奖励模型的准确率平均降低了4%，常常偏好WME对齐的文本而非AAL对齐的文本，并且即使在被提示使用AAL文本的情况下，也会引导对话走向WME。我们的研究结果为LLM开发中相对未被充分研究的一个阶段提供了有针对性的分析，揭示了代表性危害，并指出了关于LLM对AAL期望行为的伦理问题。 

---
# Integrating Arithmetic Learning Improves Mathematical Reasoning in Smaller Models 

**Title (ZH)**: 将算术学习集成到小型模型中可以提高数学推理能力 

**Authors**: Neeraj Gangwar, Suma P Bhat, Nickvash Kani  

**Link**: [PDF](https://arxiv.org/pdf/2502.12855)  

**Abstract**: While large models pre-trained on high-quality data exhibit excellent performance across various reasoning tasks, including mathematical reasoning (e.g. GSM8k, MultiArith), specializing smaller models to excel at mathematical reasoning remains a challenging problem. Common approaches to address this challenge include knowledge distillation, where smaller student models learn from large pre-trained teacher models, and data augmentation, such as rephrasing questions. Despite these efforts, smaller models struggle with arithmetic computations, leading to errors in mathematical reasoning. In this work, we focus on leveraging a programmatically generated arithmetic dataset to enhance the reasoning capabilities of smaller models. We investigate two key approaches to incorporate this dataset -- (1) intermediate fine-tuning, where a model is fine-tuned on the arithmetic dataset before being trained on a reasoning dataset, and (2) integrating the arithmetic dataset into the instruction-tuning mixture, allowing the model to learn arithmetic skills alongside general instruction-following abilities. Our experiments on multiple reasoning benchmarks demonstrate that incorporating an arithmetic dataset, whether through targeted fine-tuning or within the instruction-tuning mixture, enhances the models' arithmetic capabilities, which in turn improves their mathematical reasoning performance. 

**Abstract (ZH)**: 尽管在高质量数据上进行预训练的大模型在各种推理任务中表现出色，包括数学推理（例如GSM8K、MultiArith），但对于如何使较小的模型在数学推理方面表现出色仍是一个具有挑战性的问题。为解决这一问题，常见的方法包括知识蒸馏，即较小的学生模型从大型预训练教师模型中学习，以及数据增强，例如重新表述问题。尽管采取了这些措施，但较小的模型在算术计算方面仍存在问题，导致数学推理中的错误。本研究旨在利用程序生成的算术数据集来提升较小模型的推理能力。我们研究了两种关键方法来整合这个数据集——(1) 中介微调，即将模型在算术数据集上进行微调后再在推理数据集上进行训练；(2) 将算术数据集整合进指令调整混合模型中，使模型能够在遵循一般指令的同时学习算术技能。我们在多个推理基准上的实验表明，通过目标性微调或整合进指令调整混合模型来引入算术数据集，能够增强模型的算术能力，进而提高其数学推理性能。 

---
# S$^2$R: Teaching LLMs to Self-verify and Self-correct via Reinforcement Learning 

**Title (ZH)**: S$^2$R：通过强化学习使大语言模型学会自我验证和自我修正 

**Authors**: Ruotian Ma, Peisong Wang, Cheng Liu, Xingyan Liu, Jiaqi Chen, Bang Zhang, Xin Zhou, Nan Du, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.12853)  

**Abstract**: Recent studies have demonstrated the effectiveness of LLM test-time scaling. However, existing approaches to incentivize LLMs' deep thinking abilities generally require large-scale data or significant training efforts. Meanwhile, it remains unclear how to improve the thinking abilities of less powerful base models. In this work, we introduce S$^2$R, an efficient framework that enhances LLM reasoning by teaching models to self-verify and self-correct during inference. Specifically, we first initialize LLMs with iterative self-verification and self-correction behaviors through supervised fine-tuning on carefully curated data. The self-verification and self-correction skills are then further strengthened by both outcome-level and process-level reinforcement learning, with minimized resource requirements, enabling the model to adaptively refine its reasoning process during inference. Our results demonstrate that, with only 3.1k self-verifying and self-correcting behavior initialization samples, Qwen2.5-math-7B achieves an accuracy improvement from 51.0\% to 81.6\%, outperforming models trained on an equivalent amount of long-CoT distilled data. Extensive experiments and analysis based on three base models across both in-domain and out-of-domain benchmarks validate the effectiveness of S$^2$R. Our code and data are available at this https URL. 

**Abstract (ZH)**: 近年来的研究表明，LLM的测试时缩放技术是有效的。然而，现有激励LLM深入思考能力的方法通常需要大量的数据或显著的训练努力。同时，如何提高较弱基础模型的思考能力仍然是一个未解之谜。本文中，我们提出了S$^2$R，这是一个高效的框架，通过教导模型在推理过程中自我验证和自我纠正来增强其推理能力。具体而言，我们首先通过在精心策划的数据上进行监督微调，初始化LLM的迭代自我验证和自我纠正行为。然后，通过结果层面和过程层面的强化学习进一步加强这两种技能，同时尽量减少资源需求，使模型能够适应性地在其推理过程中进行自我修正和优化。我们的实验结果表明，仅通过使用3100个自我验证和自我纠正行为的初始化样本，Qwen2.5-math-7B 的准确率从51.0%提高到81.6%，并且超过了在同等数量的长级链推理精化数据上进行训练的模型。基于三种不同基础模型的广泛实验和分析在内部和外部基准测试中的结果验证了S$^2$R的有效性。我们的代码和数据可以在以下链接找到：this https URL。 

---
# MVL-SIB: A Massively Multilingual Vision-Language Benchmark for Cross-Modal Topical Matching 

**Title (ZH)**: MVL-SIB：一种大规模多语言跨模态主题匹配基准 

**Authors**: Fabian David Schmidt, Florian Schneider, Chris Biemann, Goran Glavaš  

**Link**: [PDF](https://arxiv.org/pdf/2502.12852)  

**Abstract**: Existing multilingual vision-language (VL) benchmarks often only cover a handful of languages. Consequently, evaluations of large vision-language models (LVLMs) predominantly target high-resource languages, underscoring the need for evaluation data for low-resource languages. To address this limitation, we introduce MVL-SIB, a massively multilingual vision-language benchmark that evaluates both cross-modal and text-only topical matching across 205 languages -- over 100 more than the most multilingual existing VL benchmarks encompass. We then benchmark a range of of open-weight LVLMs together with GPT-4o(-mini) on MVL-SIB. Our results reveal that LVLMs struggle in cross-modal topic matching in lower-resource languages, performing no better than chance on languages like N'Koo. Our analysis further reveals that VL support in LVLMs declines disproportionately relative to textual support for lower-resource languages, as evidenced by comparison of cross-modal and text-only topical matching performance. We further observe that open-weight LVLMs do not benefit from representing a topic with more than one image, suggesting that these models are not yet fully effective at handling multi-image tasks. By correlating performance on MVL-SIB with other multilingual VL benchmarks, we highlight that MVL-SIB serves as a comprehensive probe of multilingual VL understanding in LVLMs. 

**Abstract (ZH)**: 现有的多语言视觉-语言（VL）基准通常仅涵盖少数几种语言。因此，对于大规模视觉-语言模型（LVLM）的评估大多集中在高资源语言上，这凸显了为低资源语言提供评估数据的必要性。为了解决这一局限性，我们引入了MVL-SIB，这是一个涵盖205种语言的巨量多语言视觉-语言基准，比现有最多样化的多语言VL基准多出逾100种语言，同时涵盖了跨模态和纯文本主题匹配。随后，我们对一系列开放参数的LVLM及GPT-4o(-mini)在MVL-SIB上的性能进行了评估。我们的结果显示，LVLM在低资源语言的跨模态主题匹配中表现糟糕，甚至在如N’Koo这样的语言上表现不如随机猜测。进一步的分析表明，与纯文本支持相比，LVLM对低资源语言的跨模态支持下降更为显著，这一点通过跨模态和纯文本主题匹配性能的对比可以体现。我们还观察到，开放参数的LVLM在使用多张图像表示一个主题时并未获益，这表明这些模型尚未完全有效地处理多图像任务。通过将MVL-SIB上的性能与其它多语言VL基准的相关性进行关联，我们强调MVL-SIB是评估LVLM多语言VL理解能力的一个全面探针。 

---
# MeMo: Towards Language Models with Associative Memory Mechanisms 

**Title (ZH)**: MeMo：面向关联记忆机制的语言模型 

**Authors**: Fabio Massimo Zanzotto, Elena Sofia Ruzzetti, Giancarlo A. Xompero, Leonardo Ranaldi, Davide Venditti, Federico Ranaldi, Cristina Giannone, Andrea Favalli, Raniero Romagnoli  

**Link**: [PDF](https://arxiv.org/pdf/2502.12851)  

**Abstract**: Memorization is a fundamental ability of Transformer-based Large Language Models, achieved through learning. In this paper, we propose a paradigm shift by designing an architecture to memorize text directly, bearing in mind the principle that memorization precedes learning. We introduce MeMo, a novel architecture for language modeling that explicitly memorizes sequences of tokens in layered associative memories. By design, MeMo offers transparency and the possibility of model editing, including forgetting texts. We experimented with the MeMo architecture, showing the memorization power of the one-layer and the multi-layer configurations. 

**Abstract (ZH)**: 记忆能力是基于 Transformer 的大型语言模型的一项基本能力，通过学习获得。本文提出了一种范式转变，通过设计一种直接记忆文本的架构，秉承记忆先于学习的原则。我们提出了一种名为 MeMo 的新型语言模型架构，该架构在分层关联记忆中显式地记忆序列的令牌。设计上，MeMo 提供了透明性和模型编辑的可能性，包括忘记文本。我们对 MeMo 架构进行了实验，展示了单层和多层配置的记忆能力。 

---
# An LLM-Powered Agent for Physiological Data Analysis: A Case Study on PPG-based Heart Rate Estimation 

**Title (ZH)**: 基于PPG的心率估计：由一个由LLM驱动的代理进行生理数据测算的案例研究 

**Authors**: Mohammad Feli, Iman Azimi, Pasi Liljeberg, Amir M.Rahmani  

**Link**: [PDF](https://arxiv.org/pdf/2502.12836)  

**Abstract**: Large language models (LLMs) are revolutionizing healthcare by improving diagnosis, patient care, and decision support through interactive communication. More recently, they have been applied to analyzing physiological time-series like wearable data for health insight extraction. Existing methods embed raw numerical sequences directly into prompts, which exceeds token limits and increases computational costs. Additionally, some studies integrated features extracted from time-series in textual prompts or applied multimodal approaches. However, these methods often produce generic and unreliable outputs due to LLMs' limited analytical rigor and inefficiency in interpreting continuous waveforms. In this paper, we develop an LLM-powered agent for physiological time-series analysis aimed to bridge the gap in integrating LLMs with well-established analytical tools. Built on the OpenCHA, an open-source LLM-powered framework, our agent features an orchestrator that integrates user interaction, data sources, and analytical tools to generate accurate health insights. To evaluate its effectiveness, we implement a case study on heart rate (HR) estimation from Photoplethysmogram (PPG) signals using a dataset of PPG and Electrocardiogram (ECG) recordings in a remote health monitoring study. The agent's performance is benchmarked against OpenAI GPT-4o-mini and GPT-4o, with ECG serving as the gold standard for HR estimation. Results demonstrate that our agent significantly outperforms benchmark models by achieving lower error rates and more reliable HR estimations. The agent implementation is publicly available on GitHub. 

**Abstract (ZH)**: 大规模语言模型（LLMs）正在通过改进诊断、患者护理和决策支持，以交互交流的方式革新医疗行业。近年来，它们已被应用于分析生理时间序列数据（如穿戴设备数据），以提取健康洞察。当前方法直接将原始数值序列嵌入提示中，这超出了标记限制并增加了计算成本。此外，一些研究将时间序列特征嵌入文本提示中，或采用多模态方法。然而，由于LLMs在分析和解释连续波形方面的能力有限，这些方法往往会产生通用且不可靠的输出。在本研究中，我们开发了一个基于LLMs的代理工具，旨在弥合将LLMs与成熟的分析工具集成的差距。基于一个开放源代码的LLMs框架OpenCHA，我们的代理工具具有一个调度程序，该调度程序整合用户交互、数据源和分析工具，以生成准确的健康洞察。为评估其有效性，我们在远程健康监测研究中，使用光电容积描记仪（PPG）和心电图（ECG）记录数据集，实现了心率（HR）从PPG信号估计的案例研究。该代理的性能与OpenAI GPT-4o-mini和GPT-4o进行了基准测试，ECG作为HR估计的金标准。结果表明，我们的代理工具在准确率和HR估计的可靠性方面显著优于基准模型。该代理的实现已在GitHub上开源。 

---
# Subword models struggle with word learning, but surprisal hides it 

**Title (ZH)**: 子词模型在单词学习方面存在困难，但 surprisal 隐藏了这一点 

**Authors**: Bastian Bunzeck, Sina Zarrieß  

**Link**: [PDF](https://arxiv.org/pdf/2502.12835)  

**Abstract**: We study word learning in subword and character language models with the psycholinguistic lexical decision task. While subword LMs struggle to discern words and non-words with high accuracy, character LMs solve this task easily and consistently. Furthermore, when comparing word learning and syntactic learning, both processes are separable in character LM where word learning predates syntactic learning, whereas these processes are simultaneous in subword LM. This raises questions about the adequacy of subword LMs for modeling language acquisition and positions character LMs as a viable alternative. 

**Abstract (ZH)**: 我们使用心理语言学的词汇判断任务研究了子词和字符语言模型中的单词学习。虽然子词语言模型在高准确率地区分单词和非单词方面面临困难，但字符语言模型能够轻松且一致地解决这一任务。此外，在对比单词学习与句法学习时，字符语言模型中的这两个过程是可以区分开的，其中单词学习先于句法学习；而在子词语言模型中，这两个过程则是同时发生的。这一发现引发了关于子词语言模型在语言习得建模中的适用性的质疑，并将字符语言模型定位为一种可行的替代选择。 

---
# KazMMLU: Evaluating Language Models on Kazakh, Russian, and Regional Knowledge of Kazakhstan 

**Title (ZH)**: KazMMLU：评估 Kazakh 语言模型在喀山知识、俄语以及哈萨克斯坦区域知识上的表现

这里的翻译保持了原文的学术风格，同时也确保了句子的流畅性和准确性。KazMMLU 是一个评估语言模型在 Kazakh（哈萨克语）、俄语以及与哈萨克斯坦相关的知识上的性能的评估标准或数据集的名称，保持不变。 

**Authors**: Mukhammed Togmanov, Nurdaulet Mukhituly, Diana Turmakhan, Jonibek Mansurov, Maiya Goloburda, Akhmed Sakip, Zhuohan Xie, Yuxia Wang, Bekassyl Syzdykov, Nurkhan Laiyk, Alham Fikri Aji, Ekaterina Kochmar, Preslav Nakov, Fajri Koto  

**Link**: [PDF](https://arxiv.org/pdf/2502.12829)  

**Abstract**: Despite having a population of twenty million, Kazakhstan's culture and language remain underrepresented in the field of natural language processing. Although large language models (LLMs) continue to advance worldwide, progress in Kazakh language has been limited, as seen in the scarcity of dedicated models and benchmark evaluations. To address this gap, we introduce KazMMLU, the first MMLU-style dataset specifically designed for Kazakh language. KazMMLU comprises 23,000 questions that cover various educational levels, including STEM, humanities, and social sciences, sourced from authentic educational materials and manually validated by native speakers and educators. The dataset includes 10,969 Kazakh questions and 12,031 Russian questions, reflecting Kazakhstan's bilingual education system and rich local context. Our evaluation of several state-of-the-art multilingual models (Llama-3.1, Qwen-2.5, GPT-4, and DeepSeek V3) demonstrates substantial room for improvement, as even the best-performing models struggle to achieve competitive performance in Kazakh and Russian. These findings underscore significant performance gaps compared to high-resource languages. We hope that our dataset will enable further research and development of Kazakh-centric LLMs. Data and code will be made available upon acceptance. 

**Abstract (ZH)**: 尽管哈萨克斯坦拥有2000万人口，但其文化和语言在自然语言处理领域中的表现仍然相对不足。尽管大规模语言模型（LLMs）在全球范围内不断发展，但哈萨克语领域的进展有限，这体现在缺乏专门的模型和基准评估方面。为填补这一空白，我们提出了KazMMLU，这是首个专门针对哈萨克语的MMLU样式数据集。KazMMLU包含23,000道题目，涵盖了从STEM到人文学科和社会科学等各个教育水平，这些题目来源于真实的教育材料，并由母语者和教育工作者手动验证。数据集包括10,969道哈萨克语题目和12,031道俄语题目，反映了哈萨克斯坦的双语教育体系及其丰富的本地背景。我们对几种最先进的多语言模型（Llama-3.1、Qwen-2.5、GPT-4 和 DeepSeek V3）进行了评估，并发现即使是最优秀的模型在哈萨克语和俄语上的表现也难以达到竞争力。这些发现强调了与高资源语言相比存在显著的性能差距。我们希望我们的数据集能够促进针对哈萨克语的进一步研究和模型开发。数据和代码将在接受后提供。 

---
# Reasoning and the Trusting Behavior of DeepSeek and GPT: An Experiment Revealing Hidden Fault Lines in Large Language Models 

**Title (ZH)**: 深度搜索和GPT的推理与信任行为：一项揭示大型语言模型潜在缺陷的实验研究 

**Authors**: Rubing Lu, João Sedoc, Arun Sundararajan  

**Link**: [PDF](https://arxiv.org/pdf/2502.12825)  

**Abstract**: When encountering increasingly frequent performance improvements or cost reductions from a new large language model (LLM), developers of applications leveraging LLMs must decide whether to take advantage of these improvements or stay with older tried-and-tested models. Low perceived switching frictions can lead to choices that do not consider more subtle behavior changes that the transition may induce. Our experiments use a popular game-theoretic behavioral economics model of trust to show stark differences in the trusting behavior of OpenAI's and DeepSeek's models. We highlight a collapse in the economic trust behavior of the o1-mini and o3-mini models as they reconcile profit-maximizing and risk-seeking with future returns from trust, and contrast it with DeepSeek's more sophisticated and profitable trusting behavior that stems from an ability to incorporate deeper concepts like forward planning and theory-of-mind. As LLMs form the basis for high-stakes commercial systems, our results highlight the perils of relying on LLM performance benchmarks that are too narrowly defined and suggest that careful analysis of their hidden fault lines should be part of any organization's AI strategy. 

**Abstract (ZH)**: 在遇到来自新型大型语言模型（LLM）日益频繁的性能改进或成本降低时，利用LLM的应用开发者必须决定是否利用这些改进或继续使用更成熟但可能欠新的模型。感知到的转换摩擦较低可能导致做出不考虑过渡过程中可能引起的更微妙行为变化的选择。我们的实验利用了一个流行的行为经济学博弈理论模型来展示OpenAI的和DeepSeek的模型在信任行为上的显著差异。我们强调了o1-mini和o3-mini模型在权衡利润最大化和风险偏好与未来信任收益时，信任经济行为的崩溃，并将其与DeepSeek更具复杂性和盈利能力的信任行为进行对比，后者得益于将更深层次的概念，如前瞻性计划和理论之思维纳入考虑。由于LLM构成了高风险商业系统的基础，我们的研究结果揭示了依赖于定义过于狭窄的LLM性能基准的危害，并建议对潜在的技术缺陷进行仔细分析应成为任何组织人工智能战略的一部分。 

---
# Pitfalls of Scale: Investigating the Inverse Task of Redefinition in Large Language Models 

**Title (ZH)**: 规模的陷阱：探究大型语言模型中重新定义的逆向任务 

**Authors**: Elena Stringli, Maria Lymperaiou, Giorgos Filandrianos, Giorgos Stamou  

**Link**: [PDF](https://arxiv.org/pdf/2502.12821)  

**Abstract**: Inverse tasks can uncover potential reasoning gaps as Large Language Models (LLMs) scale up. In this work, we explore the redefinition task, in which we assign alternative values to well-known physical constants and units of measure, prompting LLMs to respond accordingly. Our findings show that not only does model performance degrade with scale, but its false confidence also rises. Moreover, while factors such as prompting strategies or response formatting are influential, they do not preclude LLMs from anchoring to memorized values. 

**Abstract (ZH)**: 随着大规模语言模型（LLMs）的规模扩大，逆向任务可以揭示潜在的推理缺口。在本研究中，我们探讨了一种重新定义任务，即我们为熟知的物理常数和度量单位赋予替代值，促使LLMs据此作出回应。我们的研究发现，模型的性能随着规模的扩大而下降，其错误的信心也随之增加。此外，虽然诸如提示策略或响应格式等因素具有影响作用，但它们并不能阻止LLMs锚定于记忆中的值。 

---
# Simulating User Diversity in Task-Oriented Dialogue Systems using Large Language Models 

**Title (ZH)**: 使用大型语言模型模拟任务导向对话系统中的用户多样性 

**Authors**: Adnan Ahmad, Stefan Hillmann, Sebastian Möller  

**Link**: [PDF](https://arxiv.org/pdf/2502.12813)  

**Abstract**: In this study, we explore the application of Large Language Models (LLMs) for generating synthetic users and simulating user conversations with a task-oriented dialogue system and present detailed results and their analysis. We propose a comprehensive novel approach to user simulation technique that uses LLMs to create diverse user profiles, set goals, engage in multi-turn dialogues, and evaluate the conversation success. We employ two proprietary LLMs, namely GPT-4o and GPT-o1 (Achiam et al., 2023), to generate a heterogeneous base of user profiles, characterized by varied demographics, multiple user goals, different conversational styles, initial knowledge levels, interests, and conversational objectives. We perform a detailed analysis of the user profiles generated by LLMs to assess the diversity, consistency, and potential biases inherent in these LLM-generated user simulations. We find that GPT-o1 generates more heterogeneous user distribution across most user attributes, while GPT-4o generates more skewed user attributes. The generated set of user profiles are then utilized to simulate dialogue sessions by interacting with a task-oriented dialogue system. 

**Abstract (ZH)**: 在本研究中，我们探讨了大型语言模型（LLMs）在生成合成用户及使用面向任务的对话系统模拟用户对话中的应用，并详细呈现了相关结果及分析。我们提出了一种全面且新颖的用户模拟方法，利用LLMs创建多样化的用户画像，设定目标，进行多轮对话，并评估对话的成功率。我们使用了两个内部开发的LLMs，即GPT-4o和GPT-o1（Achiam等，2023），来生成具有多种特征的用户基础画像，包括不同的年龄、性别、教育程度、职业、收入水平、多重用户目标、对话风格、初始知识水平、兴趣以及对话目标。我们对LLMs生成的用户画像进行了详细分析，以评估这些LLM生成的用户模拟在多样性、一致性和潜在偏见方面的特征。研究发现，GPT-o1在大多数用户属性上生成了更具有多样性的用户分布，而GPT-4o则在某些用户属性上生成了更偏斜的用户分布。所生成的用户画像随后被用于通过与面向任务的对话系统互动来模拟对话会话。 

---
# Towards Text-Image Interleaved Retrieval 

**Title (ZH)**: 面向文本-图像交织检索的研究 

**Authors**: Xin Zhang, Ziqi Dai, Yongqi Li, Yanzhao Zhang, Dingkun Long, Pengjun Xie, Meishan Zhang, Jun Yu, Wenjie Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12799)  

**Abstract**: Current multimodal information retrieval studies mainly focus on single-image inputs, which limits real-world applications involving multiple images and text-image interleaved content. In this work, we introduce the text-image interleaved retrieval (TIIR) task, where the query and document are interleaved text-image sequences, and the model is required to understand the semantics from the interleaved context for effective retrieval. We construct a TIIR benchmark based on naturally interleaved wikiHow tutorials, where a specific pipeline is designed to generate interleaved queries. To explore the task, we adapt several off-the-shelf retrievers and build a dense baseline by interleaved multimodal large language model (MLLM). We then propose a novel Matryoshka Multimodal Embedder (MME), which compresses the number of visual tokens at different granularity, to address the challenge of excessive visual tokens in MLLM-based TIIR models. Experiments demonstrate that simple adaption of existing models does not consistently yield effective results. Our MME achieves significant improvements over the baseline by substantially fewer visual tokens. We provide extensive analysis and will release the dataset and code to facilitate future research. 

**Abstract (ZH)**: 当前的多模态信息检索研究主要集中在单张图片的输入上，这限制了涉及多张图片和图文交错内容的实际应用。在这项工作中，我们引入了图文交错检索（TIIR，Text-Image Interleaved Retrieval）任务，其中查询和文档是交错的文本-图片序列，模型需要理解交错上下文中的语义以实现有效的检索。我们基于自然交错的wikiHow教程构建了一个TIIR基准，设计了一个特定的流水线来生成交错查询。为了探索该任务，我们调整了几种现成的检索器，并通过交错的多模态大型语言模型(MLLM)构建了一个密集基线。然后，我们提出了一种新颖的马特罗什卡多模态嵌入器(MME)，它在不同粒度上压缩了视觉令牌的数量，以解决基于MLLM的TIIR模型中视觉令牌过多的问题。实验表明，现有模型的简单调整并不总是能取得有效结果。我们的MME通过大幅减少视觉令牌的数量在基线的基础上实现了显著的提升。我们进行了详尽的分析，并将发布该数据集和代码，以促进未来的研究。 

---
# Commonsense Reasoning in Arab Culture 

**Title (ZH)**: 阿拉伯文化中的常识推理 

**Authors**: Abdelrahman Sadallah, Junior Cedric Tonga, Khalid Almubarak, Saeed Almheiri, Farah Atif, Chatrine Qwaider, Karima Kadaoui, Sara Shatnawi, Yaser Alesh, Fajri Koto  

**Link**: [PDF](https://arxiv.org/pdf/2502.12788)  

**Abstract**: Despite progress in Arabic large language models, such as Jais and AceGPT, their evaluation on commonsense reasoning has largely relied on machine-translated datasets, which lack cultural depth and may introduce Anglocentric biases. Commonsense reasoning is shaped by geographical and cultural contexts, and existing English datasets fail to capture the diversity of the Arab world. To address this, we introduce \datasetname, a commonsense reasoning dataset in Modern Standard Arabic (MSA), covering cultures of 13 countries across the Gulf, Levant, North Africa, and the Nile Valley. The dataset was built from scratch by engaging native speakers to write and validate culturally relevant questions for their respective countries. \datasetname spans 12 daily life domains with 54 fine-grained subtopics, reflecting various aspects of social norms, traditions, and everyday experiences. Zero-shot evaluations show that open-weight language models with up to 32B parameters struggle to comprehend diverse Arab cultures, with performance varying across regions. These findings highlight the need for more culturally aware models and datasets tailored to the Arabic-speaking world. 

**Abstract (ZH)**: 尽管在阿拉伯大型语言模型方面取得了一定进展，如Jais和AceGPT，但它们在常识推理方面的评估很大程度上依赖于机器翻译的数据集，这些数据集缺乏文化深度，可能存在英文化偏见。常识推理受地理和文化背景的影响，现有英语数据集未能捕捉阿拉伯世界的多样性。为解决这一问题，我们引入了\datasetname，这是一个以现代标准阿拉伯语（MSA）编写的常识推理数据集，涵盖了来自海湾地区、黎凡特、北非和尼罗河谷13个国家的文化。数据集从零开始构建，通过与母语者合作，为各自国家编写并验证具有文化相关性的问题。\datasetname覆盖了12个日常生活领域，包括54个细粒度子话题，反映了社会规范、传统和日常生活体验的各种方面。零样本评估表明，具有多达320亿参数的开放权重语言模型在理解多样化的阿拉伯文化方面存在困难，地区间的性能有所差异。这些发现突显了需要更多文化意识更强的模型和数据集，这些模型和数据集能够针对阿拉伯语世界进行定制。 

---
# Mind the Gap: Aligning the Brain with Language Models Requires a Nonlinear and Multimodal Approach 

**Title (ZH)**: 注意差距：将大脑与语言模型对齐需要非线性和多模态的方法 

**Authors**: Danny Dongyeop Han, Yunju Cho, Jiook Cha, Jay-Yoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.12771)  

**Abstract**: Self-supervised language and audio models effectively predict brain responses to speech. However, traditional prediction models rely on linear mappings from unimodal features, despite the complex integration of auditory signals with linguistic and semantic information across widespread brain networks during speech comprehension. Here, we introduce a nonlinear, multimodal prediction model that combines audio and linguistic features from pre-trained models (e.g., LLAMA, Whisper). Our approach achieves a 17.2% and 17.9% improvement in prediction performance (unnormalized and normalized correlation) over traditional unimodal linear models, as well as a 7.7% and 14.4% improvement, respectively, over prior state-of-the-art models. These improvements represent a major step towards future robust in-silico testing and improved decoding performance. They also reveal how auditory and semantic information are fused in motor, somatosensory, and higher-level semantic regions, aligning with existing neurolinguistic theories. Overall, our work highlights the often neglected potential of nonlinear and multimodal approaches to brain modeling, paving the way for future studies to embrace these strategies in naturalistic neurolinguistics research. 

**Abstract (ZH)**: 自监督的语言和音频模型能够有效预测对语音的脑响应。然而，传统预测模型依赖于从单模特征到线性映射的关系，尽管在言语理解过程中，听觉信号与语言和语义信息的综合跨越了大量的脑网络，是一种复杂的集成过程。在此，我们引入了一个非线性的多模态预测模型，该模型结合了预训练模型（如LLAMA、Whisper）的音频和语言特征。我们的方法在未标准化和标准化相关性上分别比传统的单模态线性模型提高了17.2%和17.9%，并比之前最先进的模型分别提高了7.7%和14.4%。这些改进标志着朝着未来稳健的计算机模拟测试和解码性能提升迈出的重要一步。它们还揭示了听觉和语义信息如何在运动、躯体感觉以及更高层次的语义区域融合，与现有的神经语言学理论相一致。总的来说，我们的研究突显了非线性和多模态方法在脑建模中的潜在价值，为未来研究在自然语言神经科学中的采用这些策略铺平了道路。 

---
# How Much Do LLMs Hallucinate across Languages? On Multilingual Estimation of LLM Hallucination in the Wild 

**Title (ZH)**: 多语言环境中语言模型的幻觉程度有多高？关于自然场景下多语言语言模型幻觉的估算研究 

**Authors**: Saad Obaid ul Islam, Anne Lauscher, Goran Glavaš  

**Link**: [PDF](https://arxiv.org/pdf/2502.12769)  

**Abstract**: In the age of misinformation, hallucination -- the tendency of Large Language Models (LLMs) to generate non-factual or unfaithful responses -- represents the main risk for their global utility. Despite LLMs becoming increasingly multilingual, the vast majority of research on detecting and quantifying LLM hallucination are (a) English-centric and (b) focus on machine translation (MT) and summarization, tasks that are less common ``in the wild'' than open information seeking. In contrast, we aim to quantify the extent of LLM hallucination across languages in knowledge-intensive long-form question answering. To this end, we train a multilingual hallucination detection model and conduct a large-scale study across 30 languages and 6 open-source LLM families. We start from an English hallucination detection dataset and rely on MT to generate (noisy) training data in other languages. We also manually annotate gold data for five high-resource languages; we then demonstrate, for these languages, that the estimates of hallucination rates are similar between silver (LLM-generated) and gold test sets, validating the use of silver data for estimating hallucination rates for other languages. For the final rates estimation, we build a knowledge-intensive QA dataset for 30 languages with LLM-generated prompts and Wikipedia articles as references. We find that, while LLMs generate longer responses with more hallucinated tokens for higher-resource languages, there is no correlation between length-normalized hallucination rates of languages and their digital representation. Further, we find that smaller LLMs exhibit larger hallucination rates than larger models. 

**Abstract (ZH)**: 在信息泛滥的时代，幻觉——大规模语言模型（LLMs）生成非事实或不忠实响应的趋势——代表了它们全球实用性的主要风险。尽管LLMs变得日益多语言化，但关于检测和量化LLM幻觉的研究主要（a）以英文为中心，并且（b）集中在机器翻译（MT）和摘要等任务上，而这些任务在实际应用中不如开放信息检索更为常见。相比之下，我们旨在跨多种语言量化知识密集型长文本问答中的LLM幻觉。为此，我们训练了一个多语言幻觉检测模型，并在30种语言和6个开源LLM家族中进行了大规模研究。我们从英文幻觉检测数据集开始，并利用机器翻译生成其他语言的（嘈杂的）训练数据。我们也为五种高资源语言人工标注了黄金数据；然后，我们表明，对于这些语言而言，幻觉率在银数据（LLM生成的）测试集和黄金测试集之间相似，验证了使用银数据来估计其他语言的幻觉率的有效性。在最终的幻觉率估计中，我们为30种语言构建了一个知识密集型问答数据集，其中LLM生成的提示和维基百科文章作为参考。我们发现，虽然高资源语言的LLM生成更长的回答且包含更多的幻觉词汇，但长度校正后的幻觉率与语言的数字化表示之间没有相关性。进一步地，我们发现较小的LLM的幻觉率高于较大的模型。 

---
# R2-KG: General-Purpose Dual-Agent Framework for Reliable Reasoning on Knowledge Graphs 

**Title (ZH)**: R2-KG：知识图上可靠推理的通用双代理框架 

**Authors**: Sumin Jo, Junseong Choi, Jiho Kim, Edward Choi  

**Link**: [PDF](https://arxiv.org/pdf/2502.12767)  

**Abstract**: Recent studies have combined Large Language Models (LLMs) with Knowledge Graphs (KGs) to enhance reasoning, improving inference accuracy without additional training while mitigating hallucination. However, existing frameworks are often rigid, struggling to adapt to KG or task changes. They also rely heavily on powerful LLMs for reliable (i.e., trustworthy) reasoning. To address this, We introduce R2-KG, a plug-and-play, dual-agent framework that separates reasoning into two roles: an Operator (a low-capacity LLM) that gathers evidence and a Supervisor (a high-capacity LLM) that makes final judgments. This design is cost-efficient for LLM inference while still maintaining strong reasoning accuracy. Additionally, R2-KG employs an Abstention mechanism, generating answers only when sufficient evidence is collected from KG, which significantly enhances reliability. Experiments across multiple KG-based reasoning tasks show that R2-KG consistently outperforms baselines in both accuracy and reliability, regardless of the inherent capability of LLMs used as the Operator. Further experiments reveal that the single-agent version of R2-KG, equipped with a strict self-consistency strategy, achieves significantly higher-than-baseline reliability while reducing inference cost. However, it also leads to a higher abstention rate in complex KGs. Our findings establish R2-KG as a flexible and cost-effective solution for KG-based reasoning. It reduces reliance on high-capacity LLMs while ensuring trustworthy inference. 

**Abstract (ZH)**: 最近的研究将大型语言模型（LLMs）与知识图谱（KGs）结合起来，以提高推理能力，在不需要额外训练的情况下提高推理准确性，同时减轻虚构现象。然而，现有的框架通常较为僵硬，难以适应KG或任务的变化。它们还严重依赖强大且可靠的LLMs来进行推理。为了解决这一问题，我们提出了R2-KG，这是一种插拔式、双代理框架，将推理分为两个角色：操作员（具有较低能力的LLM），负责收集证据；监护者（具有较高能力的LLM），进行最终判断。此设计在保持强大推理准确性的同时，使LLM推理成本更为经济。此外，R2-KG 还采用了一个弃权机制，仅在从知识图谱收集到足够证据时生成答案，这显着增强了其可靠性。在多个基于知识图谱的推理任务上的实验表明，R2-KG 在准确性和可靠性方面始终优于基准模型，即使使用的操作员LLM的固有能力有所不同。进一步的实验表明，配备了严格自我一致性策略的单代理版本R2-KG，在保持比基准模型更高的可靠性的同时，降低了推理成本。但是，这也导致在复杂知识图谱中弃权率的增加。我们的研究结果表明，R2-KG 是一个灵活且成本效益高的知识图谱推理解决方案，它减少了对高容量LLM的依赖，同时确保了可靠的推理。 

---
# Efficient Machine Translation Corpus Generation: Integrating Human-in-the-Loop Post-Editing with Large Language Models 

**Title (ZH)**: 高效机器翻译语料库生成：结合人类在环后编辑的大语言模型 

**Authors**: Kamer Ali Yuksel, Ahmet Gunduz, Abdul Baseet Anees, Hassan Sawaf  

**Link**: [PDF](https://arxiv.org/pdf/2502.12755)  

**Abstract**: This paper introduces an advanced methodology for machine translation (MT) corpus generation, integrating semi-automated, human-in-the-loop post-editing with large language models (LLMs) to enhance efficiency and translation quality. Building upon previous work that utilized real-time training of a custom MT quality estimation metric, this system incorporates novel LLM features such as Enhanced Translation Synthesis and Assisted Annotation Analysis, which improve initial translation hypotheses and quality assessments, respectively. Additionally, the system employs LLM-Driven Pseudo Labeling and a Translation Recommendation System to reduce human annotator workload in specific contexts. These improvements not only retain the original benefits of cost reduction and enhanced post-edit quality but also open new avenues for leveraging cutting-edge LLM advancements. The project's source code is available for community use, promoting collaborative developments in the field. The demo video can be accessed here. 

**Abstract (ZH)**: 本文介绍了一种先进的机器翻译（MT）语料库生成方法，将半自动的人工辅助后编辑与大规模语言模型（LLMs）结合，以提高效率和翻译质量。在以往利用实时训练自定义MT质量估计指标的研究基础上，该系统整合了增强翻译合成和辅助注释分析等新型LLM功能，分别改善了初始翻译假设和质量评估。此外，系统采用了LLM驱动的伪标签生成和翻译推荐系统，在特定情境下减轻了人工注释员的工作负担。这些改进不仅保留了原有益处，即降低成本并提高后编辑质量，还为利用前沿的大规模语言模型进步开辟了新的途径。该项目的源代码已对社区开放，促进该领域的协作开发。详细的演示视频可在此处访问。 

---
# MediaMind: Revolutionizing Media Monitoring using Agentification 

**Title (ZH)**: MediaMind：通过代理化革命性地革新媒体监控 

**Authors**: Ahmet Gunduz, Kamer Ali Yuksel, Hassan Sawaf  

**Link**: [PDF](https://arxiv.org/pdf/2502.12745)  

**Abstract**: In an era of rapid technological advancements, agentification of software tools has emerged as a critical innovation, enabling systems to function autonomously and adaptively. This paper introduces MediaMind as a case study to demonstrate the agentification process, highlighting how existing software can be transformed into intelligent agents capable of independent decision-making and dynamic interaction. Developed by aiXplain, MediaMind leverages agent-based architecture to autonomously monitor, analyze, and provide insights from multilingual media content in real time. The focus of this paper is on the technical methodologies and design principles behind agentifying MediaMind, showcasing how agentification enhances adaptability, efficiency, and responsiveness. Through detailed case studies and practical examples, we illustrate how the agentification of MediaMind empowers organizations to streamline workflows, optimize decision-making, and respond to evolving trends. This work underscores the broader potential of agentification to revolutionize software tools across various domains. 

**Abstract (ZH)**: 在技术飞速发展的时代，软件工具的代理化已成为一项关键创新，使系统能够自主运行并适应环境。本文以MediaMind为例，展示了代理化的过程，强调了如何通过现有的软件创建出能够独立做出决策并进行动态交互的智能代理。MediaMind是由aiXplain开发的一种基于代理架构的工具，能够实时自主监控、分析并提供多种语言媒体内容的见解。本文的重点在于介绍代理化MediaMind的技术方法和设计理念，展示代理化如何增强系统的适应性、效率和响应性。通过详细的案例研究和实际示例，本文展示了如何通过代理化MediaMind使组织简化工作流程、优化决策并应对不断变化的趋势。本文还凸显了代理化在各个领域彻底改变软件工具的广泛潜力。 

---
# Self-Enhanced Reasoning Training: Activating Latent Reasoning in Small Models for Enhanced Reasoning Distillation 

**Title (ZH)**: 自我增强推理训练：激活小型模型中的潜在推理能力以提升推理精炼 

**Authors**: Yong Zhang, Bingyuan Zhang, Zhitao Li, Ming Li, Ning Cheng, Minchuan Chen, Tao Wei, Jun Ma, Shaojun Wang, Jing Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2502.12744)  

**Abstract**: The rapid advancement of large language models (LLMs) has significantly enhanced their reasoning abilities, enabling increasingly complex tasks. However, these capabilities often diminish in smaller, more computationally efficient models like GPT-2. Recent research shows that reasoning distillation can help small models acquire reasoning capabilities, but most existing methods focus primarily on improving teacher-generated reasoning paths. Our observations reveal that small models can generate high-quality reasoning paths during sampling, even without chain-of-thought prompting, though these paths are often latent due to their low probability under standard decoding strategies. To address this, we propose Self-Enhanced Reasoning Training (SERT), which activates and leverages latent reasoning capabilities in small models through self-training on filtered, self-generated reasoning paths under zero-shot conditions. Experiments using OpenAI's GPT-3.5 as the teacher model and GPT-2 models as the student models demonstrate that SERT enhances the reasoning abilities of small models, improving their performance in reasoning distillation. 

**Abstract (ZH)**: 大语言模型（LLMs）的迅速进步显著提升了其推理能力，使其能够完成更加复杂的任务。然而，这些能力在更小、计算效率更高的模型如GPT-2中往往会减弱。最近的研究表明，推理蒸馏可以帮助小型模型获得推理能力，但大多数现有方法主要集中在改进教师生成的推理路径上。我们的观察发现，在不使用思考链提示的情况下，小型模型在采样过程中仍能生成高质量的推理路径，尽管这些路径由于标准解码策略下的低概率而往往处于潜在状态。为解决这一问题，我们提出了自我增强推理训练（SERT），通过零样本条件下对过滤和自我生成的推理路径进行自我训练，激活并利用小型模型中潜在的推理能力。使用OpenAI的GPT-3.5作为教师模型，GPT-2模型作为学生模型的实验表明，SERT能够增强小型模型的推理能力，在推理蒸馏中提高其性能。 

---
# "I know myself better, but not really greatly": Using LLMs to Detect and Explain LLM-Generated Texts 

**Title (ZH)**: “我了解自己更多，但仅仅是一点点”：使用大型语言模型检测和解释由大型语言模型生成的文字

这个标题翻译成中文后，既保持了原意，又符合学术规范。原文中引号里的内容被翻译成中文中的引号内容，以便更好地传达原文的意思。 

**Authors**: Jiazhou Ji, Jie Guo, Weidong Qiu, Zheng Huang, Yang Xu, Xinru Lu, Xiaoyu Jiang, Ruizhe Li, Shujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.12743)  

**Abstract**: Large language models (LLMs) have demonstrated impressive capabilities in generating human-like texts, but the potential misuse of such LLM-generated texts raises the need to distinguish between human-generated and LLM-generated content. This paper explores the detection and explanation capabilities of LLM-based detectors of LLM-generated texts, in the context of a binary classification task (human-generated texts vs LLM-generated texts) and a ternary classification task (human-generated texts, LLM-generated texts, and undecided). By evaluating on six close/open-source LLMs with different sizes, our findings reveal that while self-detection consistently outperforms cross-detection, i.e., LLMs can detect texts generated by themselves more accurately than those generated by other LLMs, the performance of self-detection is still far from ideal, indicating that further improvements are needed. We also show that extending the binary to the ternary classification task with a new class "Undecided" can enhance both detection accuracy and explanation quality, with improvements being statistically significant and consistent across all LLMs. We finally conducted comprehensive qualitative and quantitative analyses on the explanation errors, which are categorized into three types: reliance on inaccurate features (the most frequent error), hallucinations, and incorrect reasoning. These findings with our human-annotated dataset emphasize the need for further research into improving both self-detection and self-explanation, particularly to address overfitting issues that may hinder generalization. 

**Abstract (ZH)**: 大型语言模型（LLMs）在生成类人类文本方面展示了令人印象深刻的性能，但这些LLM生成文本的潜在滥用性引发了区分人类生成和LLM生成内容的需要。本文探讨了基于LLM的LLM生成文本检测和解释能力，特别是在二分类任务（人类生成文本 vs LLM生成文本）和三分类任务（人类生成文本、LLM生成文本和不确定）的背景下。通过在六个不同规模的闭源/开源LLM上进行评估，我们的研究发现虽然自我检测的性能总体上优于跨检测，即LLM可以更准确地检测到自己生成的文本而非其他LLM生成的文本，但自我检测的性能仍然远未达到理想状态，表明仍需进一步改进。我们还展示了将二分类任务扩展为三分类任务，增加一个新的类别“不确定”可以同时提高检测准确性和解释质量，这种改进在所有LLM上具有统计意义上显著且一致的表现。最后，我们对解释错误进行了全面的定性和定量分析，这些错误被分类为三类：依赖不准确特征（最常见的错误）、幻觉和错误推理。基于我们的人工标注数据集的研究结果强调了进一步研究以改进自我检测和自我解释的必要性，特别是为了应对可能阻碍泛化的过拟合问题。 

---
# Beyond Seen Data: Improving KBQA Generalization Through Schema-Guided Logical Form Generation 

**Title (ZH)**: 超越已见数据：通过基于模式的逻辑形式生成提高知识图谱问答的一般性 

**Authors**: Shengxiang Gao, Jey Han Lau, Jianzhong Qi  

**Link**: [PDF](https://arxiv.org/pdf/2502.12737)  

**Abstract**: Knowledge base question answering (KBQA) aims to answer user questions in natural language using rich human knowledge stored in large KBs. As current KBQA methods struggle with unseen knowledge base elements at test time,we introduce SG-KBQA: a novel model that injects schema contexts into entity retrieval and logical form generation to tackle this issue. It uses the richer semantics and awareness of the knowledge base structure provided by schema contexts to enhance generalizability. We show that SG-KBQA achieves strong generalizability, outperforming state-of-the-art models on two commonly used benchmark datasets across a variety of test settings. Code will be released upon paper publication. 

**Abstract (ZH)**: 知识图谱问答（KBQA）旨在利用大型知识库中存储的丰富人类知识来回答用户在自然语言中的问题。由于当前的KBQA方法在测试时难以处理未见过的知识库元素，我们提出了SG-KBQA：一种新型模型，通过注入模式上下文来改善实体检索和逻辑形式生成，从而解决这一问题。SG-KBQA 利用模式上下文提供的更丰富语义和知识库结构的意识，以增强其泛化能力。我们展示了SG-KBQA在多种测试场景下，在两个常用的基准数据集上显著超出当前最先进的模型，具备强大的泛化能力。论文发表后将会公开相关代码。 

---
# Playing with Voices: Tabletop Role-Playing Game Recordings as a Diarization Challenge 

**Title (ZH)**: 《玩转声音：桌面角色扮演游戏录音的日记记录挑战》

这个翻译符合学术规范，同时保留了原文的核心意思。若需进一步调整以适应具体学术风格或期刊要求，请告诉我。 

**Authors**: Lian Remme, Kevin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12714)  

**Abstract**: This paper provides a proof of concept that audio of tabletop role-playing games (TTRPG) could serve as a challenge for diarization systems. TTRPGs are carried out mostly by conversation. Participants often alter their voices to indicate that they are talking as a fictional character. Audio processing systems are susceptible to voice conversion with or without technological assistance. TTRPG present a conversational phenomenon in which voice conversion is an inherent characteristic for an immersive gaming experience. This could make it more challenging for diarizers to pick the real speaker and determine that impersonating is just that. We present the creation of a small TTRPG audio dataset and compare it against the AMI and the ICSI corpus. The performance of two diarizers, this http URL and wespeaker, were evaluated. We observed that TTRPGs' properties result in a higher confusion rate for both diarizers. Additionally, wespeaker strongly underestimates the number of speakers in the TTRPG audio files. We propose TTRPG audio as a promising challenge for diarization systems. 

**Abstract (ZH)**: 本文提供了一个概念证明，表明桌面角色扮演游戏（TTRPG）的音频可以成为语音 diarization 系统的一大挑战。TTRPG 主要是通过对话进行的。参与者经常通过改变声音来表明他们是在扮演虚构的角色。语音处理系统在有或没有技术支持的情况下都容易受到语音转换的影响。TTRPG 提供了一个对话现象，在这种现象中，语音转换是实现沉浸式游戏体验的固有特征。这使得 diarization 系统更难以识别真实说话人并准确判断这种方式是否为角色扮演。本文展示了创建一个小规模的 TTRPG 音频数据集，并将其与 AMI 和 ICSI 系统进行比较。评估了两个 diarization 系统“这个网址”和 wespeaker 的性能。我们发现 TTRPG 的特性导致两种 diarization 系统的混淆率更高。此外，wespeaker 显著低估了 TTRPG 音频文件中的说话人数目。我们提出 TTRPG 音频为 diarization 系统设定了一个有前景的挑战。 

---
# Translate Smart, not Hard: Cascaded Translation Systems with Quality-Aware Deferral 

**Title (ZH)**: 不拼努力拼智能：具备质量意识的级联翻译系统 

**Authors**: António Farinhas, Nuno M. Guerreiro, Sweta Agrawal, Ricardo Rei, André F.T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2502.12701)  

**Abstract**: Larger models often outperform smaller ones but come with high computational costs. Cascading offers a potential solution. By default, it uses smaller models and defers only some instances to larger, more powerful models. However, designing effective deferral rules remains a challenge. In this paper, we propose a simple yet effective approach for machine translation, using existing quality estimation (QE) metrics as deferral rules. We show that QE-based deferral allows a cascaded system to match the performance of a larger model while invoking it for a small fraction (30% to 50%) of the examples, significantly reducing computational costs. We validate this approach through both automatic and human evaluation. 

**Abstract (ZH)**: 较大的模型通常表现出更好的性能，但伴随而来的是高昂的计算成本。级联提供了一种潜在的解决方案。默认情况下，它使用较小的模型，并仅将部分实例递交给较大、更强的模型。然而，设计有效的递归规则仍然是一个挑战。在本文中，我们提出了一种简单而有效的机器翻译方法，利用现有的质量估计（QE）指标作为递归规则。我们展示了基于QE的递归使得级联系统能够在一小部分示例（约30%至50%）上调用更大的模型，从而大幅降低计算成本，同时匹配较大模型的性能。我们通过自动评估和人工评估验证了这一方法的有效性。 

---
# Multi-Novelty: Improve the Diversity and Novelty of Contents Generated by Large Language Models via inference-time Multi-Views Brainstorming 

**Title (ZH)**: 多新颖性：通过推理时多视角创意Brainstorming提升大型语言模型生成内容的多样性和新颖性 

**Authors**: Arash Lagzian, Srinivas Anumasa, Dianbo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12700)  

**Abstract**: Large Language Models (LLMs) demonstrate remarkable proficiency in generating accurate and fluent text. However, they often struggle with diversity and novelty, leading to repetitive or overly deterministic responses. These limitations stem from constraints in training data, including gaps in specific knowledge domains, outdated information, and an over-reliance on textual sources. Such shortcomings reduce their effectiveness in tasks requiring creativity, multi-perspective reasoning, and exploratory thinking, such as LLM based AI scientist agents and creative artist agents . To address this challenge, we introduce inference-time multi-view brainstorming method, a novel approach that enriches input prompts with diverse perspectives derived from both textual and visual sources, which we refere to as "Multi-Novelty". By incorporating additional contextual information as diverse starting point for chain of thoughts, this method enhances the variety and creativity of generated outputs. Importantly, our approach is model-agnostic, requiring no architectural modifications and being compatible with both open-source and proprietary LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）在生成准确流畅文本方面表现出色。然而，它们往往在多样性与新颖性方面存在不足，导致回答内容重复或过于凡尔赛。这些局限性源自训练数据的限制，包括特定知识领域的数据缺口、过时信息以及对文本源的过度依赖。这些不足降低了其在需要创造力、多视角推理和探索性思考的任务中的有效性，如基于LLM的AI科学家代理和创造性艺术家代理。为解决这一挑战，我们提出了一种推理时多视角头脑风暴方法，这是一种新颖的方法，通过将来自文本和视觉来源的多样化视角丰富输入提示，使我们称之为“多新颖性”（Multi-Novelty）。通过将额外的上下文信息作为思考链的多样化起点，这种方法提高了生成输出的多样性和创造性。重要的是，我们的方法是模型无关的，不需要对模型架构进行任何修改，并且兼容开源和专有LLM。 

---
# Theoretical Guarantees for Minimum Bayes Risk Decoding 

**Title (ZH)**: 最小贝叶斯风险解码的理论保证 

**Authors**: Yuki Ichihara, Yuu Jinnai, Kaito Ariu, Tetsuro Morimura, Eiji Uchibe  

**Link**: [PDF](https://arxiv.org/pdf/2502.12685)  

**Abstract**: Minimum Bayes Risk (MBR) decoding optimizes output selection by maximizing the expected utility value of an underlying human distribution. While prior work has shown the effectiveness of MBR decoding through empirical evaluation, few studies have analytically investigated why the method is effective. As a result of our analysis, we show that, given the size $n$ of the reference hypothesis set used in computation, MBR decoding approaches the optimal solution with high probability at a rate of $O\left(n^{-\frac{1}{2}}\right)$, under certain assumptions, even though the language space $Y$ is significantly larger $Y\gg n$. This result helps to theoretically explain the strong performance observed in several prior empirical studies on MBR decoding. In addition, we provide the performance gap for maximum-a-posteriori (MAP) decoding and compare it to MBR decoding. The result of this paper indicates that MBR decoding tends to converge to the optimal solution faster than MAP decoding in several cases. 

**Abstract (ZH)**: 最小贝叶斯风险（Minimum Bayes Risk, MBR）解码通过最大化潜在人类分布的预期效用值来优化输出选择。尽管先前的研究通过实证评估展示了MBR解码的有效性，但很少有研究从理论上分析其有效性的原因。通过我们的分析，我们证明，在满足某些假设的前提下，给定用于计算的参考假设集大小为 $n$，即使语言空间 $Y$ 显著大于 $Y \gg n$，MBR解码也以 $O\left(n^{-\frac{1}{2}}\right)$ 的概率逼近最优解。这有助于从理论上解释在若干先前的实证研究中观察到的MBR解码的强大性能。此外，我们还提供了最大后验概率（Maximum a Posteriori, MAP）解码的性能差距，并将其与MBR解码进行了比较。本文的结果表明，在某些情况下，MBR解码相较于MAP解码更倾向于更快地收敛到最优解。 

---
# Speech-FT: A Fine-tuning Strategy for Enhancing Speech Representation Models Without Compromising Generalization Ability 

**Title (ZH)**: Speech-FT：一种在不牺牲泛化能力的前提下提升语音表示模型的微调策略 

**Authors**: Tzu-Quan Lin, Wei-Ping Huang, Hao Tang, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.12672)  

**Abstract**: Speech representation models are highly effective at extracting general features for various tasks. While fine-tuning can enhance these representations for specific applications, it often compromises their generalization ability. To address this challenge, we propose Speech-FT, a fine-tuning strategy for speech representation models that leverages model merging to preserve generalization ability while still benefiting from fine-tuning. Speech-FT is effective across different fine-tuning scenarios and is compatible with various types of speech representation models, providing a versatile solution. Speech-FT offers an efficient and practical approach to further improving general speech representations after pre-training. 

**Abstract (ZH)**: 语音表示模型在各种任务中高度有效地提取通用特征。虽然微调可以增强这些表示以适应特定应用，但它往往牺牲了泛化能力。为了解决这一挑战，我们提出了一种名为Speech-FT的微调策略，该策略利用模型合并来保留泛化能力，同时仍能从微调中受益。Speech-FT在不同的微调场景中有效，并且与各种类型的语音表示模型兼容，提供了一个灵活的解决方案。Speech-FT为预训练后的进一步改进通用语音表示提供了一种高效和实用的方法。 

---
# Baichuan-M1: Pushing the Medical Capability of Large Language Models 

**Title (ZH)**: Baichuan-M1：推动大型语言模型在医学领域的能力 

**Authors**: Bingning Wang, Haizhou Zhao, Huozhi Zhou, Liang Song, Mingyu Xu, Wei Cheng, Xiangrong Zeng, Yupeng Zhang, Yuqi Huo, Zecheng Wang, Zhengyun Zhao, Da Pan, Fan Yang, Fei Kou, Fei Li, Fuzhong Chen, Guosheng Dong, Han Liu, Hongda Zhang, Jin He, Jinjie Yang, Kangxi Wu, Kegeng Wu, Lei Su, Linlin Niu, Linzhuang Sun, Mang Wang, Pengcheng Fan, Qianli Shen, Rihui Xin, Shunya Dang, Songchi Zhou, Weipeng Chen, Wenjing Luo, Xin Chen, Xin Men, Xionghai Lin, Xuezhen Dong, Yan Zhang, Yifei Duan, Yuyan Zhou, Zhi Ma, Zhiying Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12671)  

**Abstract**: The current generation of large language models (LLMs) is typically designed for broad, general-purpose applications, while domain-specific LLMs, especially in vertical fields like medicine, remain relatively scarce. In particular, the development of highly efficient and practical LLMs for the medical domain is challenging due to the complexity of medical knowledge and the limited availability of high-quality data. To bridge this gap, we introduce Baichuan-M1, a series of large language models specifically optimized for medical applications. Unlike traditional approaches that simply continue pretraining on existing models or apply post-training to a general base model, Baichuan-M1 is trained from scratch with a dedicated focus on enhancing medical capabilities. Our model is trained on 20 trillion tokens and incorporates a range of effective training methods that strike a balance between general capabilities and medical expertise. As a result, Baichuan-M1 not only performs strongly across general domains such as mathematics and coding but also excels in specialized medical fields. We have open-sourced Baichuan-M1-14B, a mini version of our model, which can be accessed through the following links. 

**Abstract (ZH)**: 当前一代大型语言模型（LLMs）通常设计用于广泛的通用应用，而专门针对特定领域的LLMs，尤其是在医学等垂直领域，仍然相对稀缺。特别是，开发高效且实用的医学专用LLMs极具挑战性，这主要是由于医学知识的复杂性和高质量数据的有限可用性。为解决这一问题，我们介绍了Baichuan-M1系列大型语言模型，专门针对医疗应用进行优化。与传统方法不同，传统方法仅仅是基于现有模型继续预训练或在通用基模型上进行后训练，Baichuan-M1则是从零开始训练，并专注于增强医学能力。我们的模型基于20万亿个标记进行训练，并结合了一系列有效的训练方法，能够在保持一般能力的同时增强医学专长。因此，Baichuan-M1不仅在数学和编程等通用领域表现出色，还在专门的医学领域中表现出色。我们已开源了Baichuan-M1-14B，这是一个模型的迷你版本，可以通过以下链接访问。 

---
# Evaluation of Best-of-N Sampling Strategies for Language Model Alignment 

**Title (ZH)**: 最佳-of-N（Best-of-N）采样策略对语言模型对齐效果的评估 

**Authors**: Yuki Ichihara, Yuu Jinnai, Tetsuro Morimura, Kaito Ariu, Kenshi Abe, Mitsuki Sakamoto, Eiji Uchibe  

**Link**: [PDF](https://arxiv.org/pdf/2502.12668)  

**Abstract**: Best-of-N (BoN) sampling with a reward model has been shown to be an effective strategy for aligning Large Language Models (LLMs) with human preferences at the time of decoding. BoN sampling is susceptible to a problem known as reward hacking. Since the reward model is an imperfect proxy for the true objective, an excessive focus on optimizing its value can lead to a compromise of its performance on the true objective. Previous work proposes Regularized BoN sampling (RBoN), a BoN sampling with regularization to the objective, and shows that it outperforms BoN sampling so that it mitigates reward hacking and empirically (Jinnai et al., 2024). However, Jinnai et al. (2024) introduce RBoN based on a heuristic and they lack the analysis of why such regularization strategy improves the performance of BoN sampling. The aim of this study is to analyze the effect of BoN sampling on regularization strategies. Using the regularization strategies corresponds to robust optimization, which maximizes the worst case over a set of possible perturbations in the proxy reward. Although the theoretical guarantees are not directly applicable to RBoN, RBoN corresponds to a practical implementation. This paper proposes an extension of the RBoN framework, called Stochastic RBoN sampling (SRBoN), which is a theoretically guaranteed approach to worst-case RBoN in proxy reward. We then perform an empirical evaluation using the AlpacaFarm and Anthropic's hh-rlhf datasets to evaluate which factors of the regularization strategies contribute to the improvement of the true proxy reward. In addition, we also propose another simple RBoN method, the Sentence Length Regularized BoN, which has a better performance in the experiment as compared to the previous methods. 

**Abstract (ZH)**: 最佳N（BoN）采样结合奖励模型已被证明是一种有效策略，用于在解码时将大型语言模型（LLMs）与人类偏好对齐。BoN采样易受名为“奖励作弊”的问题影响。由于奖励模型只是真实目标的不完美代理，过度关注优化其价值会导致在真实目标上的表现下降。先前的研究提出了一种正则化BoN采样（RBoN），即结合目标正则化的BoN采样方法，并证明它优于普通BoN采样，能够减轻奖励作弊并且实验证明有效（Jinnai等，2024）。然而，Jinnai等（2024）基于启发式提出了RBoN，缺乏关于为何这种正则化策略能改进BoN采样性能的分析。本研究旨在分析BoN采样对正则化策略效果的影响。采用正则化策略相当于鲁棒优化，其目标是在代理奖励可能的扰动集合中最大化最坏情况。尽管RBoN的理论保证不直接适用，但它是一种实用的实现。本文提出了RBoN框架的扩展，称为随机RBoN采样（SRBoN），这是一种理论上保证的代理奖励最坏情况下RBoN的方法。我们随后使用AlpacaFarm和Anthropic的hh-rlhf数据集进行实证评估，以确定正则化策略中的哪些因素有助于提高代理奖励的真实度。此外，我们还提出了一种简化的RBoN方法，称为句子长度正则化BoN，在实验中其性能优于以前的方法。 

---
# A$^2$ATS: Retrieval-Based KV Cache Reduction via Windowed Rotary Position Embedding and Query-Aware Vector Quantization 

**Title (ZH)**: A$^2$ATS：基于窗口旋转位置嵌入和查询感知向量量化的内容检索缓存减少方法 

**Authors**: Junhui He, Junna Xing, Nan Wang, Rui Xu, Shangyu Wu, Peng Zhou, Qiang Liu, Chun Jason Xue, Qingan Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.12665)  

**Abstract**: Long context large language models (LLMs) pose significant challenges for efficient serving due to the large memory footprint and high access overhead of KV cache. Retrieval-based KV cache reduction methods can mitigate these challenges, typically by offloading the complete KV cache to CPU and retrieving necessary tokens on demand during inference. However, these methods still suffer from unsatisfactory accuracy degradation and extra retrieval overhead. To address these limitations, this paper proposes A$^2$ATS, a novel retrieval-based KV cache reduction method. A$^2$ATS aims to obtain an accurate approximation of attention scores by applying the vector quantization technique to key states, thereby enabling efficient and precise retrieval of the top-K tokens. First, we propose Windowed Rotary Position Embedding, which decouples the positional dependency from query and key states after position embedding. Then, we propose query-aware vector quantization that optimizes the objective of attention score approximation directly. Finally, we design the heterogeneous inference architecture for KV cache offloading, enabling long context serving with larger batch sizes. Experimental results demonstrate that A$^2$ATS can achieve a lower performance degradation with similar or lower overhead compared to existing methods, thereby increasing long context serving throughput by up to $2.7 \times$. 

**Abstract (ZH)**: 长上下文大型语言模型（LLMs）由于键值缓存（KV cache）的巨大内存占用和高访问开销，在高效服务方面面临重大挑战。基于检索的键值缓存减少方法可以通过将完整的键值缓存卸载到CPU并在推理时按需检索必要tokens来缓解这些挑战。然而，这些方法仍然存在不满意的准确度下降和额外的检索开销。为解决这些问题，本文提出了一种名为A$^2$ATS的新颖基于检索的键值缓存减少方法。A$^2$ATS旨在通过将向量量化技术应用于键状态，获得注意力分数的准确近似，从而实现高效和精准的top-K tokens检索。

具体来说，首先，本文提出了一种分段旋转位置嵌入（Windowed Rotary Position Embedding），在位置嵌入后将位置依赖性与查询和键状态分离。然后，本文提出了一种查询感知的向量量化，直接优化了注意力分数近似的目标。最后，设计了一种异构推理架构用于键值缓存卸载，从而实现更大批量的长上下文服务。实验结果表明，相比于现有方法，A$^2$ATS可以以相似或更低的开销实现更低的性能下降，从而将长上下文服务吞吐量提高到最高2.7倍。 

---
# Demystifying Multilingual Chain-of-Thought in Process Reward Modeling 

**Title (ZH)**: 揭开多语言链式思维在过程奖励建模中的神秘面纱 

**Authors**: Weixuan Wang, Minghao Wu, Barry Haddow, Alexandra Birch  

**Link**: [PDF](https://arxiv.org/pdf/2502.12663)  

**Abstract**: Large language models (LLMs) are designed to perform a wide range of tasks. To improve their ability to solve complex problems requiring multi-step reasoning, recent research leverages process reward modeling to provide fine-grained feedback at each step of the reasoning process for reinforcement learning (RL), but it predominantly focuses on English. In this paper, we tackle the critical challenge of extending process reward models (PRMs) to multilingual settings. To achieve this, we train multilingual PRMs on a dataset spanning seven languages, which is translated from English. Through comprehensive evaluations on two widely used reasoning benchmarks across 11 languages, we demonstrate that multilingual PRMs not only improve average accuracy but also reduce early-stage reasoning errors. Furthermore, our results highlight the sensitivity of multilingual PRMs to both the number of training languages and the volume of English data, while also uncovering the benefits arising from more candidate responses and trainable parameters. This work opens promising avenues for robust multilingual applications in complex, multi-step reasoning tasks. In addition, we release the code to foster research along this line. 

**Abstract (ZH)**: 大规模语言模型（LLMs）旨在执行广泛的任务。为了提高它们解决需要多步推理的复杂问题的能力，最近的研究利用过程奖励建模来在增强学习（RL）的每一步提供精细的反馈，但主要集中在英语上。在本文中，我们面临着将过程奖励模型（PRMs）扩展到多语言环境中的关键挑战。为此，我们在涵盖七种语言的数据集上训练了多语言PRMs，该数据集是从英语翻译过来的。通过在两个广泛使用的推理基准上进行全面评估，其中包括11种语言，我们证明了多语言PRMs不仅提高了平均准确性，还减少了早期阶段的推理错误。此外，我们的结果强调了多语言PRMs对训练语言数量和英语数据量的敏感性，并揭示了候选响应数量和可训练参数更多所带来的好处。这项研究为在复杂的多步推理任务中实现稳健的多语言应用开辟了有希望的新途径。此外，我们发布了代码以促进该领域的研究。 

---
# R.R.: Unveiling LLM Training Privacy through Recollection and Ranking 

**Title (ZH)**: R.R.: 探索大模型训练隐私的回忆与排序方法 

**Authors**: Wenlong Meng, Zhenyuan Guo, Lenan Wu, Chen Gong, Wenyan Liu, Weixian Li, Chengkun Wei, Wenzhi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.12658)  

**Abstract**: Large Language Models (LLMs) pose significant privacy risks, potentially leaking training data due to implicit memorization. Existing privacy attacks primarily focus on membership inference attacks (MIAs) or data extraction attacks, but reconstructing specific personally identifiable information (PII) in LLM's training data remains challenging. In this paper, we propose R.R. (Recollect and Rank), a novel two-step privacy stealing attack that enables attackers to reconstruct PII entities from scrubbed training data where the PII entities have been masked. In the first stage, we introduce a prompt paradigm named recollection, which instructs the LLM to repeat a masked text but fill in masks. Then we can use PII identifiers to extract recollected PII candidates. In the second stage, we design a new criterion to score each PII candidate and rank them. Motivated by membership inference, we leverage the reference model as a calibration to our criterion. Experiments across three popular PII datasets demonstrate that the R.R. achieves better PII identical performance compared to baselines. These results highlight the vulnerability of LLMs to PII leakage even when training data has been scrubbed. We release the replicate package of R.R. at a link. 

**Abstract (ZH)**: 大型语言模型（LLMs）提出了重大的隐私风险，可能由于隐式记忆泄露训练数据。现有的隐私攻击主要集中在成员身份推断攻击（MIA）或数据提取攻击上，但从已脱敏的训练数据中重构特定的个人可识别信息（PII）仍然是一个挑战。在本文中，我们提出了一种名为R.R.（Recollect and Rank）的新颖两步隐私窃取攻击，使攻击者能够在已屏蔽的训练数据中重构PII实体。在第一步中，我们引入了一种名为回忆的提示框架，指示LLM重复带有屏蔽的内容但填充屏蔽部分。然后，我们可以使用PII标识符来提取候选的PII。在第二步中，我们设计了一种新的评分标准来对每个PII候选进行评分并进行排序。受成员身份推断的启发，我们利用参照模型作为我们标准的校准工具。我们在三个流行的PII数据集上进行的实验表明，R.R.相比基线模型能够更好地实现PII的一致性性能。这些结果突显了即使脱敏处理过的训练数据，LLMs仍然存在PII泄露的脆弱性。我们已将R.R.的复现包发布在一个链接中。 

---
# \textit{One Size doesn't Fit All}: A Personalized Conversational Tutoring Agent for Mathematics Instruction 

**Title (ZH)**: 《一刀切不适合所有人》：一种个性化的数学辅导对话代理 

**Authors**: Ben Liu, Jihan Zhang, Fangquan Lin, Xu Jia, Min Peng  

**Link**: [PDF](https://arxiv.org/pdf/2502.12633)  

**Abstract**: Large language models (LLMs) have been increasingly employed in various intelligent educational systems, simulating human tutors to facilitate effective human-machine interaction. However, previous studies often overlook the significance of recognizing and adapting to individual learner characteristics. Such adaptation is crucial for enhancing student engagement and learning efficiency, particularly in mathematics instruction, where diverse learning styles require personalized strategies to promote comprehension and enthusiasm. In this paper, we propose a \textbf{P}erson\textbf{A}lized \textbf{C}onversational tutoring ag\textbf{E}nt (PACE) for mathematics instruction. PACE simulates students' learning styles based on the Felder and Silverman learning style model, aligning with each student's persona. In this way, our PACE can effectively assess the personality of students, allowing to develop individualized teaching strategies that resonate with their unique learning styles. To further enhance students' comprehension, PACE employs the Socratic teaching method to provide instant feedback and encourage deep thinking. By constructing personalized teaching data and training models, PACE demonstrates the ability to identify and adapt to the unique needs of each student, significantly improving the overall learning experience and outcomes. Moreover, we establish multi-aspect evaluation criteria and conduct extensive analysis to assess the performance of personalized teaching. Experimental results demonstrate the superiority of our model in personalizing the educational experience and motivating students compared to existing methods. 

**Abstract (ZH)**: 以下是经过学术规范翻译后的中文内容：

大规模语言模型（LLMs）已在各种智能教育系统中被广泛应用，模拟真人导师以促进有效的机器-人类互动。然而，以往的研究往往忽视了识别和适应个体学习者特征的重要性。这种适应性对于提升学生参与度和学习效率至关重要，尤其是在数学教学中，不同的学习风格需要个性化策略来促进理解和热情。本文提出了一种应用于数学教学的个性化对话式辅导代理（PACE）。PACE根据Felder和Silverman学习风格模型模拟学生的学习风格，并与每位学生的个性相契合。通过这种方式，我们的PACE能够有效评估学生的人格特质，进而开发与学生独特学习风格相契合的教学策略。为进一步增强学生的理解能力，PACE采用苏格拉底式教学方法提供即时反馈并鼓励深层次思考。通过构建个性化教学数据并训练模型，PACE展示了识别并适应每位学生独特需求的能力，显著改善了整体学习体验和成果。此外，我们建立了多方面评估标准，并进行了广泛分析来评估个性化教学的效果。实验结果表明，与现有方法相比，我们的模型在个性化教育体验方面表现出优越性，并能更好地激发学生的学习动力。 

---
# Improving Chain-of-Thought Reasoning via Quasi-Symbolic Abstractions 

**Title (ZH)**: 通过类符号抽象提升链式推理能力 

**Authors**: Leonardo Ranaldi, Marco Valentino, Alexander Polonsky, Andrè Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2502.12616)  

**Abstract**: Chain-of-Though (CoT) represents a common strategy for reasoning in Large Language Models (LLMs) by decomposing complex tasks into intermediate inference steps. However, explanations generated via CoT are susceptible to content biases that negatively affect their robustness and faithfulness. To mitigate existing limitations, recent work has proposed using logical formalisms coupled with external symbolic solvers. However, fully symbolic approaches possess the bottleneck of requiring a complete translation from natural language to formal languages, a process that affects efficiency and flexibility. To achieve a trade-off, this paper investigates methods to disentangle content from logical reasoning without a complete formalisation. In particular, we present QuaSAR (for Quasi-Symbolic Abstract Reasoning), a variation of CoT that guides LLMs to operate at a higher level of abstraction via quasi-symbolic explanations. Our framework leverages the capability of LLMs to formalise only relevant variables and predicates, enabling the coexistence of symbolic elements with natural language. We show the impact of QuaSAR for in-context learning and for constructing demonstrations to improve the reasoning capabilities of smaller models. Our experiments show that quasi-symbolic abstractions can improve CoT-based methods by up to 8% accuracy, enhancing robustness and consistency on challenging adversarial variations on both natural language (i.e. MMLU-Redux) and symbolic reasoning tasks (i.e., GSM-Symbolic). 

**Abstract (ZH)**: 链式思考（CoT）代表了一种在大型语言模型（LLM）中进行推理的常见策略，通过将复杂任务分解为中间推理步骤。然而，通过CoT生成的解释可能存在内容偏见，这对其稳健性和忠实性产生了负面影响。为了减轻现有限制，最近的研究提出了结合外部符号求解器的逻辑形式化方法。然而，完全符号化的方法存在瓶颈，即需要将自然语言完全翻译为形式语言，这一过程影响了效率和灵活性。为了在效率和灵活性之间找到平衡，本文探讨了如何在不完全形式化的情况下分离内容与逻辑推理的方法。具体而言，我们提出了QuaSAR（准符号抽象推理），这是一种改进的CoT方法，旨在通过准符号解释引导LLM在更高的抽象层次上运行。我们的框架利用了LLM仅形式化相关变量和谓词的能力，从而实现了符号元素与自然语言的共存。我们展示了QuaSAR在上下文学习中的影响以及在构建示例以提高较小模型推理能力方面的效果。我们的实验表明，准符号抽象可以将基于CoT的方法的准确性提高8%，并增强在自然语言（例如MMLU-Redux）和符号推理任务（例如GSM-Symbolic）中的挑战性对抗变体上的稳健性和一致性。 

---
# Label Drop for Multi-Aspect Relation Modeling in Universal Information Extraction 

**Title (ZH)**: 多方面关系建模的标签-drop 在通用信息提取中的应用 

**Authors**: Lu Yang, Jiajia Li, En Ci, Lefei Zhang, Zuchao Li, Ping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12614)  

**Abstract**: Universal Information Extraction (UIE) has garnered significant attention due to its ability to address model explosion problems effectively. Extractive UIE can achieve strong performance using a relatively small model, making it widely adopted. Extractive UIEs generally rely on task instructions for different tasks, including single-target instructions and multiple-target instructions. Single-target instruction UIE enables the extraction of only one type of relation at a time, limiting its ability to model correlations between relations and thus restricting its capability to extract complex relations. While multiple-target instruction UIE allows for the extraction of multiple relations simultaneously, the inclusion of irrelevant relations introduces decision complexity and impacts extraction accuracy. Therefore, for multi-relation extraction, we propose LDNet, which incorporates multi-aspect relation modeling and a label drop mechanism. By assigning different relations to different levels for understanding and decision-making, we reduce decision confusion. Additionally, the label drop mechanism effectively mitigates the impact of irrelevant relations. Experiments show that LDNet outperforms or achieves competitive performance with state-of-the-art systems on 9 tasks, 33 datasets, in both single-modal and multi-modal, few-shot and zero-shot settings.\footnote{this https URL} 

**Abstract (ZH)**: 通用信息提取（UIE）因其能够有效解决模型爆炸问题而引起了广泛关注。抽取式UIE能够使用相对较小的模型实现良好的性能，因此被广泛采用。抽取式UIE通常依赖于不同任务的任务指令，包括单目标指令和多目标指令。单目标指令UIE只能一次提取一种关系，这限制了它建模关系间关联的能力，从而限制了其提取复杂关系的能力。而多目标指令UIE允许同时提取多种关系，但由于包括无关关系在内，增加了决策复杂性并影响提取准确性。因此，为了进行多关系提取，我们提出了LDNet，它结合了多方面关系建模和标签剔除机制。通过将不同关系分配到不同的层次进行理解和决策，我们减少了决策混乱。此外，标签剔除机制有效缓解了无关关系的影响。实验结果表明，LDNet在9个任务、33个数据集上，在单模态和多模态、少量样本和零样本设置中均优于或与当前最先进的系统具有竞争力。\footnote{该研究的详细内容和链接：this https URL} 

---
# Who Writes What: Unveiling the Impact of Author Roles on AI-generated Text Detection 

**Title (ZH)**: 《谁在撰写：揭示作者角色对AI生成文本检测影响的研究》

这个标题翻译成中文既符合学术规范，又保留了原文的意思。如果有具体的论文内容需要翻译，请提供具体内容，我会帮助您进行翻译。 

**Authors**: Jiatao Li, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2502.12611)  

**Abstract**: The rise of Large Language Models (LLMs) necessitates accurate AI-generated text detection. However, current approaches largely overlook the influence of author characteristics. We investigate how sociolinguistic attributes-gender, CEFR proficiency, academic field, and language environment-impact state-of-the-art AI text detectors. Using the ICNALE corpus of human-authored texts and parallel AI-generated texts from diverse LLMs, we conduct a rigorous evaluation employing multi-factor ANOVA and weighted least squares (WLS). Our results reveal significant biases: CEFR proficiency and language environment consistently affected detector accuracy, while gender and academic field showed detector-dependent effects. These findings highlight the crucial need for socially aware AI text detection to avoid unfairly penalizing specific demographic groups. We offer novel empirical evidence, a robust statistical framework, and actionable insights for developing more equitable and reliable detection systems in real-world, out-of-domain contexts. This work paves the way for future research on bias mitigation, inclusive evaluation benchmarks, and socially responsible LLM detectors. 

**Abstract (ZH)**: 大语言模型（LLMs）的兴起迫切需要准确的AI生成文本检测技术。然而，当前的方法大多忽视了作者特征的影响。我们研究了社会语言学属性（性别、CEFR水平、学术领域和语言环境）如何影响最先进的AI文本检测器的效果。通过使用ICNALE语料库中的人类撰写的文本和来自多种LLM的平行AI生成文本，我们采用多因素方差分析（ANOVA）和加权最小二乘法（WLS）进行了严格的评估。研究结果揭示了明显的偏差：CEFR水平和语言环境对检测器的准确性影响一致，而性别和学术领域则表现出检测器相关的效应。这些发现突显了在避免不公平地惩罚特定人口群体方面，社会意识强的AI文本检测技术的重要性。我们提供了新的实证证据、稳健的统计框架以及在实际应用和外域情境下开发更公平、可靠检测系统的操作性见解。这项研究为未来的研究指明了方向，包括偏见缓解、包容性评估基准和负责的LLM检测器开发。 

---
# COPU: Conformal Prediction for Uncertainty Quantification in Natural Language Generation 

**Title (ZH)**: COPU：自然语言生成中不确定性量化的一种可信预测方法 

**Authors**: Sean Wang, Yicheng Jiang, Yuxin Tang, Lu Cheng, Hanjie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.12601)  

**Abstract**: Uncertainty Quantification (UQ) for Natural Language Generation (NLG) is crucial for assessing the performance of Large Language Models (LLMs), as it reveals confidence in predictions, identifies failure modes, and gauges output reliability. Conformal Prediction (CP), a model-agnostic method that generates prediction sets with a specified error rate, has been adopted for UQ in classification tasks, where the size of the prediction set indicates the model's uncertainty. However, when adapting CP to NLG, the sampling-based method for generating candidate outputs cannot guarantee the inclusion of the ground truth, limiting its applicability across a wide range of error rates. To address this, we propose \ourmethod, a method that explicitly adds the ground truth to the candidate outputs and uses logit scores to measure nonconformity. Our experiments with six LLMs on four NLG tasks show that \ourmethod outperforms baseline methods in calibrating error rates and empirical cover rates, offering accurate UQ across a wide range of user-specified error rates. 

**Abstract (ZH)**: 自然语言生成（NLG）中的不确定性量化（UQ）对于评估大型语言模型（LLMs）的表现至关重要，因为它能够揭示预测的信心、识别失败模式并衡量输出的可靠性。一种模型无关的方法——符合性预测（Conformal Prediction, CP），可以通过指定错误率生成预测集，这种方法在分类任务中已经被用于不确定性量化，预测集的大小反映了模型的不确定性。然而，在将CP应用于NLG时，基于采样的候选输出生成方法不能保证包含真实标签，这限制了其在各种错误率范围内的适用性。为了解决这个问题，我们提出了一种名为\ourmethod的方法，该方法明确地将真实标签包含在候选输出中，并使用logit分数来衡量不符合性。我们在六种LLM上进行的四项NLG任务实验结果表明，\ourmethod 在校准错误率和经验覆盖率方面优于基线方法，能够在广泛的用户指定错误率范围内提供准确的不确定性量化。 

---
# Bring Your Own Knowledge: A Survey of Methods for LLM Knowledge Expansion 

**Title (ZH)**: 自备知识：大规模语言模型知识扩展方法综述 

**Authors**: Mingyang Wang, Alisa Stoll, Lukas Lange, Heike Adel, Hinrich Schütze, Jannik Strötgen  

**Link**: [PDF](https://arxiv.org/pdf/2502.12598)  

**Abstract**: Adapting large language models (LLMs) to new and diverse knowledge is essential for their lasting effectiveness in real-world applications. This survey provides an overview of state-of-the-art methods for expanding the knowledge of LLMs, focusing on integrating various knowledge types, including factual information, domain expertise, language proficiency, and user preferences. We explore techniques, such as continual learning, model editing, and retrieval-based explicit adaptation, while discussing challenges like knowledge consistency and scalability. Designed as a guide for researchers and practitioners, this survey sheds light on opportunities for advancing LLMs as adaptable and robust knowledge systems. 

**Abstract (ZH)**: 适应大语言模型（LLMs）的新颖和多元知识对于其实际应用中的持续有效性至关重要。本综述概述了最新的方法，用于扩展LLMs的知识，重点关注整合多种类型的知识，包括事实信息、领域专业知识、语言 proficiency 和用户偏好。我们探讨了持续学习、模型编辑和基于检索的显式适应等技术，同时讨论了知识一致性与可扩展性等挑战。作为研究者和实践者的指南，本综述揭示了改进LLMs为适应性强且鲁棒的知识系统的机遇。 

---
# PASER: Post-Training Data Selection for Efficient Pruned Large Language Model Recovery 

**Title (ZH)**: PASER：高效剪枝大型语言模型训练后数据选择恢复方法 

**Authors**: Bowei He, Lihao Yin, Hui-Ling Zhen, Xiaokun Zhang, Mingxuan Yuan, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2502.12594)  

**Abstract**: Model pruning is an effective approach for compressing large language models. However, this process often leads to significant degradation of model capabilities. While post-training techniques such as instruction tuning are commonly employed to recover model performance, existing methods often overlook the uneven deterioration of model capabilities and incur high computational costs. Moreover, some instruction data irrelevant to model capability recovery may introduce negative effects. To address these challenges, we propose the \textbf{P}ost-training d\textbf{A}ta \textbf{S}election method for \textbf{E}fficient pruned large language model \textbf{R}ecovery (\textbf{PASER}). PASER aims to identify instructions where model capabilities are most severely compromised within a certain recovery data budget. Our approach first applies manifold learning and spectral clustering to group recovery data in the semantic space, revealing capability-specific instruction sets. We then adaptively allocate the data budget to different clusters based on the degrees of model capability degradation. In each cluster, we prioritize data samples where model performance has declined dramatically. To mitigate potential negative transfer, we also detect and filter out conflicting or irrelevant recovery data. Extensive experiments demonstrate that PASER significantly outperforms conventional baselines, effectively recovering the general capabilities of pruned LLMs while utilizing merely 4\%-20\% of the original post-training data. 

**Abstract (ZH)**: 模型剪枝是压缩大型语言模型的有效方法，但这一过程通常会导致模型能力显著下降。尽管训练后技术如指令调优常被用来恢复模型性能，但现有方法往往忽略了模型能力分布不均的下降，并且会带来较高的计算成本。此外，一些与模型能力恢复无关的指令数据可能会产生负面影响。为了解决这些挑战，我们提出了一种称为PASER（Post-Training Data Selection for Efficient Recovery of Pruned Large Language Models）的方法。PASER旨在在一个特定的恢复数据预算下识别出模型能力最严重受损的指令。

我们的方法首先使用流形学习和谱聚类将恢复数据在语义空间中分组，揭示出特定能力的指令集。然后，根据模型能力下降的程度，适应性地分配数据预算到不同的聚类中。在每个聚类中，我们优先选择模型性能显著下降的数据样本。为了减轻潜在的负迁移，我们还检测并过滤掉冲突或无关的恢复数据。广泛的经验表明，PASER 显著优于传统的基线方法，在利用仅4%-20%的原始训练后数据的情况下，有效地恢复了剪枝的大语言模型的一般能力。 

---
# RSMLP: A light Sampled MLP Structure for Incomplete Utterance Rewrite 

**Title (ZH)**: RSMLP：一种用于不完整话语重写的轻量级采样MLP结构 

**Authors**: Lunjun Liu, Weilai Jiang, Yaonan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12587)  

**Abstract**: The Incomplete Utterance Rewriting (IUR) task has garnered significant attention in recent years. Its goal is to reconstruct conversational utterances to better align with the current context, thereby enhancing comprehension. In this paper, we introduce a novel and versatile lightweight method, Rewritten-Sampled MLP (RSMLP). By employing an MLP based architecture with a carefully designed down-sampling strategy, RSMLP effectively extracts latent semantic information between utterances and makes appropriate edits to restore incomplete utterances. Due to its simple yet efficient structure, our method achieves competitive performance on public IUR datasets and in real-world applications. 

**Abstract (ZH)**: 不完备命题重写（IUR）任务近年来受到了广泛关注。其目标是重构对话中的命题，使其更好地与当前背景相匹配，从而提高理解度。在本文中，我们提出了一种新颖且通用的轻量级方法——重写采样MLP（RSMLP）。该方法采用基于MLP的架构，并结合一种精心设计的下采样策略，有效抽取了命题间的潜在语义信息，并进行适当的编辑以恢复不完整的命题。由于其简单而高效的结构，我们的方法在公共IUR数据集和实际应用中均取得了竞争力的表现。 

---
# LongFaith: Enhancing Long-Context Reasoning in LLMs with Faithful Synthetic Data 

**Title (ZH)**: LongFaith: 通过忠实合成数据增强大型语言模型的长上下文推理能力 

**Authors**: Cehao Yang, Xueyuan Lin, Chengjin Xu, Xuhui Jiang, Shengjie Ma, Aofan Liu, Hui Xiong, Jian Guo  

**Link**: [PDF](https://arxiv.org/pdf/2502.12583)  

**Abstract**: Despite the growing development of long-context large language models (LLMs), data-centric approaches relying on synthetic data have been hindered by issues related to faithfulness, which limit their effectiveness in enhancing model performance on tasks such as long-context reasoning and question answering (QA). These challenges are often exacerbated by misinformation caused by lack of verification, reasoning without attribution, and potential knowledge conflicts. We propose LongFaith, a novel pipeline for synthesizing faithful long-context reasoning instruction datasets. By integrating ground truth and citation-based reasoning prompts, we eliminate distractions and improve the accuracy of reasoning chains, thus mitigating the need for costly verification processes. We open-source two synthesized datasets, LongFaith-SFT and LongFaith-PO, which systematically address multiple dimensions of faithfulness, including verified reasoning, attribution, and contextual grounding. Extensive experiments on multi-hop reasoning datasets and LongBench demonstrate that models fine-tuned on these datasets significantly improve performance. Our ablation studies highlight the scalability and adaptability of the LongFaith pipeline, showcasing its broad applicability in developing long-context LLMs. 

**Abstract (ZH)**: 尽管长上下文大规模语言模型（LLMs）取得了快速发展，依赖合成数据的数据为中心的方法因其忠实性问题而受到限制，这在提高模型在长上下文推理和问答（QA）等任务上的性能方面效果有限。这些问题往往因缺乏验证、未标明来源的推理以及潜在的知识冲突而加剧。为此，我们提出了一种名为LongFaith的新颖合成忠实长上下文推理指令数据集管道。通过整合基于事实和引文的推理提示，我们消除了干扰并提高了推理链的准确性，从而减轻了对昂贵验证过程的依赖。我们开源了两个合成数据集，即LongFaith-SFT和LongFaith-PO，这些数据集系统地解决了忠实性多个维度的问题，包括验证推理、归因和背景联系。在多跳推理数据集和LongBench上的广泛实验表明，使用这些数据集微调的模型性能显著提升。我们的消融研究展示了LongFaith管道的可扩展性和适应性，展示了其在开发长上下文LLMs方面的广泛应用潜力。 

---
# A Fuzzy Evaluation of Sentence Encoders on Grooming Risk Classification 

**Title (ZH)**: 对梳理风险分类的句编码器进行模糊评价 

**Authors**: Geetanjali Bihani, Julia Rayz  

**Link**: [PDF](https://arxiv.org/pdf/2502.12576)  

**Abstract**: With the advent of social media, children are becoming increasingly vulnerable to the risk of grooming in online settings. Detecting grooming instances in an online conversation poses a significant challenge as the interactions are not necessarily sexually explicit, since the predators take time to build trust and a relationship with their victim. Moreover, predators evade detection using indirect and coded language. While previous studies have fine-tuned Transformers to automatically identify grooming in chat conversations, they overlook the impact of coded and indirect language on model predictions, and how these align with human perceptions of grooming. In this paper, we address this gap and evaluate bi-encoders on the task of classifying different degrees of grooming risk in chat contexts, for three different participant groups, i.e. law enforcement officers, real victims, and decoys. Using a fuzzy-theoretic framework, we map human assessments of grooming behaviors to estimate the actual degree of grooming risk. Our analysis reveals that fine-tuned models fail to tag instances where the predator uses indirect speech pathways and coded language to evade detection. Further, we find that such instances are characterized by a higher presence of out-of-vocabulary (OOV) words in samples, causing the model to misclassify. Our findings highlight the need for more robust models to identify coded language from noisy chat inputs in grooming contexts. 

**Abstract (ZH)**: 随着社交媒体的发展，儿童在在线环境中面临诱骗风险的可能性越来越大。在在线对话中检测诱骗事件是一项重大挑战，因为互动未必具有性暗示性，因为诱捕者需要时间来建立信任和与受害者的亲密关系。此外，诱捕者会利用间接和编码的语言逃避检测。尽管以往的研究已经对Transformer进行了微调，以自动识别聊天对话中的诱捕行为，但这些研究忽略了编码和间接语言对模型预测的影响，以及这些影响与人类对诱捕行为的看法之间的差异。本文旨在填补这一 gaps，并评估双编码器在分类不同程度的诱捕风险任务中的表现，针对三种不同的参与者群体，即执法人员、真实受害者和诱饵。利用模糊理论框架，我们将人类对诱骗行为的评估映射到估算实际的诱捕风险等级。我们的分析表明，微调后的模型无法标记诱捕者使用间接言辞途径和编码语言来逃避检测的实例。进一步的研究发现，这些实例中的非词表词汇（OOV词）出现频率较高，导致模型错误分类。我们的研究结果强调了在诱拐情境中从嘈杂的聊天输入中识别编码语言的必要性，需要构建更稳健的模型。 

---
# A Cognitive Writing Perspective for Constrained Long-Form Text Generation 

**Title (ZH)**: 一种认知写作视角下的约束长文本生成研究 

**Authors**: Kaiyang Wan, Honglin Mu, Rui Hao, Haoran Luo, Tianle Gu, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.12568)  

**Abstract**: Like humans, Large Language Models (LLMs) struggle to generate high-quality long-form text that adheres to strict requirements in a single pass. This challenge is unsurprising, as successful human writing, according to the Cognitive Writing Theory, is a complex cognitive process involving iterative planning, translating, reviewing, and monitoring. Motivated by these cognitive principles, we aim to equip LLMs with human-like cognitive writing capabilities through CogWriter, a novel training-free framework that transforms LLM constrained long-form text generation into a systematic cognitive writing paradigm. Our framework consists of two key modules: (1) a Planning Agent that performs hierarchical planning to decompose the task, and (2) multiple Generation Agents that execute these plans in parallel. The system maintains quality via continuous monitoring and reviewing mechanisms, which evaluate outputs against specified requirements and trigger necessary revisions. CogWriter demonstrates exceptional performance on LongGenBench, a benchmark for complex constrained long-form text generation. Even when using Qwen-2.5-14B as its backbone, CogWriter surpasses GPT-4o by 22% in complex instruction completion accuracy while reliably generating texts exceeding 10,000 words. We hope this cognitive science-inspired approach provides a paradigm for LLM writing advancements: \href{this https URL}{CogWriter}. 

**Abstract (ZH)**: 如人类一样，大型语言模型（LLMs）在单次生成高质量长篇文本时往往难以符合严格的要求。这一挑战并不令人意外，因为认知写作理论认为，成功的写作是一个复杂的认知过程，涉及迭代规划、翻译、审查和监控等多个步骤。受这些认知原则的启发，我们打算通过CogWriter这一新颖的无需训练框架，赋予LLMs类似于人类的认知写作能力。该框架包括两个关键模块：（1）一个规划代理，执行分层规划以分解任务；（2）多个生成代理，同时执行这些计划。系统通过持续的监控和审查机制来维持高质量，这些机制评估输出是否符合特定要求，并触发必要的修改。CogWriter在LongGenBench（一个复杂约束长篇文本生成基准）上表现出色。即使使用Qwen-2.5-14B作为其基础模型，CogWriter在复杂指令完成准确性方面也比GPT-4o高22%，同时可靠地生成超过10,000个单词的文本。我们希望这种基于认知科学的方法能够为LLM写作的进步提供一种范式：\[CogWriter\]（请参阅\[这个链接\]）。 

---
# Self Iterative Label Refinement via Robust Unlabeled Learning 

**Title (ZH)**: 自迭代标签精炼通过稳健的未标注学习 

**Authors**: Hikaru Asano, Tadashi Kozuno, Yukino Baba  

**Link**: [PDF](https://arxiv.org/pdf/2502.12565)  

**Abstract**: Recent advances in large language models (LLMs) have yielded impressive performance on various tasks, yet they often depend on high-quality feedback that can be costly. Self-refinement methods attempt to leverage LLMs' internal evaluation mechanisms with minimal human supervision; however, these approaches frequently suffer from inherent biases and overconfidence, especially in domains where the models lack sufficient internal knowledge, resulting in performance degradation. As an initial step toward enhancing self-refinement for broader applications, we introduce an iterative refinement pipeline that employs the Unlabeled-Unlabeled learning framework to improve LLM-generated pseudo-labels for classification tasks. By exploiting two unlabeled datasets with differing positive class ratios, our approach iteratively denoises and refines the initial pseudo-labels, thereby mitigating the adverse effects of internal biases with minimal human supervision. Evaluations on diverse datasets, including low-resource language corpora, patent classifications, and protein structure categorizations, demonstrate that our method consistently outperforms both initial LLM's classification performance and the self-refinement approaches by cutting-edge models (e.g., GPT-4o and DeepSeek-R1). 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）在各种任务上取得了显著的性能，但仍往往依赖于高质量的反馈，而这可能较为昂贵。自我校hyth
user
请继续翻译，确保翻译的准确性和学术性。 

---
# Evaluating Language Models on Grooming Risk Estimation Using Fuzzy Theory 

**Title (ZH)**: 使用模糊理论评估语言模型在 grooming 风险估算中的表现 

**Authors**: Geetanjali Bihani, Tatiana Ringenberg, Julia Rayz  

**Link**: [PDF](https://arxiv.org/pdf/2502.12563)  

**Abstract**: Encoding implicit language presents a challenge for language models, especially in high-risk domains where maintaining high precision is important. Automated detection of online child grooming is one such critical domain, where predators manipulate victims using a combination of explicit and implicit language to convey harmful intentions. While recent studies have shown the potential of Transformer language models like SBERT for preemptive grooming detection, they primarily depend on surface-level features and approximate real victim grooming processes using vigilante and law enforcement conversations. The question of whether these features and approximations are reasonable has not been addressed thus far. In this paper, we address this gap and study whether SBERT can effectively discern varying degrees of grooming risk inherent in conversations, and evaluate its results across different participant groups. Our analysis reveals that while fine-tuning aids language models in learning to assign grooming scores, they show high variance in predictions, especially for contexts containing higher degrees of grooming risk. These errors appear in cases that 1) utilize indirect speech pathways to manipulate victims and 2) lack sexually explicit content. This finding underscores the necessity for robust modeling of indirect speech acts by language models, particularly those employed by predators. 

**Abstract (ZH)**: 语言模型在编码隐含语言方面面临挑战，尤其是在高风险领域中，保持高精度尤为重要。自动检测在线诱骗儿童是一种关键性的领域，罪犯通过显性和隐性语言的结合来传达有害意图，从而操纵受害者。虽然最近的研究表明，如SBERT等变换器语言模型在诱骗检测中具有潜在的作用，但它们主要依赖于表面特征，并且使用义工和执法机构的对话来近似实际的受害人的诱骗过程。至于这些特征和近似是否合理的问题，目前尚未得到解答。在本文中，我们填补了这一空白，并研究SBERT能否有效地辨别对话中不同层次的诱骗风险，并在不同参与者群体中评估其结果。我们的分析表明，虽然微调有助于语言模型学会分配诱骗评分，但它们在预测方面表现出高变异性，尤其是在包含较高层次诱骗风险的上下文中。这些错误出现在以下两种情况中：1）采用间接言语途径来操纵受害者；2）缺乏直接的性内容。这一发现凸显了语言模型在建模间接言辞行为方面的需求，特别是对于罪犯使用这些模型的情形。 

---
# SEA: Low-Resource Safety Alignment for Multimodal Large Language Models via Synthetic Embeddings 

**Title (ZH)**: SEA：通过合成嵌入实现多模态大型语言模型的低资源安全对齐 

**Authors**: Weikai Lu, Hao Peng, Huiping Zhuang, Cen Chen, Ziqian Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2502.12562)  

**Abstract**: Multimodal Large Language Models (MLLMs) have serious security this http URL safety alignment using multimodal datasets consisting of text and data of additional modalities can effectively enhance MLLM's security, it is costly to construct these datasets. Existing low-resource security alignment methods, including textual alignment, have been found to struggle with the security risks posed by additional modalities. To address this, we propose Synthetic Embedding augmented safety Alignment (SEA), which optimizes embeddings of additional modality through gradient updates to expand textual datasets. This enables multimodal safety alignment training even when only textual data is available. Extensive experiments on image, video, and audio-based MLLMs demonstrate that SEA can synthesize a high-quality embedding on a single RTX3090 GPU within 24 seconds. SEA significantly improves the security of MLLMs when faced with threats from additional modalities. To assess the security risks introduced by video and audio, we also introduced a new benchmark called VA-SafetyBench. High attack success rates across multiple MLLMs validate its challenge. Our code and data will be available at this https URL. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）存在严重的安全问题。为了增强MLLM的安全性，可以通过使用包含文本和其他模态数据的多模态数据集来进行安全对齐，但构建这些数据集的成本很高。现有的低资源安全对齐方法，如文本对齐，已被发现难以应对由其他模态所带来的安全风险。为此，我们提出了一种合成嵌入增强安全对齐（SEA）方法，通过梯度更新优化其他模态的嵌入，以扩展文本数据集。即使仅使用文本数据，也可以实现多模态安全对齐训练。在基于图像、视频和音频的MLLMs上的大量实验表明，SEA能够在单块RTX3090 GPU上于24秒内合成高质量的嵌入。SEA显著提高了MLLMs在面临其他模态带来的威胁时的安全性。为了评估由视频和音频引入的安全风险，我们还引入了一个新的基准测试VA-SafetyBench。在多个MLLMs上多次高成功率的攻击验证了其挑战性。我们的代码和数据将在此网址发布：[参考网址]。 

---
# How does a Language-Specific Tokenizer affect LLMs? 

**Title (ZH)**: 语言特定分词器如何影响大语言模型？ 

**Authors**: Jean Seo, Jaeyoon Kim, SungJoo Byun, Hyopil Shin  

**Link**: [PDF](https://arxiv.org/pdf/2502.12560)  

**Abstract**: The necessity of language-specific tokenizers intuitively appears crucial for effective natural language processing, yet empirical analyses on their significance and underlying reasons are lacking. This study explores how language-specific tokenizers influence the behavior of Large Language Models predominantly trained with English text data, through the case study of Korean. The research unfolds in two main stages: (1) the development of a Korean-specific extended tokenizer and (2) experiments to compare models with the basic tokenizer and the extended tokenizer through various Next Token Prediction tasks. Our in-depth analysis reveals that the extended tokenizer decreases confidence in incorrect predictions during generation and reduces cross-entropy in complex tasks, indicating a tendency to produce less nonsensical outputs. Consequently, the extended tokenizer provides stability during generation, potentially leading to higher performance in downstream tasks. 

**Abstract (ZH)**: 语言特异性分词器在自然语言处理中的必要性直观上看显然是关键的，但对其重要性及背后原因的实证分析却相对缺乏。本研究通过韩语案例，探讨语言特异性分词器对主要使用英文文本数据训练的大规模语言模型行为的影响。研究分为两个主要阶段：（1）开发一种韩语特定的扩展分词器；（2）通过各种下一个词预测任务比较使用基本分词器和扩展分词器的模型。我们的深入分析表明，扩展分词器降低了生成过程中错误预测的置信度，并在复杂任务中减少了交叉熵，这表明它倾向于产生更加合理的内容。因此，扩展分词器在生成过程中提供了稳定性，有可能在下游任务中提高模型性能。 

---
# Policy-to-Language: Train LLMs to Explain Decisions with Flow-Matching Generated Rewards 

**Title (ZH)**: 政策到语言：通过流匹配生成的奖励训练大语言模型解释决策 

**Authors**: Xinyi Yang, Liang Zeng, Heng Dong, Chao Yu, Xiaoran Wu, Huazhong Yang, Yu Wang, Milind Tambe, Tonghan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12530)  

**Abstract**: As humans increasingly share environments with diverse agents powered by RL, LLMs, and beyond, the ability to explain their policies in natural language will be vital for reliable coexistence. In this paper, we build a model-agnostic explanation generator based on an LLM. The technical novelty is that the rewards for training this LLM are generated by a generative flow matching model. This model has a specially designed structure with a hidden layer merged with an LLM to harness the linguistic cues of explanations into generating appropriate rewards. Experiments on both RL and LLM tasks demonstrate that our method can generate dense and effective rewards while saving on expensive human feedback; it thus enables effective explanations and even improves the accuracy of the decisions in original tasks. 

**Abstract (ZH)**: 随着人类越来越多地与被强化学习（RL）、大规模语言模型（LLMs）以及其他技术驱动的多样化代理共存，以自然语言解释其策略的能力对于可靠的共存将变得至关重要。本文中，我们基于LLM构建了一个模型无关的解释生成器。该技术的创新之处在于，用于训练这个LLM的奖励是由一个生成流匹配模型生成的。该模型具有一专门设计的结构，其中隐藏层与LLM结合，以利用解释中的语言线索来生成适当的奖励。实验结果表明，我们的方法能够在节省昂贵的人工反馈成本的同时，生成密集且有效的奖励；因此，它可以促进有效的解释，并甚至可以提高原始任务中决策的准确性。 

---
# Can LLMs Extract Frame-Semantic Arguments? 

**Title (ZH)**: 大型语言模型能否提取框架语义论元？ 

**Authors**: Jacob Devasier, Rishabh Mediratta, Chengkai Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.12516)  

**Abstract**: Frame-semantic parsing is a critical task in natural language understanding, yet the ability of large language models (LLMs) to extract frame-semantic arguments remains underexplored. This paper presents a comprehensive evaluation of LLMs on frame-semantic argument identification, analyzing the impact of input representation formats, model architectures, and generalization to unseen and out-of-domain samples. Our experiments, spanning models from 0.5B to 78B parameters, reveal that JSON-based representations significantly enhance performance, and while larger models generally perform better, smaller models can achieve competitive results through fine-tuning. We also introduce a novel approach to frame identification leveraging predicted frame elements, achieving state-of-the-art performance on ambiguous targets. Despite strong generalization capabilities, our analysis finds that LLMs still struggle with out-of-domain data. 

**Abstract (ZH)**: 框架语义解析是自然语言理解中的关键任务，然而大型语言模型（LLMs）在提取框架语义论元的能力方面尚未得到充分探索。本文对LLMs在框架语义论元识别方面的性能进行了全面评估，并分析了输入表示格式、模型架构以及对未见过和领域外样本的泛化能力的影响。我们的实验覆盖了从500百万到780亿参数多种模型，结果显示基于JSON的表示显著提高了性能，尽管更大规模的模型一般表现更好，但通过微调，小型模型也能达到与之竞争的结果。此外，我们还介绍了一种新颖的框架识别方法，利用预测的框架元素，该方法在含糊目标上达到了最先进的性能。尽管LLMs具有较强的泛化能力，但我们的分析表明，它们仍然难以处理领域外数据。 

---
# Aspect-Guided Multi-Level Perturbation Analysis of Large Language Models in Automated Peer Review 

**Title (ZH)**: 面向方面的大规模语言模型自动化同伴评审的多层次扰动分析 

**Authors**: Jiatao Li, Yanheng Li, Xinyu Hu, Mingqi Gao, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2502.12510)  

**Abstract**: We propose an aspect-guided, multi-level perturbation framework to evaluate the robustness of Large Language Models (LLMs) in automated peer review. Our framework explores perturbations in three key components of the peer review process-papers, reviews, and rebuttals-across several quality aspects, including contribution, soundness, presentation, tone, and completeness. By applying targeted perturbations and examining their effects on both LLM-as-Reviewer and LLM-as-Meta-Reviewer, we investigate how aspect-based manipulations, such as omitting methodological details from papers or altering reviewer conclusions, can introduce significant biases in the review process. We identify several potential vulnerabilities: review conclusions that recommend a strong reject may significantly influence meta-reviews, negative or misleading reviews may be wrongly interpreted as thorough, and incomplete or hostile rebuttals can unexpectedly lead to higher acceptance rates. Statistical tests show that these biases persist under various Chain-of-Thought prompting strategies, highlighting the lack of robust critical evaluation in current LLMs. Our framework offers a practical methodology for diagnosing these vulnerabilities, thereby contributing to the development of more reliable and robust automated reviewing systems. 

**Abstract (ZH)**: 我们提出了一种基于方面导向、多层扰动框架，以评估大型语言模型（LLMs）在自动同伴评审中的稳健性。该框架在投稿、评审和反驳三个关键评审组件上进行了探索，并涵盖了多个质量维度，包括贡献、准确性、呈现、语气和完整性。通过应用有针对性的扰动并检验其对LLM作为评审员和LLM作为元评审员的影响，我们研究了基于方面的方法如何引入对评审过程的重大偏差，例如省略论文中的方法学细节或改变评审员的结论。我们发现了一些潜在的脆弱性：推荐强烈拒绝的评审结论可能显著影响元评审，负面或误导性的评审可能被错误地解读为全面细致的评审，而不完整或敌对的反驳可能会意外地导致更高的接受率。统计测试表明，在各种思维链提示策略下，这些偏差依然存在，突显了当前LLMs缺乏稳健的批判性评估能力。该框架提供了一种实用的方法来诊断这些脆弱性，从而有助于开发更可靠和稳健的自动评审系统。 

---
# LegalCore: A Dataset for Legal Documents Event Coreference Resolution 

**Title (ZH)**: LegalCore：法律文件事件共指解析数据集 

**Authors**: Kangda Wei, Xi Shi, Jonathan Tong, Sai Ramana Reddy, Anandhavelu Natarajan, Rajiv Jain, Aparna Garimella, Ruihong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12509)  

**Abstract**: Recognizing events and their coreferential mentions in a document is essential for understanding semantic meanings of text. The existing research on event coreference resolution is mostly limited to news articles. In this paper, we present the first dataset for the legal domain, LegalCore, which has been annotated with comprehensive event and event coreference information. The legal contract documents we annotated in this dataset are several times longer than news articles, with an average length of around 25k tokens per document. The annotations show that legal documents have dense event mentions and feature both short-distance and super long-distance coreference links between event mentions. We further benchmark mainstream Large Language Models (LLMs) on this dataset for both event detection and event coreference resolution tasks, and find that this dataset poses significant challenges for state-of-the-art open-source and proprietary LLMs, which perform significantly worse than a supervised baseline. We will publish the dataset as well as the code. 

**Abstract (ZH)**: 识别文档中事件及其同指提及对于理解文本语义意义至关重要。现有关于事件同指消解的研究主要集中在新闻文章上。本文介绍了一个首个适用于法律领域的数据集——LegalCore，该数据集已被注释了全面的事件及事件同指信息。我们在此数据集中注释的法律合同文件长度远超新闻文章，平均每份文档包含约25k个词汇。注释结果显示，法律文件中的事件提及密集，事件提及之间存在短距离和超长距离的同指链接。我们进一步在该数据集上 benchmarks 主流大规模语言模型（LLMs）的事件检测和事件同指消解能力，并发现这些数据集对最先进的开源和专有LLMs构成了显著挑战，这些模型的表现远逊于有监督的基础模型。我们将发布该数据集及其对应的代码。 

---
# Efficient OpAmp Adaptation for Zoom Attention to Golden Contexts 

**Title (ZH)**: 高效的运算放大器适配以关注金标准上下文的变焦注意力机制 

**Authors**: Haoyuan Wu, Rui Ming, Haisheng Zheng, Zhuolun He, Bei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12502)  

**Abstract**: Large language models (LLMs) have shown significant promise in question-answering (QA) tasks, particularly in retrieval-augmented generation (RAG) scenarios and long-context applications. However, their performance is hindered by noisy reference documents, which often distract from essential information. Despite fine-tuning efforts, Transformer-based architectures struggle to prioritize relevant content. This is evidenced by their tendency to allocate disproportionate attention to irrelevant or later-positioned documents. Recent work proposes the differential attention mechanism to address this issue, but this mechanism is limited by an unsuitable common-mode rejection ratio (CMRR) and high computational costs. Inspired by the operational amplifier (OpAmp), we propose the OpAmp adaptation to address these challenges, which is implemented with adapters efficiently. By integrating the adapter into pre-trained Transformer blocks, our approach enhances focus on the golden context without costly training from scratch. Empirical evaluations on noisy-context benchmarks reveal that our Qwen2.5-OpAmp-72B model, trained with our OpAmp adaptation, surpasses the performance of state-of-the-art LLMs, including DeepSeek-V3 and GPT-4o. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在问答（QA）任务中展示了显著的潜力，特别是在检索增强生成（RAG）场景和长上下文应用中。然而，它们的性能受到嘈杂参考文档的限制，这些文档常常分散了重要的信息。尽管进行了细调努力，基于Transformer的架构在优先处理相关内容方面仍然存在困难。这体现在它们倾向于不适当分配注意力给无关或位于后期的文档上。近期研究表明，差分注意机制可以解决这一问题，但该机制因其不合适的共同模式抑制比（CMRR）和高计算成本而受到限制。受运算放大器（OpAmp）的启发，我们提出了一种OpAmp适应机制来应对这些挑战，该机制通过高效地使用适配器实现。通过将适配器整合到预训练Transformer块中，我们的方法增强了对黄金上下文的聚焦，而无需从头开始的昂贵训练。在嘈杂上下文基准测试中的实验评估表明，我们训练的Qwen2.5-OpAmp-72B模型超过了包括DeepSeek-V3和GPT-4o在内的最新LLM的性能。 

---
# Crowd Comparative Reasoning: Unlocking Comprehensive Evaluations for LLM-as-a-Judge 

**Title (ZH)**: 人群对比推理：解锁LLM作为法官的全面评估 

**Authors**: Qiyuan Zhang, Yufei Wang, Yuxin Jiang, Liangyou Li, Chuhan Wu, Yasheng Wang, Xin Jiang, Lifeng Shang, Ruiming Tang, Fuyuan Lyu, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2502.12501)  

**Abstract**: LLM-as-a-Judge, which generates chain-of-thought (CoT) judgments, has become a widely adopted auto-evaluation method. However, its reliability is compromised by the CoT reasoning's inability to capture comprehensive and deeper details, often leading to incomplete outcomes. Existing methods mainly rely on majority voting or criteria expansion, which is insufficient to address the limitation in CoT. We propose Crowd-based Comparative Evaluation, which introduces additional crowd responses to compare with the candidate responses, thereby exposing deeper and more comprehensive details within the candidate responses. This process effectively guides LLM-as-a-Judge to provide a more detailed CoT judgment. Extensive experiments demonstrate that our approach enhances evaluation reliability, achieving an average accuracy gain of 6.7% across five benchmarks. Moreover, our method produces higher-quality CoTs that facilitate judge distillation and exhibit superior performance in rejection sampling for supervised fine-tuning (SFT), referred to as crowd rejection sampling, thereby enabling more efficient SFT. Our analysis confirms that CoTs generated by ours are more comprehensive and of higher quality, and evaluation accuracy improves as inference scales. 

**Abstract (ZH)**: 将下面的论文内容或标题翻译成中文，要符合学术规范：

LLM-as-a-Judge，其生成链式思维（CoT）判断，已经成为一种广泛采用的自动评估方法。然而，CoT推理无法捕捉到全面和深入的细节，经常导致结果不完整，从而削弱了其可靠性。现有方法主要依赖于多数投票或标准扩展，但不足以解决CoT的局限性。我们提出了一种基于众包的比较评估方法，通过引入额外的众包响应来比较候选响应，从而揭示候选响应中的更深层次和更全面的细节。这个过程有效地引导LLM-as-a-Judge提供更详细的CoT判断。广泛实验证明，我们的方法提高了评估可靠性，在五个基准测试中平均准确率提高了6.7%。此外，我们的方法生成了更高质量的CoT，有助于法官提炼，并在监督微调（SFT）中的拒绝采样方面表现出色，称为众包拒绝采样，从而能够更高效地进行SFT。我们的分析证实，我们生成的CoT更全面和更高质量，并且评估的准确性随着推理规模的扩大而提高。 

---
# UniGenCoder: Merging Seq2Seq and Seq2Tree Paradigms for Unified Code Generation 

**Title (ZH)**: UniGenCoder：结合Seq2Seq和Seq2Tree范式的统一代码生成 

**Authors**: Liangying Shao, Yanfu Yan, Denys Poshyvanyk, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2502.12490)  

**Abstract**: Deep learning-based code generation has completely transformed the way developers write programs today. Existing approaches to code generation have focused either on the Sequence-to-Sequence paradigm, which generates target code as a sequence of tokens, or the Sequence-to-Tree paradigm, which outputs code as a sequence of actions. While these two paradigms are intuitively complementary, their combination has not been previously explored. By comparing the code generated under these two paradigms, we find that integrating them holds significant potential. In this paper, we propose UniGenCoder for code-related generation tasks, which consists of a shared encoder, a shared decoder with a minimal set of additional parameters to unify two paradigms, and a selector that dynamically chooses optimal paradigm for each instance. Also, during the model training, we first perform the multi-task learning and distillation strategies to facilitate knowledge transfer between two paradigms, and then leverage contrastive learning to train the selector. Experimental results on the text-to-code and code-to-code generation tasks demonstrate the effectiveness of our proposed model. We release our code at this https URL. 

**Abstract (ZH)**: 基于深度学习的代码生成已经彻底改变了开发人员编写程序的方式。现有的代码生成方法主要集中在两种范式上：序列到序列（Sequence-to-Sequence, seq2seq）范式，将目标代码生成为一组标记序列；以及序列到树（Sequence-to-Tree, seq2tree）范式，将代码作为一系列动作输出。尽管这两种范式在直觉上具有互补性，但它们的结合从未被探索过。通过比较这两种范式生成的代码，我们发现将它们结合在一起具有显著的潜力。在本文中，我们提出了UniGenCoder，用于代码相关生成任务。UniGenCoder 包含一个共享编码器、一个带有最少附加参数的共享解码器，以及一个选择器，该选择器动态地为每个实例选择最佳范式。此外，在模型训练过程中，我们首先采用多任务学习和蒸馏策略来促进两种范式之间的知识转移，然后利用对比学习训练选择器。在文本到代码和代码到代码生成任务中的实验结果表明，我们提出的模型具有显著的效果。我们已将代码发布在如下链接：[该 https URL]。 

---
# EPO: Explicit Policy Optimization for Strategic Reasoning in LLMs via Reinforcement Learning 

**Title (ZH)**: EPO：通过强化学习进行显式策略优化以在大规模语言模型中实现战略推理 

**Authors**: Xiaoqian Liu, Ke Wang, Yongbin Li, Yuchuan Wu, Wentao Ma, Aobo Kong, Fei Huang, Jianbin Jiao, Junge Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12486)  

**Abstract**: Large Language Models (LLMs) have shown impressive reasoning capabilities in well-defined problems with clear solutions, such as mathematics and coding. However, they still struggle with complex real-world scenarios like business negotiations, which require strategic reasoning-an ability to navigate dynamic environments and align long-term goals amidst uncertainty. Existing methods for strategic reasoning face challenges in adaptability, scalability, and transferring strategies to new contexts. To address these issues, we propose explicit policy optimization (EPO) for strategic reasoning, featuring an LLM that provides strategies in open-ended action space and can be plugged into arbitrary LLM agents to motivate goal-directed behavior. To improve adaptability and policy transferability, we train the strategic reasoning model via multi-turn reinforcement learning (RL) using process rewards and iterative self-play, without supervised fine-tuning (SFT) as a preliminary step. Experiments across social and physical domains demonstrate EPO's ability of long-term goal alignment through enhanced strategic reasoning, achieving state-of-the-art performance on social dialogue and web navigation tasks. Our findings reveal various collaborative reasoning mechanisms emergent in EPO and its effectiveness in generating novel strategies, underscoring its potential for strategic reasoning in real-world applications. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在数学和编程等具有明确解决方案的限定性问题中展示了令人印象深刻的推理能力。然而，它们在需要战略性推理的复杂现实场景中仍表现不佳，例如商务谈判，这要求具备在动态环境中导航并调整长期目标的能力，同时要考虑不确定性。现有战略性推理方法在适应性、可扩展性以及策略在新情境中的转移方面面临挑战。为解决这些问题，我们提出了一种明确的策略优化（Explicit Policy Optimization, EPO）方法，该方法包含一个在开放动作空间中提供策略的大规模语言模型，并且可以插入任意的大规模语言模型代理中以激发目标导向行为。为了提高适应性和策略的可转移性，我们通过过程奖励的多轮强化学习（Multi-turn Reinforcement Learning, RL）训练战略性推理模型，并采用迭代自我对弈的方式，未进行监督微调（Supervised Fine-tuning, SFT）作为初步步骤。在社会和物理领域的实验中，EPO 通过增强的战略性推理展示了长期目标对齐的能力，并在社会对话和网页导航任务中达到了最先进的性能。我们的研究发现揭示了 EPO 中多种协作推理机制的涌现，以及它在生成新颖策略方面的有效性，这突显了其在现实世界应用中的战略性推理潜力。 

---
# Safe at the Margins: A General Approach to Safety Alignment in Low-Resource English Languages -- A Singlish Case Study 

**Title (ZH)**: 处于边缘地带的安全性：一种低资源英语语言安全对齐的一般方法——以新加坡英语（Singlish）为案例研究 

**Authors**: Isaac Lim, Shaun Khoo, Watson Chua, Goh Jiayi, Jessica Foo  

**Link**: [PDF](https://arxiv.org/pdf/2502.12485)  

**Abstract**: To ensure safe usage, Large Language Models (LLMs) typically undergo alignment with human-defined values. However, this alignment often relies on primarily English data and is biased towards Western-centric values, limiting its effectiveness in low-resource language settings. In this paper, we describe our approach for aligning SEA-Lion-v2.1-Instruct (a Llama3-8B variant) to minimize toxicity in Singlish, an English creole specific to Singapore. We find that supervised fine-tuning and Kahneman-Tversky Optimization (KTO) on paired and unpaired preferences is more sample efficient and yields significantly better results than Direct Preference Optimization (DPO). Our analysis reveals that DPO implicitly enforces a weaker safety objective than KTO, and that SFT complements KTO by improving training stability. Finally, we introduce a simple but novel modification to KTO, KTO-S, which improves training stability through better gradient exploitation. Overall, we present a general approach for safety alignment conducive to low-resource English languages, successfully reducing toxicity by 99\% on our Singlish benchmark, with gains generalizing to the broader TOXIGEN dataset while maintaining strong performance across standard LLM benchmarks. 

**Abstract (ZH)**: 为了确保安全使用，大型语言模型（LLMs）通常会与人类定义的价值观进行对齐。然而，这种对齐往往主要依赖于英文数据，并且倾向于西方价值观，这限制了其在低资源语言环境中的有效性。本文我们介绍了将SEA-Lion-v2.1-Instruct（一种Llama3-8B变体）对齐以最小化Singlish毒性（注：Singlish是新加坡特定的英语方言）的方法。我们发现，监督微调和配对及非配对偏好上的Kahneman-Tversky优化（KTO）在样本效率上更高，并且比直接偏好优化（DPO）的效果显著更好。我们的分析表明，DPO隐式地设置了一个比KTO更为宽松的安全目标，而SFT通过提高训练稳定性来补充KTO。最后，我们引入了一种KTO-S的简单但新颖的变种，通过更好地利用梯度来提高训练稳定性。总体而言，本文提出了一种适用于低资源英语语言的安全对齐方法，在我们的Singlish基准上成功将毒性降低了99%，并且该方法在TOXIGEN数据集上表现出良好的泛化效果，同时在标准LLM基准测试中保持了强劲的表现。 

---
# The Knowledge Microscope: Features as Better Analytical Lenses than Neurons 

**Title (ZH)**: 知识显微镜：特征作为比神经元更优秀的分析透镜 

**Authors**: Yuheng Chen, Pengfei Cao, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.12483)  

**Abstract**: Previous studies primarily utilize MLP neurons as units of analysis for understanding the mechanisms of factual knowledge in Language Models (LMs); however, neurons suffer from polysemanticity, leading to limited knowledge expression and poor interpretability. In this paper, we first conduct preliminary experiments to validate that Sparse Autoencoders (SAE) can effectively decompose neurons into features, which serve as alternative analytical units. With this established, our core findings reveal three key advantages of features over neurons: (1) Features exhibit stronger influence on knowledge expression and superior interpretability. (2) Features demonstrate enhanced monosemanticity, showing distinct activation patterns between related and unrelated facts. (3) Features achieve better privacy protection than neurons, demonstrated through our proposed FeatureEdit method, which significantly outperforms existing neuron-based approaches in erasing privacy-sensitive information from this http URL and dataset will be available. 

**Abstract (ZH)**: 以往的研究主要利用多层感知器（MLP）神经元作为分析单元，以理解语言模型（LMs）中事实知识的机制；然而，神经元存在多义性问题，导致知识表达有限且可解释性差。本文首先进行了初步实验，验证了稀疏自编码器（Sparse Autoencoders, SAE）能够有效地将神经元分解为特征，这些特征作为替代分析单元。在此基础上，我们的核心发现揭示了特征相对于神经元的三大优势：（1）特征在知识表达方面更具影响力，且具有更好的可解释性。（2）特征展示了增强的单义性，在相关事实和无关事实之间展现出不同的激活模式。（3）特征在隐私保护方面表现更好，通过我们提出的FeatureEdit方法得到了验证，该方法在删除敏感隐私信息方面显著优于现有的基于神经元的方法。数据集将在发布论文后提供。 

---
# MSE-Adapter: A Lightweight Plugin Endowing LLMs with the Capability to Perform Multimodal Sentiment Analysis and Emotion Recognition 

**Title (ZH)**: MSE-Adapter：一种轻量级插件，赋予大语言模型进行多模态情感分析和情绪识别的能力 

**Authors**: Yang Yang, Xunde Dong, Yupeng Qiang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12478)  

**Abstract**: Current Multimodal Sentiment Analysis (MSA) and Emotion Recognition in Conversations (ERC) methods based on pre-trained language models exhibit two primary limitations:
1) Once trained for MSA and ERC tasks, these pre-trained language models lose their original generalized capabilities. 2) They demand considerable computational resources. As the size of pre-trained language models continues to grow, training larger multimodal sentiment analysis models using previous approaches could result in unnecessary computational cost. In response to this challenge, we propose \textbf{M}ultimodal \textbf{S}entiment Analysis and \textbf{E}motion Recognition \textbf{Adapter} (MSE-Adapter), a lightweight and adaptable plugin. This plugin enables a large language model (LLM) to carry out MSA or ERC tasks with minimal computational overhead (only introduces approximately 2.6M to 2.8M trainable parameters upon the 6/7B models), while preserving the intrinsic capabilities of the LLM. In the MSE-Adapter, the Text-Guide-Mixer (TGM) module is introduced to establish explicit connections between non-textual and textual modalities through the Hadamard product. This allows non-textual modalities to better align with textual modalities at the feature level, promoting the generation of higher-quality pseudo tokens. Extensive experiments were conducted on four public English and Chinese datasets using consumer-grade GPUs and open-source LLMs (Qwen-1.8B, ChatGLM3-6B-base, and LLaMA2-7B) as the backbone. The results demonstrate the effectiveness of the proposed plugin. The code will be released on GitHub after a blind review. 

**Abstract (ZH)**: 当前基于预训练语言模型的多模态情感分析（MSA）和对话情感识别（ERC）方法存在两个主要限制：
1) 在训练用于MSA和ERC任务后，这些预训练语言模型失去了其原有的泛化能力。
2) 它们需要大量的计算资源。随着预训练语言模型规模的不断扩大，使用之前的方法训练更大的多模态情感分析模型可能会导致不必要的计算成本。为应对这一挑战，我们提出了一种轻量级且可适应的插件：多模态情感分析与情感识别适配器（MSE-Adapter）。该插件能让大型语言模型（LLM）以最小的计算开销执行MSA或ERC任务（在6/7B模型上仅引入约2.6M到2.8M可训练参数），同时保留LLM的内在能力。在MSE-Adapter中，引入了Text-Guide-Mixer (TGM) 模块，通过哈达玛积建立非文本模态与文本模态之间的显式连接。这使得非文本模态在特征层面更好地与文本模态对齐，促进生成高质量的伪令牌。通过使用消费者级GPU和开源LLM（Qwen-1.8B、ChatGLM3-6B-base和LLaMA2-7B）作为骨干，在四个公开的英语和汉语数据集上进行了广泛实验。结果表明所提出插件的有效性。代码将在盲审结束后在GitHub上发布。 

---
# Savaal: Scalable Concept-Driven Question Generation to Enhance Human Learning 

**Title (ZH)**: Savaal：面向概念的可扩展性问题生成以增强人类学习 

**Authors**: Kimia Noorbakhsh, Joseph Chandler, Pantea Karimi, Mohammad Alizadeh, Hari Balakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2502.12477)  

**Abstract**: Assessing and enhancing human learning through question-answering is vital, yet automating this process remains challenging. While large language models (LLMs) excel at summarization and query responses, their ability to generate meaningful questions for learners is underexplored.
We propose Savaal, a scalable question-generation system with three objectives: (i) scalability, enabling question generation from hundreds of pages of text (ii) depth of understanding, producing questions beyond factual recall to test conceptual reasoning, and (iii) domain-independence, automatically generating questions across diverse knowledge areas. Instead of providing an LLM with large documents as context, Savaal improves results with a three-stage processing pipeline. Our evaluation with 76 human experts on 71 papers and PhD dissertations shows that Savaal generates questions that better test depth of understanding by 6.5X for dissertations and 1.5X for papers compared to a direct-prompting LLM baseline. Notably, as document length increases, Savaal's advantages in higher question quality and lower cost become more pronounced. 

**Abstract (ZH)**: 通过问答评估和提升人类学习至关重要，但自动化这一过程仍具有挑战性。尽管大型语言模型（LLMs）在总结和查询响应方面表现优异，但它们生成有意义的问题以促进学习者学习的能力尚未得到充分探索。

我们提出了Savaal，这是一种可扩展的问题生成系统，旨在实现以下三个目标：（i）可扩展性，能够从数百页的文本中生成问题；（ii）深度理解，产出超越事实回忆的问题，以测试概念性推理；（iii）领域独立性，在不同知识领域自动生成问题。Savaal 不是直接将大量文档作为上下文提供给LLM，而是通过一个三阶段处理管道来改进结果。我们的评估结果显示，与直接提示的LLM基线相比，Savaal 在博士论文上的深度理解问题生成上提高了6.5倍，在论文上提高了1.5倍。值得注意的是，随着文档长度的增加，Savaal 在高质量问题生成和成本降低方面的优势愈发显著。 

---
# CoCo-CoLa: Evaluating Language Adherence in Multilingual LLMs 

**Title (ZH)**: CoCo-CoLa：评估多语言大型语言模型的语言规范性 

**Authors**: Elnaz Rahmati, Alireza S. Ziabari, Morteza Dehghani  

**Link**: [PDF](https://arxiv.org/pdf/2502.12476)  

**Abstract**: Multilingual Large Language Models (LLMs) develop cross-lingual abilities despite being trained on limited parallel data. However, they often struggle to generate responses in the intended language, favoring high-resource languages such as English. In this work, we introduce CoCo-CoLa (Correct Concept - Correct Language), a novel metric to evaluate language adherence in multilingual LLMs. Using fine-tuning experiments on a closed-book QA task across seven languages, we analyze how training in one language affects others' performance. Our findings reveal that multilingual models share task knowledge across languages but exhibit biases in the selection of output language. We identify language-specific layers, showing that final layers play a crucial role in determining output language. Accordingly, we propose a partial training strategy that selectively fine-tunes key layers, improving language adherence while significantly reducing computational cost. Our method achieves comparable or superior performance to full fine-tuning, particularly for low-resource languages, offering a more efficient multilingual adaptation. 

**Abstract (ZH)**: 多语言大型语言模型（LLMs）即使在有限的平行数据上进行训练，也能发展出跨语言的能力。然而，它们往往难以在目标语言中生成适当的响应，倾向于使用资源丰富的语言，如英语。在这项工作中，我们引入了CoCo-CoLa（正确概念-正确语言）这一新的评价指标，用于评估多语言LLMs的语言一致性。通过在七种不同语言上的闭卷问答任务中进行微调实验，我们分析了在一种语言上的训练如何影响其他语言的表现。我们的研究发现，多语言模型在语言之间共享任务知识，但在输出语言的选择上表现出偏见。我们确定了语言特有的层，表明最终层在决定输出语言方面发挥着关键作用。据此，我们提出了一个部分训练策略，有选择地微调关键层，从而提高语言一致性并显著降低计算成本。我们的方法在低资源语言中尤其表现优异，提供了更高效的多语言适应方案。 

---
# Reasoning on a Spectrum: Aligning LLMs to System 1 and System 2 Thinking 

**Title (ZH)**: 在连续统一度量上推理：将大语言模型与系统1和系统2思维对齐 

**Authors**: Alireza S. Ziabari, Nona Ghazizadeh, Zhivar Sourati, Farzan Karimi-Malekabadi, Payam Piray, Morteza Dehghani  

**Link**: [PDF](https://arxiv.org/pdf/2502.12470)  

**Abstract**: Large Language Models (LLMs) exhibit impressive reasoning abilities, yet their reliance on structured step-by-step processing reveals a critical limitation. While human cognition fluidly adapts between intuitive, heuristic (System 1) and analytical, deliberative (System 2) reasoning depending on the context, LLMs lack this dynamic flexibility. This rigidity can lead to brittle and unreliable performance when faced with tasks that deviate from their trained patterns. To address this, we create a dataset of 2,000 samples with valid System 1 and System 2 answers, explicitly align LLMs with these reasoning styles, and evaluate their performance across reasoning benchmarks. Our results reveal an accuracy-efficiency trade-off: System 2-aligned models excel in arithmetic and symbolic reasoning, while System 1-aligned models perform better in commonsense tasks. A mechanistic analysis of model responses shows that System 1 models employ more definitive answers, whereas System 2 models demonstrate greater uncertainty. Interpolating between these extremes produces a monotonic transition in reasoning accuracy, preserving coherence. This work challenges the assumption that step-by-step reasoning is always optimal and highlights the need for adapting reasoning strategies based on task demands. 

**Abstract (ZH)**: 大型语言模型（LLMs）展示了令人印象深刻的能力推理能力，但由于其依赖于结构化的逐步处理，这揭示了一个关键的局限性。人类认知能够在不同背景下灵活地在直观的直觉推理（系统1）和分析的审慎推理（系统2）之间转变，而LLMs缺乏这种动态的灵活性。这种刚性可能导致它们在偏离训练模式的任务中出现脆弱且不可靠的表现。为了应对这一问题，我们创建了一个包含2,000个样例的数据集，其中包含有效的系统1和系统2答案，并明确地将LLMs与这些推理风格对齐，评估它们在推理基准测试中的表现。我们的结果显示了一个准确性和效率之间的权衡：系统2对齐的模型在算术和符号推理方面表现出色，而系统1对齐的模型在常识性任务中表现更好。对模型响应的机制分析表明，系统1模型使用更明确的答案，而系统2模型表现出更大的不确定性。在这些极端情况之间的插值产生了一个单调的推理准确性的过渡，同时保持了连贯性。这项工作挑战了逐步推理总是最佳的假设，并强调了根据任务需求调整推理策略的需求。 

---
# SafeRoute: Adaptive Model Selection for Efficient and Accurate Safety Guardrails in Large Language Models 

**Title (ZH)**: SafeRoute：高效的准确的安全护栏的自适应模型选择方法 

**Authors**: Seanie Lee, Dong Bok Lee, Dominik Wagner, Minki Kang, Haebin Seong, Tobias Bocklet, Juho Lee, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12464)  

**Abstract**: Deploying large language models (LLMs) in real-world applications requires robust safety guard models to detect and block harmful user prompts. While large safety guard models achieve strong performance, their computational cost is substantial. To mitigate this, smaller distilled models are used, but they often underperform on "hard" examples where the larger model provides accurate predictions. We observe that many inputs can be reliably handled by the smaller model, while only a small fraction require the larger model's capacity. Motivated by this, we propose SafeRoute, a binary router that distinguishes hard examples from easy ones. Our method selectively applies the larger safety guard model to the data that the router considers hard, improving efficiency while maintaining accuracy compared to solely using the larger safety guard model. Experimental results on multiple benchmark datasets demonstrate that our adaptive model selection significantly enhances the trade-off between computational cost and safety performance, outperforming relevant baselines. 

**Abstract (ZH)**: 在实际应用中部署大型语言模型（LLMs）需要具备 robust 安全防护模型来检测并阻止有害的用户提示。虽然大型安全防护模型能够取得出色的性能，但其计算成本也是相当巨大的。为降低这种成本，可以使用较小的蒸馏模型，但这些模型在处理“困难”示例时往往表现不佳，而这些示例是大型模型能够准确预测的。我们观察到，许多输入可以由较小的模型可靠地处理，而只有少数输入需要使用大型模型的能力。基于这一发现，我们提出了 SafeRoute，一种二元路由器，用于区分困难示例和简单示例。我们的方法仅在路由器认为困难的数据上应用较大的安全防护模型，从而提高效率并保持与仅使用大型安全防护模型相当的准确性。在多个基准数据集上的实验结果表明，我们提出的自适应模型选择方法显著提高了计算成本和安全性能之间的权衡，优于相关基线方法。 

---
# Emulating Retrieval Augmented Generation via Prompt Engineering for Enhanced Long Context Comprehension in LLMs 

**Title (ZH)**: 通过提示工程模拟检索增强生成，以增强大型语言模型的长上下文理解 

**Authors**: Joon Park, Kyohei Atarashi, Koh Takeuchi, Hisashi Kashima  

**Link**: [PDF](https://arxiv.org/pdf/2502.12462)  

**Abstract**: This paper addresses the challenge of comprehending very long contexts in Large Language Models (LLMs) by proposing a method that emulates Retrieval Augmented Generation (RAG) through specialized prompt engineering and chain-of-thought (CoT) reasoning. While recent LLMs support over 100,000 tokens in a single prompt, simply enlarging context windows has not guaranteed robust multi-hop reasoning when key details are scattered across massive input. Our approach treats the model as both the retriever and the reasoner: it first tags relevant segments within a long passage, then employs a stepwise CoT workflow to integrate these pieces of evidence. This single-pass method thereby reduces reliance on an external retriever, yet maintains focus on crucial segments. We evaluate our approach on selected tasks from BABILong, which interleaves standard bAbI QA problems with large amounts of distractor text. Compared to baseline (no retrieval) and naive RAG pipelines, our approach more accurately handles multi-fact questions such as object location tracking, counting, and indefinite knowledge. Furthermore, we analyze how prompt structure, including the order of question, relevant-text tags, and overall instructions, significantly affects performance. These findings underscore that optimized prompt engineering, combined with guided reasoning, can enhance LLMs' long-context comprehension and serve as a lightweight alternative to traditional retrieval pipelines. 

**Abstract (ZH)**: 本文通过提出一种通过专门的提示工程和链式推理（CoT）方式模拟检索增强生成（RAG）的方法，以解决大型语言模型（LLMs）在理解非常长上下文时的挑战。尽管最近的LLMs在单一提示中支持超过10万个令牌，但简单地扩大上下文窗口并未确保在关键细节分散在大量输入中时进行稳健的多跳推理。我们的方法将模型视为检索器和推理器：首先，它在长段落中标记出相关的部分，然后使用逐步的CoT工作流来整合这些证据。这种方法通过单次过程减少了对外部检索器的依赖，同时仍能聚焦于关键部分。我们通过对BABILong选择的测试任务进行评估，其中这些任务将标准的bAbI问答问题与大量的干扰文本交织在一起。与基准方法（没有检索）和简单的RAG流水线相比，我们的方法更准确地处理了多事实问题，如对象位置跟踪、计数和不确定知识。此外，我们分析了提示结构，包括问题的顺序、相关文本标记和总体指令如何显著影响性能。这些发现表明，优化的提示工程与引导式推理相结合，可以增强LLMs在长上下文理解方面的能力，并作为传统的检索流水线的一种轻量级替代方案。 

---
# Stress Testing Generalization: How Minor Modifications Undermine Large Language Model Performance 

**Title (ZH)**: 泛化能力的压力测试：细微修改如何削弱大型语言模型的性能 

**Authors**: Guangxiang Zhao, Saier Hu, Xiaoqi Jian, Jinzhu Wu, Yuhan Wu, Change Jia, Lin Sun, Xiangzheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12459)  

**Abstract**: This paper investigates the fragility of Large Language Models (LLMs) in generalizing to novel inputs, specifically focusing on minor perturbations in well-established benchmarks (e.g., slight changes in question format or distractor length). Despite high benchmark scores, LLMs exhibit significant accuracy drops and unexpected biases (e.g., preference for longer distractors) when faced with these minor but content-preserving modifications. For example, Qwen 2.5 1.5B's MMLU score rises from 60 to 89 and drops from 89 to 36 when option lengths are changed without altering the question. Even GPT-4 experiences a 25-point accuracy loss when question types are changed, with a 6-point drop across all three modification categories. These analyses suggest that LLMs rely heavily on superficial cues rather than forming robust, abstract representations that generalize across formats, lexical variations, and irrelevant content shifts. This work aligns with the ACL 2025 theme track on the Generalization of NLP models, proposing a "Generalization Stress Test" to assess performance shifts under controlled perturbations. The study calls for reevaluating benchmarks and developing more reliable evaluation methodologies to capture LLM generalization abilities better. 

**Abstract (ZH)**: 本文探讨了大型语言模型（LLMs）在泛化到新输入时的脆弱性，具体关注于在经过广泛认可的基准测试中进行的细微变动（例如问题格式的微小变化或干扰项长度的变化）。尽管基准测试得分很高，但当面对这些细微但内容保留的修改时，LLMs会表现出显著的准确性下降和意想不到的偏差（例如对更长的干扰项有偏好）。例如，Qwen 2.5 1.5B的MMLU分数在选项长度不变的情况下从60上升到89，然后又从89跌至36。即使GPT-4在问题类型发生变化时也经历了25个准确点的下降，这种下降在三个修改类别中都出现了6个点。这些分析表明，LLMs过分依赖于表面上的线索，而不是形成能够跨格式、词汇变化和不相关信息转移泛化的稳健、抽象表征。这项工作与ACL 2025主题研讨会上的语言处理模型泛化主题相契合，提出了“泛化应力测试”来评估在可控扰动下的性能变化。研究呼吁重新评估基准测试，并开发更可靠的方法来更好地捕捉LLM的泛化能力。 

---
# An Empirical Evaluation of Encoder Architectures for Fast Real-Time Long Conversational Understanding 

**Title (ZH)**: 对快速实时长对话理解中编码器架构的实证评估 

**Authors**: Annamalai Senthilnathan, Kristjan Arumae, Mohammed Khalilia, Zhengzheng Xing, Aaron R. Colak  

**Link**: [PDF](https://arxiv.org/pdf/2502.12458)  

**Abstract**: Analyzing long text data such as customer call transcripts is a cost-intensive and tedious task. Machine learning methods, namely Transformers, are leveraged to model agent-customer interactions. Unfortunately, Transformers adhere to fixed-length architectures and their self-attention mechanism scales quadratically with input length. Such limitations make it challenging to leverage traditional Transformers for long sequence tasks, such as conversational understanding, especially in real-time use cases. In this paper we explore and evaluate recently proposed efficient Transformer variants (e.g. Performer, Reformer) and a CNN-based architecture for real-time and near real-time long conversational understanding tasks. We show that CNN-based models are dynamic, ~2.6x faster to train, ~80% faster inference and ~72% more memory efficient compared to Transformers on average. Additionally, we evaluate the CNN model using the Long Range Arena benchmark to demonstrate competitiveness in general long document analysis. 

**Abstract (ZH)**: 分析客户通话记录等长文本数据是一项成本高昂且繁琐的任务。利用机器学习方法，特别是变换器（Transformers），可以建模坐席与客户之间的互动。然而，变换器依赖于固定长度的架构，并且其自我注意力机制随着输入长度的增加而呈二次增长。这些限制使得在长序列任务（如对话理解）中使用传统的变换器变得困难，尤其是在实时场景中。在本文中，我们探讨并评估了最近提出的高效变换器变体（例如 Performer、Reformer）以及基于卷积神经网络（CNN）的架构，以应对实时和近乎实时的长对话理解任务。结果显示，在平均情况下，基于CNN的模型训练速度比变换器快约2.6倍，推理速度比变换器快约80%，并且内存效率高出约72%。此外，我们使用Long Range Arena基准测试了CNN模型，以展示其在一般长文档分析中的竞争力。 

---
# DSMoE: Matrix-Partitioned Experts with Dynamic Routing for Computation-Efficient Dense LLMs 

**Title (ZH)**: DSMoE：用于计算高效的密集大语言模型的矩阵分割专家与动态路由方法 

**Authors**: Minxuan Lv, Zhenpeng Su, Leiyu Pan, Yizhe Xiong, Zijia Lin, Hui Chen, Wei Zhou, Jungong Han, Guiguang Ding, Cheng Luo, Di Zhang, Kun Gai, Songlin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12455)  

**Abstract**: As large language models continue to scale, computational costs and resource consumption have emerged as significant challenges. While existing sparsification methods like pruning reduce computational overhead, they risk losing model knowledge through parameter removal. This paper proposes DSMoE (Dynamic Sparse Mixture-of-Experts), a novel approach that achieves sparsification by partitioning pre-trained FFN layers into computational blocks. We implement adaptive expert routing using sigmoid activation and straight-through estimators, enabling tokens to flexibly access different aspects of model knowledge based on input complexity. Additionally, we introduce a sparsity loss term to balance performance and computational efficiency. Extensive experiments on LLaMA models demonstrate that under equivalent computational constraints, DSMoE achieves superior performance compared to existing pruning and MoE approaches across language modeling and downstream tasks, particularly excelling in generation tasks. Analysis reveals that DSMoE learns distinctive layerwise activation patterns, providing new insights for future MoE architecture design. 

**Abstract (ZH)**: 随着大型语言模型的不断扩展，计算成本和资源消耗已成为重大挑战。尽管现有的稀疏化方法如剪枝能够降低计算开销，但可能会因参数删除而丢失模型知识。本文提出了一种名为DSMoE（Dynamic Sparse Mixture-of-Experts）的创新方法，通过将预训练的FFN层划分为计算块来实现稀疏化。我们使用Sigmoid激活和直通估计器实现了自适应专家路由，使标记能够根据输入复杂度灵活访问模型知识的不同方面。此外，我们引入了稀疏化损失项以平衡性能和计算效率。在LLaMA模型上进行的广泛实验表明，在同等计算约束条件下，DSMoE在语言建模和下游任务中均实现了优于现有剪枝和MoE方法的性能，特别是在生成任务中表现出色。分析表明，DSMoE学习到了独特的逐层激活模式，为未来MoE架构设计提供了新的见解。 

---
# Multi-Attribute Steering of Language Models via Targeted Intervention 

**Title (ZH)**: 通过目标干预实现语言模型的多属性调控 

**Authors**: Duy Nguyen, Archiki Prasad, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2502.12446)  

**Abstract**: Inference-time intervention (ITI) has emerged as a promising method for steering large language model (LLM) behavior in a particular direction (e.g., improving helpfulness) by intervening on token representations without costly updates to the LLM's parameters. However, existing ITI approaches fail to scale to multi-attribute settings with conflicts, such as enhancing helpfulness while also reducing toxicity. To address this, we introduce Multi-Attribute Targeted Steering (MAT-Steer), a novel steering framework designed for selective token-level intervention across multiple attributes. MAT-Steer learns steering vectors using an alignment objective that shifts the model's internal representations of undesirable outputs closer to those of desirable ones while enforcing sparsity and orthogonality among vectors for different attributes, thereby reducing inter-attribute conflicts. We evaluate MAT-Steer in two distinct settings: (i) on question answering (QA) tasks where we balance attributes like truthfulness, bias, and toxicity; (ii) on generative tasks where we simultaneously improve attributes like helpfulness, correctness, and coherence. MAT-Steer outperforms existing ITI and parameter-efficient finetuning approaches across both task types (e.g., 3% average accuracy gain across QA tasks and 55.82% win rate against the best ITI baseline). 

**Abstract (ZH)**: 推理时干预（ITI）已成为一种有前景的方法，通过在大型语言模型（LLM）的token表示上进行干预，而不必进行昂贵的参数更新，从而引导模型行为朝着特定方向发展（例如，提高有用性）。然而，现有的ITI方法无法应对多属性设置中的冲突（如提高有用性的同时减少毒性），无法适用于此类场景。为解决这一问题，我们引入了多属性目标导向引导（MAT-Steer）——一种专为跨多个属性进行选择性token级干预设计的新颖引导框架。MAT-Steer采用对齐目标来使模型对不 desirable 输出的内部表示向 desirable 输出的表示靠拢，同时确保不同属性向量的稀疏性和正交性，从而减少属性间的冲突。我们在两种不同的设置中评估了MAT-Steer：(i) 在问答（QA）任务上，平衡诸如准确性、偏见和毒性等属性；(ii) 在生成任务中，同时提升诸如有用性、正确性和连贯性等属性。MAT-Steer在两种任务类型中的表现均优于现有的ITI和参数有效微调方法（例如，在问答任务中平均准确率提高了3%，在与最好的ITI基线对比中胜出率为55.82%）。 

---
# Should I Trust You? Detecting Deception in Negotiations using Counterfactual RL 

**Title (ZH)**: 《我应该相信你吗？基于反事实RL的谈判中的欺骗检测》 

**Authors**: Wichayaporn Wongkamjan, Yanze Wang, Feng Gu, Denis Peskoff, Jonathan K. Kummerfeld, Jonathan May, Jordan Lee Boyd-Graber  

**Link**: [PDF](https://arxiv.org/pdf/2502.12436)  

**Abstract**: An increasingly prevalent socio-technical problem is people being taken in by offers that sound ``too good to be true'', where persuasion and trust shape decision-making. This paper investigates how \abr{ai} can help detect these deceptive scenarios. We analyze how humans strategically deceive each other in \textit{Diplomacy}, a board game that requires both natural language communication and strategic reasoning. This requires extracting logical forms of proposed agreements in player communications and computing the relative rewards of the proposal using agents' value functions. Combined with text-based features, this can improve our deception detection. Our method detects human deception with a high precision when compared to a Large Language Model approach that flags many true messages as deceptive. Future human-\abr{ai} interaction tools can build on our methods for deception detection by triggering \textit{friction} to give users a chance of interrogating suspicious proposals. 

**Abstract (ZH)**: 一项日益普遍的社技问题是人们容易被听起来“太好而不真实”的提议所迷惑，这类情况中的说服与信任影响了决策过程。本文探讨了人工智能如何帮助检测这些欺骗性场景。我们分析了在《外交》这款需要自然语言交流和策略推理的棋盘游戏中，人类如何战略性地相互欺骗。这要求提取玩家通信中提议协议的逻辑形式，并利用智能体的价值函数计算提议的相对收益。结合文本特征，这种方法可以提高我们的欺骗检测能力。我们的方法在检测人类欺骗时的精度高于一种大型语言模型的方法，该方法将许多真实的信息错误地标记为欺骗性信息。未来的人机交互工具可以借鉴我们的欺骗检测方法，在提出诱人的但可疑的提议时引入摩擦，给用户提供审查这些提议的机会。 

---
# Wi-Chat: Large Language Model Powered Wi-Fi Sensing 

**Title (ZH)**: Wi-Chat：由大规模语言模型驱动的Wi-Fi感知 

**Authors**: Haopeng Zhang, Yili Ren, Haohan Yuan, Jingzhe Zhang, Yitong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2502.12421)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse tasks. However, their potential to integrate physical model knowledge for real-world signal interpretation remains largely unexplored. In this work, we introduce Wi-Chat, the first LLM-powered Wi-Fi-based human activity recognition system. We demonstrate that LLMs can process raw Wi-Fi signals and infer human activities by incorporating Wi-Fi sensing principles into prompts. Our approach leverages physical model insights to guide LLMs in interpreting Channel State Information (CSI) data without traditional signal processing techniques. Through experiments on real-world Wi-Fi datasets, we show that LLMs exhibit strong reasoning capabilities, achieving zero-shot activity recognition. These findings highlight a new paradigm for Wi-Fi sensing, expanding LLM applications beyond conventional language tasks and enhancing the accessibility of wireless sensing for real-world deployments. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）在各种任务中展示了非凡的能力。然而，它们将物理模型知识整合到现实世界的信号解释中的潜力尚未得到充分探索。在此项工作中，我们引入了Wi-Chat，这是第一个基于LLM的Wi-Fi人体活动识别系统。我们证明了LLM能够处理原始的Wi-Fi信号，并通过将Wi-Fi传感原理集成到提示中来推断人类活动。我们的方法利用物理模型的洞察力，指导LLMs解释.Channel State Information (CSI)数据，而无需传统信号处理技术。通过在实际Wi-Fi数据集上的实验，我们展示了LLM表现出强大的推理能力，实现了零样本活动识别。这些发现突出了一种新的Wi-Fi传感范式，扩展了LLM的应用范围，使其不再局限于传统的语言任务，并增强了无线传感在实际部署中的可用性。 

---
# Sens-Merging: Sensitivity-Guided Parameter Balancing for Merging Large Language Models 

**Title (ZH)**: 感度融合：基于感度指导的大型语言模型参数平衡方法 

**Authors**: Shuqi Liu, Han Wu, Bowei He, Xiongwei Han, Mingxuan Yuan, Linqin Song  

**Link**: [PDF](https://arxiv.org/pdf/2502.12420)  

**Abstract**: Recent advances in large language models have led to numerous task-specialized fine-tuned variants, creating a need for efficient model merging techniques that preserve specialized capabilities while avoiding costly retraining. While existing task vector-based merging methods show promise, they typically apply uniform coefficients across all parameters, overlooking varying parameter importance both within and across tasks. We present Sens-Merging, a sensitivity-guided coefficient adjustment method that enhances existing model merging techniques by operating at both task-specific and cross-task levels. Our method analyzes parameter sensitivity within individual tasks and evaluates cross-task transferability to determine optimal merging coefficients. Extensive experiments on Mistral 7B and LLaMA2-7B/13B models demonstrate that Sens-Merging significantly improves performance across general knowledge, mathematical reasoning, and code generation tasks. Notably, when combined with existing merging techniques, our method enables merged models to outperform specialized fine-tuned models, particularly in code generation tasks. Our findings reveal important trade-offs between task-specific and cross-task scalings, providing insights for future model merging strategies. 

**Abstract (ZH)**: 近年来，大型语言模型的发展催生了大量专门化微调变体，这需要有效的模型合并技术来保留特定化的能力，同时避免昂贵的重新训练过程。虽然现有的任务向量基合并方法展示出潜力，但它们通常在所有参数上应用统一的系数，忽视了任务内和任务间参数的重要性差异。我们提出了Sens-Merging，这是一种基于灵敏度的系数调整方法，该方法通过任务特定和跨任务两个层面增强了现有的模型合并技术。我们的方法分析了单个任务中的参数灵敏度，并评估跨任务的可迁移性以确定最优的合并系数。在Mistral 7B和LLaMA2-7B/13B模型上的广泛实验表明，Sens-Merging在一般知识、数学推理和代码生成任务中显著增强了性能。值得注意的是，当与现有的合并技术结合时，我们的方法使得合并模型在代码生成任务中优于专门微调模型。我们的研究结果揭示了任务特定和跨任务缩放之间的重要权衡，为未来的模型合并策略提供了见解。 

---
# Lost in Transcription, Found in Distribution Shift: Demystifying Hallucination in Speech Foundation Models 

**Title (ZH)**: 在转录中迷失，在分布偏移中发现：揭秘语音基础模型中的幻觉现象 

**Authors**: Hanin Atwany, Abdul Waheed, Rita Singh, Monojit Choudhury, Bhiksha Raj  

**Link**: [PDF](https://arxiv.org/pdf/2502.12414)  

**Abstract**: Speech foundation models trained at a massive scale, both in terms of model and data size, result in robust systems capable of performing multiple speech tasks, including automatic speech recognition (ASR). These models transcend language and domain barriers, yet effectively measuring their performance remains a challenge. Traditional metrics like word error rate (WER) and character error rate (CER) are commonly used to evaluate ASR performance but often fail to reflect transcription quality in critical contexts, particularly when detecting fabricated outputs. This phenomenon, known as hallucination, is especially concerning in high-stakes domains such as healthcare, legal, and aviation, where errors can have severe consequences. In our work, we address this gap by investigating hallucination in ASR models. We examine how factors such as distribution shifts, model size, and model architecture influence the hallucination error rate (HER), a metric we introduce to quantify hallucinations. Our analysis of 20 ASR models reveals \numinsights~key insights: (1) High WERs can mask low hallucination rates, while low WERs may conceal dangerous hallucinations. (2) Synthetic noise, both adversarial and common perturbations like white noise, pitch shift, and time stretching, increase HER. (3) Distribution shift correlates strongly with HER ($\alpha = 0.91$). Our findings highlight the importance of incorporating HER alongside traditional metrics like WER to better assess ASR model performance, particularly in high-stakes domains. 

**Abstract (ZH)**: 大规模训练的语音基础模型，无论是模型规模还是数据规模，都能够构建出稳健的系统，具备执行多种语音任务的能力，包括自动语音识别（ASR）。这些模型能够跨越语言和领域的障碍，但在衡量其性能方面仍然面临挑战。传统的评估指标如字错误率（WER）和字符错误率（CER）常被用来评估ASR性能，但在关键语境下，这些指标往往无法反映转录质量，尤其是在检测伪造输出时。这种现象被称为“幻觉”，在医疗、法律和航空等高风险领域尤为令人担忧，这些领域中的错误可能导致严重的后果。在我们的工作中，我们通过研究ASR模型中的幻觉现象来弥补这一缺口。我们探讨了诸如分布偏移、模型规模和模型架构等因素如何影响幻觉错误率（HER），这是一种我们新引入的用于量化幻觉的指标。对20个ASR模型的分析揭示了以下关键见解：（1）高WER可能掩盖低幻觉率，而低WER可能掩盖危险的幻觉。（2）合成噪声，包括对抗性噪声和常见扰动（如白噪声、音调偏移和时间拉伸），会增加HER。（3）分布偏移与HER高度相关（相关系数α=0.91）。我们的研究结果强调了在高风险领域中，除了传统的评估指标如WER以外，还应纳入HER来更好地评估ASR模型的性能的重要性。 

---
# Gradient Co-occurrence Analysis for Detecting Unsafe Prompts in Large Language Models 

**Title (ZH)**: 大型语言模型中检测不安全提示的梯度共现分析 

**Authors**: Jingyuan Yang, Bowen Yan, Rongjun Li, Ziyu Zhou, Xin Chen, Zhiyong Feng, Wei Peng  

**Link**: [PDF](https://arxiv.org/pdf/2502.12411)  

**Abstract**: Unsafe prompts pose significant safety risks to large language models (LLMs). Existing methods for detecting unsafe prompts rely on data-driven fine-tuning to train guardrail models, necessitating significant data and computational resources. In contrast, recent few-shot gradient-based methods emerge, requiring only few safe and unsafe reference prompts. A gradient-based approach identifies unsafe prompts by analyzing consistent patterns of the gradients of safety-critical parameters in LLMs. Although effective, its restriction to directional similarity (cosine similarity) introduces ``directional bias'', limiting its capability to identify unsafe prompts. To overcome this limitation, we introduce GradCoo, a novel gradient co-occurrence analysis method that expands the scope of safety-critical parameter identification to include unsigned gradient similarity, thereby reducing the impact of ``directional bias'' and enhancing the accuracy of unsafe prompt detection. Comprehensive experiments on the widely-used benchmark datasets ToxicChat and XStest demonstrate that our proposed method can achieve state-of-the-art (SOTA) performance compared to existing methods. Moreover, we confirm the generalizability of GradCoo in detecting unsafe prompts across a range of LLM base models with various sizes and origins. 

**Abstract (ZH)**: 不安全的提示对大型语言模型（LLMs）构成了显著的安全风险。现有的不安全提示检测方法依赖于数据驱动的微调来训练防护模型，这需要大量的数据和计算资源。相比之下，近期出现的少量示例梯度基于方法仅需少量的安全和不安全示例提示。梯度基于的方法通过分析大型语言模型中安全性关键参数梯度的一致模式来识别不安全的提示。尽管这种方法非常有效，但其局限于方向相似性（余弦相似性）的限制引入了“方向偏差”，限制了其对不安全提示的识别能力。为克服这一限制，我们提出了一种名为GradCoo的新颖梯度共现分析方法，它扩展了对安全性关键参数识别的范围，将其扩展到包含无符号梯度相似性，从而减少了“方向偏差”的影响并提高了不安全提示检测的准确性。在广泛使用的基准数据集ToxicChat和XStest上进行的全面实验表明，我们提出的方法在现有方法中达到了最先进的（SOTA）性能。此外，我们确认GradCoo在各种规模和来源的大规模语言模型基模中检测不安全提示的有效性。 

---
# On the Robust Approximation of ASR Metrics 

**Title (ZH)**: 关于ASR指标的稳健近似方法 

**Authors**: Abdul Waheed, Hanin Atwany, Rita Singh, Bhiksha Raj  

**Link**: [PDF](https://arxiv.org/pdf/2502.12408)  

**Abstract**: Recent advances in speech foundation models are largely driven by scaling both model size and data, enabling them to perform a wide range of tasks, including speech recognition. Traditionally, ASR models are evaluated using metrics like Word Error Rate (WER) and Character Error Rate (CER), which depend on ground truth labels. As a result of limited labeled data from diverse domains and testing conditions, the true generalization capabilities of these models beyond standard benchmarks remain unclear. Moreover, labeling data is both costly and time-consuming. To address this, we propose a novel label-free approach for approximating ASR performance metrics, eliminating the need for ground truth labels. Our method utilizes multimodal embeddings in a unified space for speech and transcription representations, combined with a high-quality proxy model to compute proxy metrics. These features are used to train a regression model to predict key ASR metrics like Word Error Rate (WER) and Character Error Rate (CER). We experiment with over 40 models across 14 datasets representing both standard and in-the-wild testing conditions. Our results show that we approximate the metrics within a single-digit absolute difference across all experimental configurations, outperforming the most recent baseline by more than 50\%. 

**Abstract (ZH)**: 近年来，语音基础模型的发展主要得益于模型规模和数据量的扩大，使它们能够执行广泛的任务，包括语音识别。传统上，语音识别（ASR）模型的评估主要使用Word Error Rate (WER)和Character Error Rate (CER)等指标，这些指标依赖于准确的标签。由于各种领域和测试条件下的标注数据有限，这些模型在标准基准之外的真正泛化能力仍不明确。此外，获取和标注数据既昂贵又耗时。为解决这一问题，我们提出了一种新的无标签方法来近似ASR性能指标，从而无需使用准确的标签。我们的方法在统一的空间中使用多模态嵌入表示语音和转写内容，并结合高质量的代理模型来计算代理指标。这些特征用于训练回归模型，以预测关键的ASR指标，如Word Error Rate (WER)和Character Error Rate (CER)。我们在14个数据集上进行了超过40个模型的实验，涵盖了标准和野外的测试条件。结果显示，在所有实验配置下，我们的方法在近似这些指标方面的偏差仅为个位数的绝对差异，并且在最近的基线之上取得了超过50%的改进。 

---
# WMT24++: Expanding the Language Coverage of WMT24 to 55 Languages & Dialects 

**Title (ZH)**: WMT24++: 扩展WMT24的语言覆盖范围至55种语言和地区方言 

**Authors**: Daniel Deutsch, Eleftheria Briakou, Isaac Caswell, Mara Finkelstein, Rebecca Galor, Juraj Juraska, Geza Kovacs, Alison Lui, Ricardo Rei, Jason Riesa, Shruti Rijhwani, Parker Riley, Elizabeth Salesky, Firas Trabelsi, Stephanie Winkler, Biao Zhang, Markus Freitag  

**Link**: [PDF](https://arxiv.org/pdf/2502.12404)  

**Abstract**: As large language models (LLM) become more and more capable in languages other than English, it is important to collect benchmark datasets in order to evaluate their multilingual performance, including on tasks like machine translation (MT). In this work, we extend the WMT24 dataset to cover 55 languages by collecting new human-written references and post-edits for 46 new languages and dialects in addition to post-edits of the references in 8 out of 9 languages in the original WMT24 dataset. The dataset covers four domains: literary, news, social, and speech. We benchmark a variety of MT providers and LLMs on the collected dataset using automatic metrics and find that LLMs are the best-performing MT systems in all 55 languages. These results should be confirmed using a human-based evaluation, which we leave for future work. 

**Abstract (ZH)**: 随着大型语言模型（LLM）在除英语之外的语言方面的能力不断增强，收集多语言基准数据集变得越来越重要，以便评估其在机器翻译（MT）等任务上的多语言性能。在本工作中，我们扩展了WMT24数据集，通过收集46种新语言和方言（以及原始WMT24数据集中9种语言中的8种语言的参考后编辑）的新手写参考和后编辑，涵盖了55种语言。数据集包含了四个领域的内容：文学、新闻、社交和口语。我们使用自动评估指标，在收集的数据集上对各种MT提供商和LLM进行了基准测试，并发现LLM在所有55种语言中都是表现最佳的MT系统。这些结果应该通过基于人工的评估来确认，这是我们未来工作留下的任务。 

---
# Pragmatics in the Era of Large Language Models: A Survey on Datasets, Evaluation, Opportunities and Challenges 

**Title (ZH)**: 大型语言模型时代的话语 pragmatics：数据集、评估、机遇与挑战综述 

**Authors**: Bolei Ma, Yuting Li, Wei Zhou, Ziwei Gong, Yang Janet Liu, Katja Jasinskaja, Annemarie Friedrich, Julia Hirschberg, Frauke Kreuter, Barbara Plank  

**Link**: [PDF](https://arxiv.org/pdf/2502.12378)  

**Abstract**: Understanding pragmatics-the use of language in context-is crucial for developing NLP systems capable of interpreting nuanced language use. Despite recent advances in language technologies, including large language models, evaluating their ability to handle pragmatic phenomena such as implicatures and references remains challenging. To advance pragmatic abilities in models, it is essential to understand current evaluation trends and identify existing limitations. In this survey, we provide a comprehensive review of resources designed for evaluating pragmatic capabilities in NLP, categorizing datasets by the pragmatics phenomena they address. We analyze task designs, data collection methods, evaluation approaches, and their relevance to real-world applications. By examining these resources in the context of modern language models, we highlight emerging trends, challenges, and gaps in existing benchmarks. Our survey aims to clarify the landscape of pragmatic evaluation and guide the development of more comprehensive and targeted benchmarks, ultimately contributing to more nuanced and context-aware NLP models. 

**Abstract (ZH)**: 理解语用学——即在特定语境中使用语言——对开发能够解释复杂语言使用的NLP系统至关重要。尽管在语言技术方面取得了近期进展，包括大型语言模型，但评估它们处理语用现象（如含蓄意义和指代）的能力仍然是一个挑战。要提高模型的语用能力，理解现有的评估趋势并识别现有局限性是至关重要的。在这篇综述中，我们提供了一篇全面的评估资源综述，这些资源旨在评估NLP中的语用能力，并按它们所解决的语用现象对数据集进行分类。我们分析了任务设计、数据收集方法、评估方法及其对实际应用的相关性。通过在现代语言模型的背景下检查这些资源，我们突出了现有基准中的新兴趋势、挑战和空白。我们的综述旨在阐明语用评估的现状，并指导开发更全面、更具针对性的基准，最终有助于开发更具有语境意识的NLP模型。 

---
# UltraGen: Extremely Fine-grained Controllable Generation via Attribute Reconstruction and Global Preference Optimization 

**Title (ZH)**: UltraGen：通过属性重构和全局偏好优化实现的极细粒度可控生成 

**Authors**: Longfei Yun, Letian Peng, Jingbo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12375)  

**Abstract**: Fine granularity is an essential requirement for controllable text generation, which has seen rapid growth with the ability of LLMs. However, existing methods focus mainly on a small set of attributes like 3 to 5, and their performance degrades significantly when the number of attributes increases to the next order of magnitude. To address this challenge, we propose a novel zero-shot approach for extremely fine-grained controllable generation (EFCG), proposing auto-reconstruction (AR) and global preference optimization (GPO). In the AR phase, we leverage LLMs to extract soft attributes (e.g., Emphasis on simplicity and minimalism in design) from raw texts, and combine them with programmatically derived hard attributes (e.g., The text should be between 300 and 400 words) to construct massive (around 45) multi-attribute requirements, which guide the fine-grained text reconstruction process under weak supervision. In the GPO phase, we apply direct preference optimization (DPO) to refine text generation under diverse attribute combinations, enabling efficient exploration of the global combination space. Additionally, we introduce an efficient attribute sampling strategy to identify and correct potentially erroneous attributes, further improving global optimization. Our framework significantly improves the constraint satisfaction rate (CSR) and text quality for EFCG by mitigating position bias and alleviating attention dilution. 

**Abstract (ZH)**: 精细粒度是可控文本生成的基本要求，随着大规模语言模型（LLMs）能力的提升，这一领域正迅速发展。然而，现有的方法主要集中在少量属性上，如3到5个属性，当属性数量增加到下一个数量级时，其性能会显著下降。为解决这一挑战，我们提出了一种新颖的零样本方法，用于极精细粒度的可控生成（EFCG），并提出了自动重构（AR）和全局偏好优化（GPO）。在AR阶段，我们利用LLMs从原始文本中提取软属性（例如，设计中注重简洁和极简主义），并将其与通过程序获取的硬属性（例如，文本应在300到400词之间）结合，构建庞大的（约45个）多属性需求，这些需求在弱监督下指导精细文本重构过程。在GPO阶段，我们应用直接偏好优化（DPO）来在多种属性组合下细化文本生成，从而有效地探索全局组合空间。此外，我们引入了一种高效属性采样策略，以识别并纠正可能的错误属性，进一步改善全局优化。我们的框架通过减轻位置偏差和缓解注意力稀释，显著提高了EFCG的约束满足率（CSR）和文本质量。 

---
# Factual Inconsistency in Data-to-Text Generation Scales Exponentially with LLM Size: A Statistical Validation 

**Title (ZH)**: 数据到文本生成中事实不一致性随大语言模型规模呈指数增长：一项统计验证 

**Authors**: Joy Mahapatra, Soumyajit Roy, Utpal Garain  

**Link**: [PDF](https://arxiv.org/pdf/2502.12372)  

**Abstract**: Monitoring factual inconsistency is essential for ensuring trustworthiness in data-to-text generation (D2T). While large language models (LLMs) have demonstrated exceptional performance across various D2T tasks, previous studies on scaling laws have primarily focused on generalization error through power law scaling to LLM size (i.e., the number of model parameters). However, no research has examined the impact of LLM size on factual inconsistency in D2T. In this paper, we investigate how factual inconsistency in D2T scales with LLM size by exploring two scaling laws: power law and exponential scaling. To rigorously evaluate and compare these scaling laws, we employ a statistical validation framework consisting of three key stages: predictive performance estimation, goodness-of-fit assessment, and comparative analysis. For a comprehensive empirical study, we analyze three popular LLM families across five D2T datasets, measuring factual inconsistency inversely using four state-of-the-art consistency metrics. Our findings, based on exhaustive empirical results and validated through our framework, reveal that, contrary to the widely assumed power law scaling, factual inconsistency in D2T follows an exponential scaling with LLM size. 

**Abstract (ZH)**: 确保数据到文本生成（D2T）的可信度至关重要，需要对事实不一致进行监测。虽然大规模语言模型（LLMs）在各种D2T任务中展现了出色的表现，但前期关于缩放规律的研究主要集中在通过幂律缩放模型大小（即模型参数数量）来评估泛化误差。然而，尚未有研究探讨模型大小对D2T中事实不一致的影响。在这篇论文中，我们通过探索两种缩放规律——幂律和指数缩放——来研究事实不一致如何随模型大小缩放。为了严谨地评估和比较这些缩放规律，我们采用了一种包含三个关键阶段的统计验证框架：预测性能估计、拟合优度评估和比较分析。为了进行全面的经验研究，我们在五组D2T数据集上分析了三组流行的LLM家族，使用四种最先进的一致性度量指标从反向测量事实不一致性。基于详尽的经验结果并通过我们的框架进行验证，我们的研究发现，与广泛假设的幂律缩放不同，D2T中的事实不一致性实际上遵循指数缩放规律。 

---
# Classifiers of Data Sharing Statements in Clinical Trial Records 

**Title (ZH)**: 临床试验记录中数据共享声明分类器的研究 

**Authors**: Saber Jelodari Mamaghani, Cosima Strantz, Dennis Toddenroth  

**Link**: [PDF](https://arxiv.org/pdf/2502.12362)  

**Abstract**: Digital individual participant data (IPD) from clinical trials are increasingly distributed for potential scientific reuse. The identification of available IPD, however, requires interpretations of textual data-sharing statements (DSS) in large databases. Recent advancements in computational linguistics include pre-trained language models that promise to simplify the implementation of effective classifiers based on textual inputs. In a subset of 5,000 textual DSS from this http URL, we evaluate how well classifiers based on domain-specific pre-trained language models reproduce original availability categories as well as manually annotated labels. Typical metrics indicate that classifiers that predicted manual annotations outperformed those that learned to output the original availability categories. This suggests that the textual DSS descriptions contain applicable information that the availability categories do not, and that such classifiers could thus aid the automatic identification of available IPD in large trial databases. 

**Abstract (ZH)**: 临床试验中的数字个案参与者数据（IPD）越来越多地被分散用于潜在的科学再利用。然而，识别可用的IPD需要对大型数据库中的文本数据共享声明（DSS）进行解释。近年来，计算语言学的最新进展包括预训练语言模型，这些模型有望简化基于文本输入的有效分类器的实现。在从<此处省略网址>获取的5,000个文本DSS子集中，我们评估了基于领域特定预训练语言模型的分类器如何再现原始可用性类别以及手动标注的标签。典型的评估指标表明，预测手动标注的分类器优于学习输出原始可用性类别的分类器。这表明，文本DSS描述可能包含适用于评估可用性但未被说明的信息，因此这样的分类器可以帮助自动识别大型试验数据库中的可用IPD。 

---
# ConFit v2: Improving Resume-Job Matching using Hypothetical Resume Embedding and Runner-Up Hard-Negative Mining 

**Title (ZH)**: ConFit v2: 提高简历与职位匹配度的假设简历嵌入和备选负样本挖掘方法 

**Authors**: Xiao Yu, Ruize Xu, Chengyuan Xue, Jinzhong Zhang, Zhou Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12361)  

**Abstract**: A reliable resume-job matching system helps a company recommend suitable candidates from a pool of resumes and helps a job seeker find relevant jobs from a list of job posts. However, since job seekers apply only to a few jobs, interaction labels in resume-job datasets are sparse. We introduce ConFit v2, an improvement over ConFit to tackle this sparsity problem. We propose two techniques to enhance the encoder's contrastive training process: augmenting job data with hypothetical reference resume generated by a large language model; and creating high-quality hard negatives from unlabeled resume/job pairs using a novel hard-negative mining strategy. We evaluate ConFit v2 on two real-world datasets and demonstrate that it outperforms ConFit and prior methods (including BM25 and OpenAI text-embedding-003), achieving an average absolute improvement of 13.8% in recall and 17.5% in nDCG across job-ranking and resume-ranking tasks. 

**Abstract (ZH)**: 一种可靠的简历-岗位匹配系统有助于公司从众多简历中推荐合适的候选人，也有助于求职者从一系列招聘信息中找到相关岗位。然而，由于求职者通常只会申请少数几个岗位，简历-岗位数据集中的交互标签稀疏。我们介绍了ConFit v2，这是对ConFit的一种改进，旨在解决这一稀疏性问题。我们提出了两种增强编码器对比训练过程的技术：一是使用大型语言模型生成的假设参考简历来扩充岗位数据；二是通过一种新的负样本挖掘策略从未标记的简历-岗位配对中创建高质量的硬负样本。我们在两个真实世界数据集上评估了ConFit v2，并证明其在招聘排序和简历排序任务中的表现优于ConFit及先前的方法（包括BM25和OpenAI text-embedding-003），在召回率上平均提高了13.8%，在nDCG上提高了17.5%。 

---
# LM Agents for Coordinating Multi-User Information Gathering 

**Title (ZH)**: 多用户信息收集的LM代理协调方法 

**Authors**: Harsh Jhamtani, Jacob Andreas, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2502.12328)  

**Abstract**: This paper introduces PeopleJoin, a benchmark for evaluating LM-mediated collaborative problem solving. Given a user request, PeopleJoin agents must identify teammates who might be able to assist, converse with these teammates to gather information, and finally compile a useful answer or summary for the original user. PeopleJoin comprises two evaluation domains: PeopleJoin-QA, focused on questions about tabular data, and PeopleJoin-DocCreation, focused on document creation tasks. The two domains are adapted from existing NLP benchmarks for database question answering and multi-document summarization; here, however, the information needed to complete these tasks is distributed across synthetic ``organizations'' of 2--20 users, simulating natural multi-user collaboration scenarios. We implemented several popular LM agent architectures, evaluating their accuracy and efficiency at completing tasks, and highlight new research questions that can be studied using PeopleJoin. 

**Abstract (ZH)**: 本文介绍了PeopleJoin，这是一个用于评估基于LM的协作问题解决系统的基准。给定一个用户请求，PeopleJoin智能体必须识别可能能够提供帮助的队友，与这些队友进行交流以收集信息，并最终为原始用户提供一个有用的答案或总结。PeopleJoin包含两个评估领域：PeopleJoin-QA，专注于表格数据相关问题，以及PeopleJoin-DocCreation，专注于文档创建任务。这两个领域是从现有的数据库问答和多文档总结NLP基准中改编而来的；然而，在这里，完成这些任务所需的信息是分布在2-20名用户的合成“组织”中，模拟了自然的多用户协作场景。我们实现了几种流行的LM智能体架构，评估它们在完成任务时的准确性和效率，并指出可以通过PeopleJoin研究的新研究问题。 

---
# From Dense to Dynamic: Token-Difficulty Driven MoEfication of Pre-Trained LLMs 

**Title (ZH)**: 从密集到动态：基于令牌难度的预训练大型语言模型的MoEfication改造 

**Authors**: Kumari Nishu, Sachin Mehta, Samira Abnar, Mehrdad Farajtabar, Maxwell Horton, Mahyar Najibi, Moin Nabi, Minsik Cho, Devang Naik  

**Link**: [PDF](https://arxiv.org/pdf/2502.12325)  

**Abstract**: Training large language models (LLMs) for different inference constraints is computationally expensive, limiting control over efficiency-accuracy trade-offs. Moreover, once trained, these models typically process tokens uniformly, regardless of their complexity, leading to static and inflexible behavior. In this paper, we introduce a post-training optimization framework, DynaMoE, that adapts a pre-trained dense LLM to a token-difficulty-driven Mixture-of-Experts model with minimal fine-tuning cost. This adaptation makes the model dynamic, with sensitivity control to customize the balance between efficiency and accuracy. DynaMoE features a token-difficulty-aware router that predicts the difficulty of tokens and directs them to the appropriate sub-networks or experts, enabling larger experts to handle more complex tokens and smaller experts to process simpler ones. Our experiments demonstrate that DynaMoE can generate a range of adaptive model variants of the existing trained LLM with a single fine-tuning step, utilizing only $10B$ tokens, a minimal cost compared to the base model's training. Each variant offers distinct trade-offs between accuracy and performance. Compared to the baseline post-training optimization framework, Flextron, our method achieves similar aggregated accuracy across downstream tasks, despite using only $\frac{1}{9}\text{th}$ of their fine-tuning cost. 

**Abstract (ZH)**: 训练大型语言模型（LLMs）以适应不同的推理约束非常耗计算资源，限制了对效率-准确度权衡的控制。此外，在训练完成后，这些模型通常会均匀处理标记，不论标记的复杂性如何，导致静态和僵化的行为。本文介绍了一种后训练优化框架——DynaMoE，该框架通过最小的微调成本将预训练的密集LLM调整为基于标记难度的专家混合模型。这种调整使模型变得动态，并可通过灵敏度控制来定制效率与准确度之间的平衡。DynaMoE 特设了一个标记难度感知路由器，该路由器可以预测标记的难度并将它们导向合适的子网络或专家，使大专家能够处理更复杂的标记，而小专家则处理较简单的标记。我们的实验表明，DynaMoE 可以通过单一步微调生成现有训练LLM的各种自适应模型变体，并仅使用100亿个标记，微调成本远低于基模型的训练成本。每种变体在准确度与性能之间提供了不同的权衡。与基线后训练优化框架 Flextron 相比，尽管只使用了其微调成本的九分之一，我们的方法仍能在下游任务中获得相似的综合准确度。 

---
# Can Language Models Learn Typologically Implausible Languages? 

**Title (ZH)**: 语言模型能够学习类型学上不可能的语言吗？ 

**Authors**: Tianyang Xu, Tatsuki Kuribayashi, Yohei Oseki, Ryan Cotterell, Alex Warstadt  

**Link**: [PDF](https://arxiv.org/pdf/2502.12317)  

**Abstract**: Grammatical features across human languages show intriguing correlations often attributed to learning biases in humans. However, empirical evidence has been limited to experiments with highly simplified artificial languages, and whether these correlations arise from domain-general or language-specific biases remains a matter of debate. Language models (LMs) provide an opportunity to study artificial language learning at a large scale and with a high degree of naturalism. In this paper, we begin with an in-depth discussion of how LMs allow us to better determine the role of domain-general learning biases in language universals. We then assess learnability differences for LMs resulting from typologically plausible and implausible languages closely following the word-order universals identified by linguistic typologists. We conduct a symmetrical cross-lingual study training and testing LMs on an array of highly naturalistic but counterfactual versions of the English (head-initial) and Japanese (head-final) languages. Compared to similar work, our datasets are more naturalistic and fall closer to the boundary of plausibility. Our experiments show that these LMs are often slower to learn these subtly implausible languages, while ultimately achieving similar performance on some metrics regardless of typological plausibility. These findings lend credence to the conclusion that LMs do show some typologically-aligned learning preferences, and that the typological patterns may result from, at least to some degree, domain-general learning biases. 

**Abstract (ZH)**: 人类语言的句法特征显示出许多令人着迷的相关性，这些相关性通常归因于人类的学习偏见。然而，现有的实验证据主要局限于使用高度简化的人工语言进行的实验，关于这些相关性是源于一般的认知偏见还是特定语言的偏见仍存在争议。语言模型（LMs）为大规模、高自然度的人工语言学习研究提供了机会。本文首先深入讨论了LMs如何帮助我们更好地确定一般学习偏见在语言普遍性中的作用。然后，我们评估了不同类型的、语法上合理和不合理的人工语言对LMs可学习性的影响，这些类型的语言遵循语言类型学家识别出的词序普遍规律。我们进行了一项对英语（主谓结构）和日语（词尾结构）的高度自然、但构想上的语言版本进行训练和测试的对称跨语言研究。相较之前的类似研究，我们的数据集更加自然，更接近合理性的边缘。实验表明，这些LMs通常在学习这些微妙不合理的语言时较为缓慢，但从某些评估标准来看，其最终性能无论在合理化程度上如何变化都基本一致。这些发现支持了LMs显示出一些与语言类型学相吻合的学习偏好的结论，并表明这种语言类型学的模式至少部分地来自于一般学习偏见。 

---
# Warmup Generations: A Task-Agnostic Approach for Guiding Sequence-to-Sequence Learning with Unsupervised Initial State Generation 

**Title (ZH)**: 温启动生成：一种面向任务的无监督初始状态生成方法，用于指导序列到序列的学习 

**Authors**: Senyu Li, Zipeng Sun, Jiayi Wang, Xue Liu, Pontus Stenetorp, Siva Reddy, David Ifeoluwa Adelani  

**Link**: [PDF](https://arxiv.org/pdf/2502.12304)  

**Abstract**: Traditional supervised fine-tuning (SFT) strategies for sequence-to-sequence tasks often train models to directly generate the target output. Recent work has shown that guiding models with intermediate steps, such as keywords, outlines, or reasoning chains, can significantly improve performance, coherence, and interpretability. However, these methods often depend on predefined intermediate formats and annotated data, limiting their scalability and generalizability. In this work, we introduce a task-agnostic framework that enables models to generate intermediate "warmup" sequences. These warmup sequences, serving as an initial state for subsequent generation, are optimized to enhance the probability of generating the target sequence without relying on external supervision or human-designed structures. Drawing inspiration from reinforcement learning principles, our method iteratively refines these intermediate steps to maximize their contribution to the final output, similar to reward-driven optimization in reinforcement learning with human feedback. Experimental results across tasks such as translation, summarization, and multi-choice question answering for logical reasoning show that our approach outperforms traditional SFT methods, and offers a scalable and flexible solution for sequence-to-sequence tasks. 

**Abstract (ZH)**: 传统的序列到序列任务监督微调（SFT）策略通常训练模型直接生成目标输出。近期研究表明，通过使用中间步骤，如关键词、大纲或推理链，可以显著提高性能、连贯性和可解释性。然而，这些方法往往依赖于预先定义的中间格式和标注数据，限制了其可扩展性和泛化能力。本文提出了一种任务无关的框架，使模型能够生成中间的“热身”序列。这些热身序列作为后续生成过程的初始状态，并且被优化以增强生成目标序列的概率，而不依赖于外部监督或人工设计的结构。受到加强学习原则的启发，我们的方法通过迭代改进中间步骤来最大化其对最终输出的贡献，类似于在有反馈的人工智能强化学习中通过奖励驱动的优化过程。我们在翻译、总结和逻辑推理多选题回答等任务上的实验结果显示，我们提出的方法优于传统的SFT方法，并为序列到序列任务提供了一种可扩展且灵活的解决方案。 

---
# SMOL: Professionally translated parallel data for 115 under-represented languages 

**Title (ZH)**: SMOL：115种欠代表性语言的专业翻译平行数据 

**Authors**: Isaac Caswell, Elizabeth Nielsen, Jiaming Luo, Colin Cherry, Geza Kovacs, Hadar Shemtov, Partha Talukdar, Dinesh Tewari, Baba Mamadi Diane, Koulako Moussa Doumbouya, Djibrila Diane, Solo Farabado Cissé  

**Link**: [PDF](https://arxiv.org/pdf/2502.12301)  

**Abstract**: We open-source SMOL (Set of Maximal Overall Leverage), a suite of training data to unlock translation for low-resource languages (LRLs). SMOL has been translated into 115 under-resourced languages, including many for which there exist no previous public resources, for a total of 6.1M translated tokens. SMOL comprises two sub-datasets, each carefully chosen for maximum impact given its size: SMOL-Sent, a set of sentences chosen for broad unique token coverage, and SMOL-Doc, a document-level source focusing on a broad topic coverage. They join the already released GATITOS for a trifecta of paragraph, sentence, and token-level content. We demonstrate that using SMOL to prompt or fine-tune Large Language Models yields robust ChrF improvements. In addition to translation, we provide factuality ratings and rationales for all documents in SMOL-Doc, yielding the first factuality datasets for most of these languages. 

**Abstract (ZH)**: 我们将开源 SMOL（最大整体杠杆集），这是一个训练数据套件，用于解锁低资源语言（LRLs）的翻译。SMOL 已被翻译成 115 种资源不足的语言，包括许多之前没有任何公开资源的语言，总共包含 610 万翻译词元。SMOL 包含两个子数据集，每个子数据集都在其规模的基础上提供最大影响：SMOL-Sent，一个覆盖广泛独特词元的句子集；以及 SMOL-Doc，一个文档级的数据集，关注广泛的主题覆盖。这两个子数据集与之前发布的 GATITOS 一起组成了段落、句子和词元水平内容的完整套件。我们证明，使用 SMOL 来提示或微调大型语言模型可以显著提高 ChrF 衡量指标。除了翻译功能之外，我们还为 SMOL-Doc 中的所有文档提供了事实性评级和理由，从而为这些语言中的大多数语言提供了首个事实性数据集。 

---
# Evaluating Step-by-step Reasoning Traces: A Survey 

**Title (ZH)**: 逐步推理轨迹的评估：一种综述 

**Authors**: Jinu Lee, Julia Hockenmaier  

**Link**: [PDF](https://arxiv.org/pdf/2502.12289)  

**Abstract**: Step-by-step reasoning is widely used to enhance the reasoning ability of large language models (LLMs) in complex problems. Evaluating the quality of reasoning traces is crucial for understanding and improving LLM reasoning. However, the evaluation criteria remain highly unstandardized, leading to fragmented efforts in developing metrics and meta-evaluation benchmarks. To address this gap, this survey provides a comprehensive overview of step-by-step reasoning evaluation, proposing a taxonomy of evaluation criteria with four top-level categories (groundedness, validity, coherence, and utility). We then categorize metrics based on their implementations, survey which metrics are used for assessing each criterion, and explore whether evaluator models can transfer across different criteria. Finally, we identify key directions for future research. 

**Abstract (ZH)**: 逐步推理在复杂问题中广泛用于增强大型语言模型（LLM）的推理能力。评价推理轨迹的质量对于理解和改进LLM的推理至关重要。然而，评价标准仍然缺乏标准化，导致在开发度量标准和元评价基准方面存在碎片化的努力。为解决这一问题，本文综述了逐步推理的评价方法，并提出了一种评价标准分类体系，其中包括四个顶层类别（现实相关性、有效性、连贯性和实用性）。然后，根据其实现方式对度量标准进行分类，调查了用于评估每个标准的方法，并探讨了评估者模型是否能在不同标准间进行迁移。最后，指出了未来研究的关键方向。 

---
# Story Grammar Semantic Matching for Literary Study 

**Title (ZH)**: 文学研究中的叙事结构语义匹配 

**Authors**: Abigail Swenor, Neil Coffee, Walter Scheirer  

**Link**: [PDF](https://arxiv.org/pdf/2502.12276)  

**Abstract**: In Natural Language Processing (NLP), semantic matching algorithms have traditionally relied on the feature of word co-occurrence to measure semantic similarity. While this feature approach has proven valuable in many contexts, its simplistic nature limits its analytical and explanatory power when used to understand literary texts. To address these limitations, we propose a more transparent approach that makes use of story structure and related elements. Using a BERT language model pipeline, we label prose and epic poetry with story element labels and perform semantic matching by only considering these labels as features. This new method, Story Grammar Semantic Matching, guides literary scholars to allusions and other semantic similarities across texts in a way that allows for characterizing patterns and literary technique. 

**Abstract (ZH)**: 在自然语言处理（NLP）领域，语义匹配算法 traditionally 依赖于词共现特征来衡量语义相似性。尽管这种方法在许多场合中已被证明是很有价值的，但在理解和分析文学文本时，其简单性限制了其分析和解释的能力。为了解决这些限制，我们提出了一种更加透明的方法，该方法利用故事情节及其相关元素。通过使用 BERT 语言模型管道，我们将散文和史诗诗歌标记为故事情节元素，并仅基于这些标签进行语义匹配。这一新方法——故事情节语义匹配（Story Grammar Semantic Matching）——能够引导文学学者在文本间寻找引用和语义相似性，并有助于识别模式和文学技巧。 

---
# InfoQuest: Evaluating Multi-Turn Dialogue Agents for Open-Ended Conversations with Hidden Context 

**Title (ZH)**: InfoQuest: 评估具有隐藏上下文的开放领域多轮对话代理系统 

**Authors**: Bryan L. M. de Oliveira, Luana G. B. Martins, Bruno Brandão, Luckeciano C. Melo  

**Link**: [PDF](https://arxiv.org/pdf/2502.12257)  

**Abstract**: While large language models excel at following explicit instructions, they often struggle with ambiguous or incomplete user requests, defaulting to verbose, generic responses rather than seeking clarification. We introduce InfoQuest, a multi-turn chat benchmark designed to evaluate how dialogue agents handle hidden context in open-ended user requests. The benchmark presents intentionally ambiguous scenarios that require models to engage in information-seeking dialogue through clarifying questions before providing appropriate responses. Our evaluation of both open and closed-source models reveals that while proprietary models generally perform better, all current assistants struggle with effectively gathering critical information, often requiring multiple turns to infer user intent and frequently defaulting to generic responses without proper clarification. We provide a systematic methodology for generating diverse scenarios and evaluating models' information-seeking capabilities, offering insights into the current limitations of language models in handling ambiguous requests through multi-turn interactions. 

**Abstract (ZH)**: 尽管大规模语言模型在遵循明确指令方面表现出色，但在处理含糊不清或不完整的用户请求时，它们往往难以应对，倾向于给出冗长且通用的回复，而不是寻求进一步澄清。我们介绍了 InfoQuest，这是一种多轮对话基准测试，旨在评估对话代理在处理开放式用户请求中的隐含语境时的能力。该基准测试提供了一系列故意设计的模糊场景，要求模型通过提出澄清问题来开展信息查询对话，之后才给出合适的回复。通过对开源和闭源模型进行评估，我们发现，尽管专有模型通常表现较好，但所有当前的助手在有效收集关键信息方面仍存在困难，往往需要多次互动才能推断出用户意图，并且在缺乏有效澄清的情况下经常给出通用回复。我们提供了一种系统的方法来生成多样化场景并评估模型的信息查询能力，揭示了当前语言模型在通过多轮交互处理含糊请求方面的局限性。 

---
# GLoT: A Novel Gated-Logarithmic Transformer for Efficient Sign Language Translation 

**Title (ZH)**: GLoT：一种新颖的门控对数变压器，用于高效的手语翻译 

**Authors**: Nada Shahin, Leila Ismail  

**Link**: [PDF](https://arxiv.org/pdf/2502.12223)  

**Abstract**: Machine Translation has played a critical role in reducing language barriers, but its adaptation for Sign Language Machine Translation (SLMT) has been less explored. Existing works on SLMT mostly use the Transformer neural network which exhibits low performance due to the dynamic nature of the sign language. In this paper, we propose a novel Gated-Logarithmic Transformer (GLoT) that captures the long-term temporal dependencies of the sign language as a time-series data. We perform a comprehensive evaluation of GloT with the transformer and transformer-fusion models as a baseline, for Sign-to-Gloss-to-Text translation. Our results demonstrate that GLoT consistently outperforms the other models across all metrics. These findings underscore its potential to address the communication challenges faced by the Deaf and Hard of Hearing community. 

**Abstract (ZH)**: 机器翻译在减少语言障碍方面发挥了关键作用，但其在手语机器翻译（SLMT）中的适应性研究较少。现有SLMT工作的主要方法是使用Transformer神经网络，但由于手语的动态特性，其表现相对较低。本文提出了一种新的门控对数变压器（GLoT），该模型能够捕捉手语作为时间序列数据的长期时间依赖关系。我们以Transformer和Transformer融合模型作为基准，对GLoT在手语到手语字母再到文本翻译中的性能进行了全面评估。结果表明，GLoT在所有评估指标上都优于其他模型。这些发现凸显了其解决听障社区沟通挑战的潜力。 

---
# Zero Token-Driven Deep Thinking in LLMs: Unlocking the Full Potential of Existing Parameters via Cyclic Refinement 

**Title (ZH)**: 基于零标记的深度思考在大语言模型中的实现：通过循环精炼解锁现有参数的全部潜力 

**Authors**: Guanghao Li, Wenhao Jiang, Li Shen, Ming Tang, Chun Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2502.12214)  

**Abstract**: Resource limitations often constrain the parameter counts of Large Language Models (LLMs), hindering their performance. While existing methods employ parameter sharing to reuse the same parameter set under fixed budgets, such approaches typically force each layer to assume multiple roles with a predetermined number of iterations, restricting efficiency and adaptability. In this work, we propose the Zero Token Transformer (ZTT), which features a head-tail decoupled parameter cycling method. We disentangle the first (head) and last (tail) layers from parameter cycling and iteratively refine only the intermediate layers. Furthermore, we introduce a Zero-Token Mechanism, an internal architectural component rather than an input token, to guide layer-specific computation. At each cycle, the model retrieves a zero token (with trainable key values) from a Zero-Token Pool, integrating it alongside regular tokens in the attention mechanism. The corresponding attention scores not only reflect each layer's computational importance but also enable dynamic early exits without sacrificing overall model accuracy. Our approach achieves superior performance under tight parameter budgets, effectively reduces computational overhead via early exits, and can be readily applied to fine-tune existing pre-trained models for enhanced efficiency and adaptability. 

**Abstract (ZH)**: 资源限制往往制约了大型语言模型（LLMs）的参数数量，从而影响其性能。现有的方法通过在固定预算下复用相同的参数集来进行参数共享，但这些方法通常会要求每一层承担多个角色，并且需要进行预先确定的迭代次数，这限制了效率和适应性。本文中，我们提出了一种零令牌变压器（ZTT），它具有头尾解耦的参数循环方法。我们分离了第一层（头）和最后一层（尾）不参与参数循环，并且仅迭代优化中间层。此外，我们引入了零令牌机制（Zero-Token Mechanism），这是一种内部架构组件而非输入令牌，以指导特定层的计算。在每次循环中，模型会从零令牌池中检索一个零令牌（带可训练的关键值），并将它与常规令牌一起整合到注意力机制中。相应的注意力分数不仅反映了每一层计算的重要性，还使得可以在不牺牲整体模型准确性的情况下实现动态提前退出。我们的方法在紧张的参数预算下实现了卓越的性能，通过提前退出有效地减少了计算开销，并且可以便捷地应用于微调现有预训练模型，以提高效率和适应性。 

---
# Enhancing Frame Detection with Retrieval Augmented Generation 

**Title (ZH)**: 增强帧检测的检索增强生成方法 

**Authors**: Papa Abdou Karim Karou Diallo, Amal Zouaq  

**Link**: [PDF](https://arxiv.org/pdf/2502.12210)  

**Abstract**: Recent advancements in Natural Language Processing have significantly improved the extraction of structured semantic representations from unstructured text, especially through Frame Semantic Role Labeling (FSRL). Despite this progress, the potential of Retrieval-Augmented Generation (RAG) models for frame detection remains under-explored. In this paper, we present the first RAG-based approach for frame detection called RCIF (Retrieve Candidates and Identify Frames). RCIF is also the first approach to operate without the need for explicit target span and comprises three main stages: (1) generation of frame embeddings from various representations ; (2) retrieval of candidate frames given an input text; and (3) identification of the most suitable frames. We conducted extensive experiments across multiple configurations, including zero-shot, few-shot, and fine-tuning settings. Our results show that our retrieval component significantly reduces the complexity of the task by narrowing the search space thus allowing the frame identifier to refine and complete the set of candidates. Our approach achieves state-of-the-art performance on FrameNet 1.5 and 1.7, demonstrating its robustness in scenarios where only raw text is provided. Furthermore, we leverage the structured representation obtained through this method as a proxy to enhance generalization across lexical variations in the task of translating natural language questions into SPARQL queries. 

**Abstract (ZH)**: 近期自然语言处理领域的进展显著提高了从无结构文本中抽取结构化语义表示的能力，尤其是在通过框架语义角色标注（FSRL）方面取得了显著进步。尽管取得了这些进展，但检索增强生成（RAG）模型在框架检测方面的潜在应用仍被严重忽视。本文介绍了首个基于RAG的框架检测方法RCIF（Retrieve Candidates and Identify Frames）。RCIF也是首个无需明确目标跨度的方法，包含三个主要阶段：（1）从各种表示生成框架嵌入；（2）根据输入文本检索候选框架；（3）识别最合适的框架。我们在多种配置下进行了广泛实验，包括零样本、少量样本和微调设置。实验结果表明，我们的检索组件通过缩小搜索空间显著降低了任务的复杂性，从而使框架识别器能够进一步细化和补充候选集。我们的方法在FrameNet 1.5和1.7上达到了最先进的性能，展示了其在仅提供原始文本的场景下的鲁棒性。此外，我们利用通过此方法获得的结构化表示作为代理，来增强在将自然语言问题翻译为SPARQL查询任务中的泛化能力。 

---
# Predicting Depression in Screening Interviews from Interactive Multi-Theme Collaboration 

**Title (ZH)**: 预测筛查面试中基于互动多主题合作的抑郁症状筛查 

**Authors**: Xianbing Zhao, Yiqing Lyu, Di Wang, Buzhou Tang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12204)  

**Abstract**: Automatic depression detection provides cues for early clinical intervention by clinicians. Clinical interviews for depression detection involve dialogues centered around multiple themes. Existing studies primarily design end-to-end neural network models to capture the hierarchical structure of clinical interview dialogues. However, these methods exhibit defects in modeling the thematic content of clinical interviews: 1) they fail to capture intra-theme and inter-theme correlation explicitly, and 2) they do not allow clinicians to intervene and focus on themes of interest. To address these issues, this paper introduces an interactive depression detection framework. This framework leverages in-context learning techniques to identify themes in clinical interviews and then models both intra-theme and inter-theme correlation. Additionally, it employs AI-driven feedback to simulate the interests of clinicians, enabling interactive adjustment of theme importance. PDIMC achieves absolute improvements of 35\% and 12\% compared to the state-of-the-art on the depression detection dataset DAIC-WOZ, which demonstrates the effectiveness of modeling theme correlation and incorporating interactive external feedback. 

**Abstract (ZH)**: 自动抑郁检测为临床早期干预提供线索。临床抑郁症检测涉及围绕多个主题进行的对话。现有研究主要设计端到端的神经网络模型来捕捉临床访谈对话的层次结构。然而，这些方法在建模临床访谈的专题内容方面存在缺陷：1) 未能明确捕捉主题内部和主题间的关系，2) 无法让临床医生介入并专注于感兴趣的专题。为了解决这些问题，本文提出了一种交互式抑郁检测框架。该框架利用上下文学习技术来识别临床访谈中的专题，并同时建模主题内部和主题间的关系。此外，它采用以AI驱动的形式提供反馈，模拟临床医生的兴趣，使专题重要性能够进行互动调整。PDIMC在抑郁症检测数据集DAIC-WOZ上取得了绝对改进，分别达到了35%和12%，这表明建模专题关系和结合交互式外部反馈的有效性。 

---
# BoT: Breaking Long Thought Processes of o1-like Large Language Models through Backdoor Attack 

**Title (ZH)**: BoT：通过后门攻击打破类似o1的大型语言模型的长期思考过程 

**Authors**: Zihao Zhu, Hongbao Zhang, Mingda Zhang, Ruotong Wang, Guanzong Wu, Ke Xu, Baoyuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12202)  

**Abstract**: Longer thought, better performance: large language models with deep reasoning capabilities, particularly o1-like models, have demonstrated remarkable performance by generating extensive thought processes during inference. This trade-off reveals a potential vulnerability: adversaries could compromise model performance by forcing immediate responses without thought processes. To this end, in this paper, we introduce a novel attack scenario targeting the long thought processes of o1-like models and propose BoT (Break CoT), which can selectively break intrinsic reasoning mechanisms through backdoor attacks. BoT constructs poisoned datasets with designed triggers and injects backdoor by either supervised fine-tuning or direct preference optimization. When triggered, the model directly generates answers without thought processes, while maintaining normal reasoning capabilities for clean inputs. Extensive experiments on open-source o1-like models, including recent DeepSeek-R1, demonstrate that BoT nearly achieves high attack success rates while maintaining clean accuracy, highlighting the critical safety risk in current models. Furthermore, the relationship between task difficulty and helpfulness reveals a potential application for good, enabling users to customize model behavior based on task complexity. Code is available at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 长期思考，更好表现：具有深入推理能力的大语言模型，特别是o1-like模型，在推断过程中生成了广泛的思考过程，展示了卓越的表现。这种权衡揭示了一种潜在的脆弱性：攻击者可以通过迫使模型立即回应而不进行思考过程来削弱模型性能。为此，本文提出了一个新的攻击场景，针对o1-like模型的长期思考过程，并提出了一种名为BoT（Break CoT）的新攻击方法，该方法可以通过后门攻击选择性地破坏内在的推理机制。BoT通过设计触发器构造了受污染的数据集，并通过监督微调或直接偏好优化注入后门。当触发时，模型直接生成答案而没有思考过程，同时对于干净的输入保持正常的推理能力。针对开源o1-like模型，包括最近的DeepSeek-R1进行的大量实验表明，BoT几乎实现了高攻击成功率，并维持了清洁的准确性，突显了当前模型中的关键安全风险。此外，任务难度与帮助性的关系揭示了BoT潜在的应用价值，使用户能够根据任务复杂度定制模型行为。相关代码已发布在[this https URL](this https URL)。 

---
# Efficient and Effective Prompt Tuning via Prompt Decomposition and Compressed Outer Product 

**Title (ZH)**: 通过提示分解和压缩外积进行高效的提示调优 

**Authors**: Pengxiang Lan, Haoyu Xu, Enneng Yang, Yuliang Liang, Guibing Guo, Jianzhe Zhao, Xingwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12200)  

**Abstract**: Prompt tuning (PT) offers a cost-effective alternative to fine-tuning large-scale pre-trained language models (PLMs), requiring only a few parameters in soft prompt tokens added before the input text. However, existing PT approaches face two significant issues: (i) They overlook intrinsic semantic associations between soft prompt tokens, leading to high discreteness and limited interactions, thus reducing the model's comprehension and effectiveness in complex tasks. (ii) Due to the complexity of downstream tasks, long soft prompt is necessitated to improve performance, but prompt length correlates positively with memory usage and computational costs. Achieving high efficiency and performance remains an ongoing challenge. To address these issues, we propose a novel Low-parameters prompt tuning (LAMP) method, which leverages prompt decomposition and compressed outer product. Specifically, the prompt decomposition module employs Truncated SVD to reduce training parameters and significantly lower the dimensionality of the soft prompt parameter space. It then utilizes a compressed outer product module to facilitate multiple interactions among prompt tokens, exploring their intrinsic associations to enhance knowledge representation. Finally, LAMP uses average pooling to reduce memory usage and training/inference time. Extensive experiments across six architectures and eight datasets demonstrate that LAMP outperforms state-of-the-art PT-based and LoRA-based methods in performance and efficiency. 

**Abstract (ZH)**: 提示调谐（Prompt Tuning, PT）提供了一种低成本的替代方案，用于微调大规模的预训练语言模型（PLMs），只需在输入文本前添加少量软提示令牌即可。然而，现有的PT方法面临两个显著问题：(i) 它们忽视了软提示令牌之间的内在语义关联，导致高离散性并限制了互动，从而降低了模型在复杂任务中的理解和效果。(ii) 由于下游任务的复杂性，需要较长的软提示以提高性能，但软提示的长度与内存使用量和计算成本呈正相关。高效性和性能的实现仍然是一个持续的挑战。为了应对这些问题，我们提出了一种新颖的低参数提示调谐（Low-parameters Prompt Tuning, LAMP）方法，该方法利用了提示分解和压缩外积。具体而言，提示分解模块采用截断奇异值分解（Truncated SVD）来减少训练参数，并显著降低软提示参数空间的维度。然后，利用压缩外积模块促进多个提示令牌之间的交互，探索其内在关联以增强知识表示。最后，LAMP使用平均池化来减少内存使用量和训练/推理时间。在六个模型架构和八个数据集上的广泛实验表明，LAMP在性能和效率方面均优于最先进的PT基和LoRA基方法。 

---
# A Closer Look at System Prompt Robustness 

**Title (ZH)**: 对系统提示鲁棒性的更深入研究 

**Authors**: Norman Mu, Jonathan Lu, Michael Lavery, David Wagner  

**Link**: [PDF](https://arxiv.org/pdf/2502.12197)  

**Abstract**: System prompts have emerged as a critical control surface for specifying the behavior of LLMs in chat and agent settings. Developers depend on system prompts to specify important context, output format, personalities, guardrails, content policies, and safety countermeasures, all of which require models to robustly adhere to the system prompt, especially when facing conflicting or adversarial user inputs. In practice, models often forget to consider relevant guardrails or fail to resolve conflicting demands between the system and the user. In this work, we study various methods for improving system prompt robustness by creating realistic new evaluation and fine-tuning datasets based on prompts collected from from OpenAI's GPT Store and HuggingFace's HuggingChat. Our experiments assessing models with a panel of new and existing benchmarks show that performance can be considerably improved with realistic fine-tuning data, as well as inference-time interventions such as classifier-free guidance. Finally, we analyze the results of recently released reasoning models from OpenAI and DeepSeek, which show exciting but uneven improvements on the benchmarks we study. Overall, current techniques fall short of ensuring system prompt robustness and further study is warranted. 

**Abstract (ZH)**: 系统提示已成为确定聊天和代理设置中大型语言模型（LLM）行为的关键控制面。开发者依赖系统提示来指定重要背景、输出格式、个性、防护栏、内容政策以及安全对策，所有这些都需要模型在面对冲突或敌对用户输入时严格遵守系统提示。实践中，模型往往忽视相关的防护栏，或无法解决系统与用户之间的冲突需求。在此项工作中，我们通过基于从OpenAI的GPT Store和HuggingFace的HuggingChat收集的提示创建新的评估和微调数据集，研究改进系统提示鲁棒性的各种方法。我们的实验使用面板中新的和现有的基准测试模型表明，使用现实世界的微调数据以及推理时的干预措施（如无分类器指导）可以显著提高性能。最后，我们分析了OpenAI和DeepSeek最近发布的推理模型的结果，这些模型在我们研究的基准测试中显示出令人兴奋但不均衡的改进。总体而言，当前的技术手段未能确保系统提示的鲁棒性，需要进一步的研究。 

---
# AI and the Law: Evaluating ChatGPT's Performance in Legal Classification 

**Title (ZH)**: 人工智能与法律：评估ChatGPT在法律分类任务中的性能 

**Authors**: Pawel Weichbroth  

**Link**: [PDF](https://arxiv.org/pdf/2502.12193)  

**Abstract**: The use of ChatGPT to analyze and classify evidence in criminal proceedings has been a topic of ongoing discussion. However, to the best of our knowledge, this issue has not been studied in the context of the Polish language. This study addresses this research gap by evaluating the effectiveness of ChatGPT in classifying legal cases under the Polish Penal Code. The results show excellent binary classification accuracy, with all positive and negative cases correctly categorized. In addition, a qualitative evaluation confirms that the legal basis provided for each case, along with the relevant legal content, was appropriate. The results obtained suggest that ChatGPT can effectively analyze and classify evidence while applying the appropriate legal rules. In conclusion, ChatGPT has the potential to assist interested parties in the analysis of evidence and serve as a valuable legal resource for individuals with less experience or knowledge in this area. 

**Abstract (ZH)**: 关于使用ChatGPT分析和分类刑事诉讼中的证据，这一直是一个持续讨论的话题。然而，据我们所知，这一问题尚未在波兰语背景下进行研究。本研究通过评估ChatGPT在分类波兰刑法案例方面的有效性，来填补这一研究空白。结果显示，ChatGPT在二分类准确性方面表现出色，所有阳性与阴性案例都被正确分类。此外，定性评价证实，为每个案例提供的法律依据及其相关法律内容是适当的。研究结果表明，ChatGPT能够有效地分析和分类证据，并正确应用相关法律规则。总之，ChatGPT具有辅助有意向参与者进行证据分析的潜力，并可以为缺乏这一领域经验或知识的个人提供宝贵的法律资源。 

---
# Self-supervised Attribute-aware Dynamic Preference Ranking Alignment 

**Title (ZH)**: 自我监督的属性感知动态偏好对齐 

**Authors**: Hongyu Yang, Qi Zhao, Zhenhua hu, Rui Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.12189)  

**Abstract**: Reinforcement Learning from Human Feedback and its variants excel in aligning with human intentions to generate helpful, harmless, and honest responses. However, most of them rely on costly human-annotated pairwise comparisons for supervised alignment, which is not suitable for list-level scenarios, such as community question answering. Additionally, human preferences are influenced by multiple intrinsic factors in responses, leading to decision-making inconsistencies. Therefore, we propose \textbf{Se}lf-supervised \textbf{A}ttribute-aware \textbf{d}ynamic \textbf{p}reference \textbf{ra}nking, called \shortname. \ It quantifies preference differences between responses based on Attribute-Perceptual Distance Factors (APDF) and dynamically determines the list-wise alignment order. Furthermore, it achieves fine-grained preference difference learning and enables precise alignment with the optimal one. We specifically constructed a challenging code preference dataset named StaCoCoQA, and introduced more cost-effective and scalable preference evaluation metrics: PrefHit and PrefRecall. Extensive experimental results show that SeAdpra exhibits superior performance and generalizability on both StaCoCoQA and preference datasets from eight popular domains. 

**Abstract (ZH)**: 从人类反馈中进行强化学习及其变体在生成有助于揭示人类意图、无害和诚实响应方面表现出色。然而，它们大多依赖于昂贵的人标注成对比较进行监督对齐，这并不适用于列表级别的场景，如社区问答。此外，人类偏好受到响应中多个内在因素的影响，导致决策不一致。因此，我们提出了一种自我监督的属性感知动态偏好排名方法，称为**SeAdpra**。它基于属性感知距离因子（APDF）量化响应之间的偏好差异，并动态确定列表级别的对齐顺序。此外，它实现了细粒度的偏好差异学习，并能够实现与最优偏好的一致对齐。我们特别构建了一个具有挑战性的代码偏好数据集，名为StaCoCoQA，并引入了更经济高效且可扩展的偏好评估指标：PrefHit和PrefRecall。广泛的实验结果表明，SeAdpra在StaCoCoQA和八个流行领域偏好数据集上均表现出优越的性能和通用性。 

---
# Hallucinations are inevitable but statistically negligible 

**Title (ZH)**: 幻觉是不可避免的，但在统计上可以忽略不计。 

**Authors**: Atsushi Suzuki, Yulan He, Feng Tian, Zhongyuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12187)  

**Abstract**: Hallucinations, a phenomenon where a language model (LM) generates nonfactual content, pose a significant challenge to the practical deployment of LMs. While many empirical methods have been proposed to mitigate hallucinations, a recent study established a computability-theoretic result showing that any LM will inevitably generate hallucinations on an infinite set of inputs, regardless of the quality and quantity of training datasets and the choice of the language model architecture and training and inference algorithms. Although the computability-theoretic result may seem pessimistic, its significance in practical viewpoints has remained unclear. In contrast, we present a positive theoretical result from a probabilistic perspective. Specifically, we prove that hallucinations can be made statistically negligible, provided that the quality and quantity of the training data are sufficient. Interestingly, our positive result coexists with the computability-theoretic result, implying that while hallucinations on an infinite set of inputs cannot be entirely eliminated, their probability can always be reduced by improving algorithms and training data. By evaluating the two seemingly contradictory results through the lens of information theory, we argue that our probability-theoretic positive result better reflects practical considerations than the computability-theoretic negative result. 

**Abstract (ZH)**: 幻觉现象指的是语言模型（LM）生成非事实性内容的现象，这是在实际部署语言模型时面临的一个重大挑战。虽然已经提出了许多经验方法来减轻幻觉现象，但最近的一项研究通过计算理论结果表明，无论训练数据集的质量和数量如何，语言模型架构的选择，以及训练和推理算法的选择如何，任何语言模型都不可避免地会在无穷多个输入上生成幻觉。尽管计算理论结果看起来较为悲观，但其在实际应用中的重要意义仍然模糊不清。相比之下，我们从概率论的角度提出了一项积极的理论结果。具体来说，我们证明，在训练数据质量与数量足够的情况下，幻觉现象可以变得统计上可忽略不计。有趣的是，我们的积极结果与计算理论结果并存，暗示虽然在一个无穷多个输入集上消除幻觉是不可能的，但通过改进算法和训练数据，幻觉发生的概率总是可以降低的。通过信息论的视角评估这两个看似矛盾的结果，我们认为我们的概率论积极结果更好地反映了实际应用中的考虑，而非计算理论的消极结果。 

---
# Large Language Models for Extrapolative Modeling of Manufacturing Processes 

**Title (ZH)**: 大型语言模型在制造业过程扩展性建模中的应用 

**Authors**: Kiarash Naghavi Khanghah, Anandkumar Patel, Rajiv Malhotra, Hongyi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12185)  

**Abstract**: Conventional predictive modeling of parametric relationships in manufacturing processes is limited by the subjectivity of human expertise and intuition on the one hand and by the cost and time of experimental data generation on the other hand. This work addresses this issue by establishing a new Large Language Model (LLM) framework. The novelty lies in combining automatic extraction of process-relevant knowledge embedded in the literature with iterative model refinement based on a small amount of experimental data. This approach is evaluated on three distinct manufacturing processes that are based on machining, deformation, and additive principles. The results show that for the same small experimental data budget the models derived by our framework have unexpectedly high extrapolative performance, often surpassing the capabilities of conventional Machine Learning. Further, our approach eliminates manual generation of initial models or expertise-dependent interpretation of the literature. The results also reveal the importance of the nature of the knowledge extracted from the literature and the significance of both the knowledge extraction and model refinement components. 

**Abstract (ZH)**: 传统的制造过程中的参数关系预测建模受限于人工经验和直觉的主观性，以及实验数据生成的成本和时间。本研究通过建立一个新的大规模语言模型（LLM）框架来解决这一问题。其创新之处在于结合了自动提取文献中嵌入的过程相关知识，并基于少量实验数据进行迭代模型优化。该方法在基于加工、变形和增材原理的三个不同制造过程中进行了评估。结果表明，对于相同的实验数据预算，由我们的框架所得到的模型具有出乎意料的外推性能，往往超过传统机器学习的能力。此外，该方法消除了手动生成初始模型或依赖专业知识的文献解释。研究结果还揭示了从文献中提取的知识的本质的重要性，以及知识提取和模型优化两个环节的显著性。 

---
# Leveraging large language models for structured information extraction from pathology reports 

**Title (ZH)**: 利用大型语言模型从病理报告中提取结构化信息 

**Authors**: Jeya Balaji Balasubramanian, Daniel Adams, Ioannis Roxanis, Amy Berrington de Gonzalez, Penny Coulson, Jonas S. Almeida, Montserrat García-Closas  

**Link**: [PDF](https://arxiv.org/pdf/2502.12183)  

**Abstract**: Background: Structured information extraction from unstructured histopathology reports facilitates data accessibility for clinical research. Manual extraction by experts is time-consuming and expensive, limiting scalability. Large language models (LLMs) offer efficient automated extraction through zero-shot prompting, requiring only natural language instructions without labeled data or training. We evaluate LLMs' accuracy in extracting structured information from breast cancer histopathology reports, compared to manual extraction by a trained human annotator.
Methods: We developed the Medical Report Information Extractor, a web application leveraging LLMs for automated extraction. We developed a gold standard extraction dataset to evaluate the human annotator alongside five LLMs including GPT-4o, a leading proprietary model, and the Llama 3 model family, which allows self-hosting for data privacy. Our assessment involved 111 histopathology reports from the Breast Cancer Now (BCN) Generations Study, extracting 51 pathology features specified in the study's data dictionary.
Results: Evaluation against the gold standard dataset showed that both Llama 3.1 405B (94.7% accuracy) and GPT-4o (96.1%) achieved extraction accuracy comparable to the human annotator (95.4%; p = 0.146 and p = 0.106, respectively). While Llama 3.1 70B (91.6%) performed below human accuracy (p <0.001), its reduced computational requirements make it a viable option for self-hosting.
Conclusion: We developed an open-source tool for structured information extraction that can be customized by non-programmers using natural language. Its modular design enables reuse for various extraction tasks, producing standardized, structured data from unstructured text reports to facilitate analytics through improved accessibility and interoperability. 

**Abstract (ZH)**: 背景：从非结构化的病理科报告中提取结构化信息可以提高临床研究的数据可访问性。专家手动提取信息花费时间且成本高昂，限制了其可扩展性。大型语言模型（LLMs）通过零样本提示提供高效自动化提取，只需自然语言指示，无需标注数据或训练。我们评估了LLMs在提取乳腺癌病理科报告中的结构化信息方面的准确度，对比了由训练有素的人工注释员进行的手动提取。

方法：我们开发了医学生物报告信息抽取器（Medical Report Information Extractor，MRIE），这是一个基于LLMs的网络应用程序，用于自动化提取。我们开发了一个黄金标准提取数据集，以便在评估有经验的注释员的同时，评估五种LLMs的表现，包括领头的专有模型GPT-4o，以及允许数据隐私自托管的Llama 3模型系列。我们的评估涉及来自Breast Cancer Now（BCN）世代研究的111份病理科报告，提取了研究数据字典中规定的51个病理特征。

结果：与黄金标准数据集的评估结果显示，Llama 3.1 405B（准确度94.7%）和GPT-4o（准确度96.1%）的提取准确度与有经验的注释员（准确度95.4%）相当，p值分别为0.146和0.106。虽然Llama 3.1 70B（准确度91.6%）低于人工准确度（p < 0.001），但其较低的计算需求使其成为自托管的可行选项。

结论：我们开发了一个开源工具，可以由非程序员通过自然语言定制，其模块化设计使其能够用于多种提取任务，从非结构化的文本报告中生成标准化、结构化的数据，从而通过改善数据的可访问性和互操作性，促进数据分析。 

---
# Sleepless Nights, Sugary Days: Creating Synthetic Users with Health Conditions for Realistic Coaching Agent Interactions 

**Title (ZH)**: 失眠之夜，甜食之日：为现实的教练代理交互创建具有健康状况的合成用户 

**Authors**: Taedong Yun, Eric Yang, Mustafa Safdari, Jong Ha Lee, Vaishnavi Vinod Kumar, S. Sara Mahdavi, Jonathan Amar, Derek Peyton, Reut Aharony, Andreas Michaelides, Logan Schneider, Isaac Galatzer-Levy, Yugang Jia, John Canny, Arthur Gretton, Maja Matarić  

**Link**: [PDF](https://arxiv.org/pdf/2502.13135)  

**Abstract**: We present an end-to-end framework for generating synthetic users for evaluating interactive agents designed to encourage positive behavior changes, such as in health and lifestyle coaching. The synthetic users are grounded in health and lifestyle conditions, specifically sleep and diabetes management in this study, to ensure realistic interactions with the health coaching agent. Synthetic users are created in two stages: first, structured data are generated grounded in real-world health and lifestyle factors in addition to basic demographics and behavioral attributes; second, full profiles of the synthetic users are developed conditioned on the structured data. Interactions between synthetic users and the coaching agent are simulated using generative agent-based models such as Concordia, or directly by prompting a language model. Using two independently-developed agents for sleep and diabetes coaching as case studies, the validity of this framework is demonstrated by analyzing the coaching agent's understanding of the synthetic users' needs and challenges. Finally, through multiple blinded evaluations of user-coach interactions by human experts, we demonstrate that our synthetic users with health and behavioral attributes more accurately portray real human users with the same attributes, compared to generic synthetic users not grounded in such attributes. The proposed framework lays the foundation for efficient development of conversational agents through extensive, realistic, and grounded simulated interactions. 

**Abstract (ZH)**: 我们提出了一种端到端框架，用于生成合成用户，以评估旨在鼓励积极行为改变的交互式代理，例如健康和生活方式指导。合成用户基于健康和生活方式条件，特别是在本研究中专注于睡眠管理和糖尿病管理，以确保与健康指导代理进行现实的交互。合成用户分为两个阶段创建：首先，根据实际的健康和生活方式因素以及基本的人口统计和行为特征生成结构化数据；其次，在结构化数据的基础上开发合成用户的完整档案。合成用户与指导代理的交互通过生成性基于代理的模型（如Concordia）模拟，或者通过直接提示语言模型来实现。使用两个独立开发的睡眠和糖尿病指导代理作为案例研究，通过分析指导代理对合成用户需求和挑战的理解，展示了该框架的有效性。最后，通过多名专家对用户教练交互的盲评测，我们证明了具有健康和行为特征的合成用户比没有这些特征支撑的通用合成用户更准确地描绘了具有相同特征的真实人类用户。所提出的框架为通过广泛的、现实的和基于特征的模拟交互高效开发对话代理奠定了基础。 

---
# Rethinking Diverse Human Preference Learning through Principal Component Analysis 

**Title (ZH)**: 通过主成分分析重新思考多元人类偏好学习 

**Authors**: Feng Luo, Rui Yang, Hao Sun, Chunyuan Deng, Jiarui Yao, Jingyan Shen, Huan Zhang, Hanjie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.13131)  

**Abstract**: Understanding human preferences is crucial for improving foundation models and building personalized AI systems. However, preferences are inherently diverse and complex, making it difficult for traditional reward models to capture their full range. While fine-grained preference data can help, collecting it is expensive and hard to scale. In this paper, we introduce Decomposed Reward Models (DRMs), a novel approach that extracts diverse human preferences from binary comparisons without requiring fine-grained annotations. Our key insight is to represent human preferences as vectors and analyze them using Principal Component Analysis (PCA). By constructing a dataset of embedding differences between preferred and rejected responses, DRMs identify orthogonal basis vectors that capture distinct aspects of preference. These decomposed rewards can be flexibly combined to align with different user needs, offering an interpretable and scalable alternative to traditional reward models. We demonstrate that DRMs effectively extract meaningful preference dimensions (e.g., helpfulness, safety, humor) and adapt to new users without additional training. Our results highlight DRMs as a powerful framework for personalized and interpretable LLM alignment. 

**Abstract (ZH)**: 理解人类偏好对于提高基础模型并构建个性化的人工智能系统至关重要。然而，偏好本身是多样且复杂的，这使得传统的奖励模型难以全面捕捉它们的范围。尽管细粒度的偏好数据有所帮助，但收集这些数据是昂贵且难以扩展的。在本文中，我们提出了一种名为分解奖励模型（DRM, Decomposed Reward Models）的新颖方法，该方法通过二元比较来提取多样化的用户偏好，而不需依赖细粒度的注解。我们关键的洞察是将人类偏好表示为向量，并使用主成分分析（PCA, Principal Component Analysis）进行分析。通过构建偏好响应和未被接受响应的嵌入差异数据集，DRMs可以识别出能够捕捉偏好不同方面的正交基向量。这些分解的奖励可以灵活组合，以满足不同的用户需求，为传统奖励模型提供了一种可解释且可扩展的替代方案。我们证明，DRMs能够有效提取有意义的偏好维度（例如，有用性、安全性、趣味性），并在无需额外训练的情况下适应新用户。我们的结果突显了DRMs作为一种强大的框架，用于个性化和可解释的LLM（大型语言模型）对齐。 

---
# Understanding and Rectifying Safety Perception Distortion in VLMs 

**Title (ZH)**: 理解并纠正VLMs中的安全感知失真 

**Authors**: Xiaohan Zou, Jian Kang, George Kesidis, Lu Lin  

**Link**: [PDF](https://arxiv.org/pdf/2502.13095)  

**Abstract**: Recent studies reveal that vision-language models (VLMs) become more susceptible to harmful requests and jailbreak attacks after integrating the vision modality, exhibiting greater vulnerability than their text-only LLM backbones. To uncover the root cause of this phenomenon, we conduct an in-depth analysis and identify a key issue: multimodal inputs introduce an modality-induced activation shift toward a "safer" direction compared to their text-only counterparts, leading VLMs to systematically overestimate the safety of harmful inputs. We refer to this issue as safety perception distortion. To mitigate such distortion, we propose Activation Shift Disentanglement and Calibration (ShiftDC), a training-free method that decomposes and calibrates the modality-induced activation shift to reduce the impact of modality on safety. By isolating and removing the safety-relevant component, ShiftDC restores the inherent safety alignment of the LLM backbone while preserving the vision-language capabilities of VLMs. Empirical results demonstrate that ShiftDC significantly enhances alignment performance on safety benchmarks without impairing model utility. 

**Abstract (ZH)**: 近年来的研究表明，在集成视觉模态后，基于视觉的语言模型（Vision-Language Models，VLMs）对有害请求和破坏性攻击变得更加易受攻击，其脆弱性大于仅基于文本的语言模型（Language Models，LLMs）的基础架构。为了探究这一现象的根本原因，我们进行了深入分析，并确定了一个关键问题：多模态输入引入了一种相对于仅基于文本的对应物向“更安全”方向的模态诱导激活偏移，导致VLMs系统地高估了有害输入的安全性。我们将这一问题称为安全性感知失真。为了减轻这种失真，我们提出了一种无需训练的方法——激活偏移解耦与校准（Activation Shift Disentanglement and Calibration，ShiftDC），该方法通过分解和校准模态诱导的激活偏移来减少模态对安全性的影响。通过隔离并移除与安全性相关的内容，ShiftDC恢复了LLM基础架构的固有安全性对齐，同时保留了VLMs的语言视觉能力。实验结果表明，ShiftDC在不损害模型实用性的情况下显著提高了在安全性基准上的对齐性能。 

---
# Agentic Deep Graph Reasoning Yields Self-Organizing Knowledge Networks 

**Title (ZH)**: 代理深度图推理生成自我组织的知识网络 

**Authors**: Markus J. Buehler  

**Link**: [PDF](https://arxiv.org/pdf/2502.13025)  

**Abstract**: We present an agentic, autonomous graph expansion framework that iteratively structures and refines knowledge in situ. Unlike conventional knowledge graph construction methods relying on static extraction or single-pass learning, our approach couples a reasoning-native large language model with a continually updated graph representation. At each step, the system actively generates new concepts and relationships, merges them into a global graph, and formulates subsequent prompts based on its evolving structure. Through this feedback-driven loop, the model organizes information into a scale-free network characterized by hub formation, stable modularity, and bridging nodes that link disparate knowledge clusters. Over hundreds of iterations, new nodes and edges continue to appear without saturating, while centrality measures and shortest path distributions evolve to yield increasingly distributed connectivity. Our analysis reveals emergent patterns, such as the rise of highly connected 'hub' concepts and the shifting influence of 'bridge' nodes, indicating that agentic, self-reinforcing graph construction can yield open-ended, coherent knowledge structures. Applied to materials design problems, we present compositional reasoning experiments by extracting node-specific and synergy-level principles to foster genuinely novel knowledge synthesis, yielding cross-domain ideas that transcend rote summarization and strengthen the framework's potential for open-ended scientific discovery. We discuss other applications in scientific discovery and outline future directions for enhancing scalability and interpretability. 

**Abstract (ZH)**: 我们提出了一种代理性和自主性的图扩展框架，该框架迭代地在 situ 状态下结构化和精炼知识。与依靠静态提取或单次学习的传统知识图谱构建方法不同，我们的方法将一个内置推理能力的大语言模型与不断更新的图表示相结合。在每一步中，系统主动生成新的概念和关系，将它们合并到全局图中，并基于其不断演变的结构制定后续提示。通过这种反馈驱动的循环，模型将信息组织成一个无标度网络，其特征在于枢纽的形成、稳定的模ularity以及连接不同知识簇的桥接节点。经过数百次迭代，新的节点和边不断出现，而不会饱和，同时中心性测量值和最短路径分布的变化使得连接性越来越均匀分布。我们的分析揭示了新兴模式，如高度连接的“枢纽”概念的兴起和“桥接”节点影响力的改变，表明代理性和自我强化的图构建可以产生开放性和一致性的知识结构。在材料设计问题上，我们通过提取节点特定性和协同效应级别的原理来实现组合推理实验，以促进真正新颖的知识综合，产生跨领域的想法，并超越机械总结，从而增强框架在开放性科学发现方面的潜力。我们讨论了其他科学研究应用，并概述了增强可扩展性和可解释性的未来方向。 

---
# Towards a Design Guideline for RPA Evaluation: A Survey of Large Language Model-Based Role-Playing Agents 

**Title (ZH)**: 基于大型语言模型的角色扮演代理：RPA评估的设计指南调研 

**Authors**: Chaoran Chen, Bingsheng Yao, Ruishi Zou, Wenyue Hua, Weimin Lyu, Toby Jia-Jun Li, Dakuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13012)  

**Abstract**: Role-Playing Agent (RPA) is an increasingly popular type of LLM Agent that simulates human-like behaviors in a variety of tasks. However, evaluating RPAs is challenging due to diverse task requirements and agent designs. This paper proposes an evidence-based, actionable, and generalizable evaluation design guideline for LLM-based RPA by systematically reviewing 1,676 papers published between Jan. 2021 and Dec. 2024. Our analysis identifies six agent attributes, seven task attributes, and seven evaluation metrics from existing literature. Based on these findings, we present an RPA evaluation design guideline to help researchers develop more systematic and consistent evaluation methods. 

**Abstract (ZH)**: 角色扮演代理（RPA，Role-Playing Agent）是一种日益流行的LLM代理类型，能够在多种任务中模拟人类行为。然而，由于任务要求和代理设计的多样性，评估RPA具有挑战性。本文通过系统审查2021年1月到2024年12月间发表的1,676篇论文，提出了一套基于证据、可操作且可泛化的LLM基础RPA评估设计指南。我们的分析从现有文献中确定了六个代理属性、七个任务属性和七个评估指标。基于这些发现，我们提出了一套RPA评估设计指南，以帮助研究人员开发更加系统和一致的评估方法。 

---
# You need to MIMIC to get FAME: Solving Meeting Transcript Scarcity with a Multi-Agent Conversations 

**Title (ZH)**: 你需要模拟以获得名声：解决会议纪要稀缺性的一种多智能体对话方法 

**Authors**: Frederic Kirstein, Muneeb Khan, Jan Philip Wahle, Terry Ruas, Bela Gipp  

**Link**: [PDF](https://arxiv.org/pdf/2502.13001)  

**Abstract**: Meeting summarization suffers from limited high-quality data, mainly due to privacy restrictions and expensive collection processes. We address this gap with FAME, a dataset of 500 meetings in English and 300 in German produced by MIMIC, our new multi-agent meeting synthesis framework that generates meeting transcripts on a given knowledge source by defining psychologically grounded participant profiles, outlining the conversation, and orchestrating a large language model (LLM) debate. A modular post-processing step refines these outputs, mitigating potential repetitiveness and overly formal tones, ensuring coherent, credible dialogues at scale. We also propose a psychologically grounded evaluation framework assessing naturalness, social behavior authenticity, and transcript difficulties. Human assessments show that FAME approximates real-meeting spontaneity (4.5/5 in naturalness), preserves speaker-centric challenges (3/5 in spoken language), and introduces richer information-oriented difficulty (4/5 in difficulty). These findings highlight that FAME is a good and scalable proxy for real-world meeting conditions. It enables new test scenarios for meeting summarization research and other conversation-centric applications in tasks requiring conversation data or simulating social scenarios under behavioral constraints. 

**Abstract (ZH)**: 会议总结因高质量数据有限而受到限制，主要原因在于隐私限制和数据收集过程的高昂成本。我们通过提出FAME数据集来填补这一空白，该数据集包含500个英文会议和300个德文会议，由我们新开发的多智能体会议合成框架MIMIC生成。MIMIC通过定义心理依据的参与者配置文件、概述对话内容并组织大型语言模型（LLM）辩论，生成给定知识源的会议纪要。一个模块化后期处理步骤进一步细化这些输出，减少潜在的重复性和过于正式的语言，确保在大规模情况下生成连贯且可信的对话。我们还提出了一种基于心理学的评估框架，从自然性、社会行为的可信度以及对话内容的难度三个方面进行评估。人类评估表明，FAME在自然性（4.5/5分）方面接近真实的会议环境，在口语文本中的说话者中心挑战（3/5分）方面得到保留，同时增加了更丰富的信息导向难度（4/5分）。这些发现表明，FAME是一个良好且可扩展的模拟真实会议条件的代理。它为会议总结研究提供了新的测试场景，并在需要会话数据或在行为限制下模拟社会情境的任务中，为其他以会话为中心的应用程序提供了支持。 

---
# Adaptive Tool Use in Large Language Models with Meta-Cognition Trigger 

**Title (ZH)**: 带有元认知触发的大语言模型的自适应工具使用研究 

**Authors**: Wenjun Li, Dexun Li, Kuicai Dong, Cong Zhang, Hao Zhang, Weiwen Liu, Yasheng Wang, Ruiming Tang, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12961)  

**Abstract**: Large language models (LLMs) have shown remarkable emergent capabilities, transforming the execution of functional tasks by leveraging external tools for complex problems that require specialized processing or real-time data. While existing research expands LLMs access to diverse tools (e.g., program interpreters, search engines, weather/map apps), the necessity of using these tools is often overlooked, leading to indiscriminate tool invocation. This naive approach raises two key issues:(1) increased delays due to unnecessary tool calls, and (2) potential errors resulting from faulty interactions with external tools. In this paper, we introduce meta-cognition as a proxy for LLMs self-assessment of their capabilities, representing the model's awareness of its own limitations. Based on this, we propose MeCo, an adaptive decision-making strategy for external tool use. MeCo quantifies metacognitive scores by capturing high-level cognitive signals in the representation space, guiding when to invoke tools. Notably, MeCo is fine-tuning-free and incurs minimal cost. Our experiments show that MeCo accurately detects LLMs' internal cognitive signals and significantly improves tool-use decision-making across multiple base models and benchmarks. 

**Abstract (ZH)**: 大型语言模型（LLMs）展现出了卓越的涌现能力，通过利用外部工具来解决需要专业化处理或实时数据的复杂问题，从而改变功能任务的执行方式。虽然现有的研究扩展了LLMs对接多种工具的能力（例如程序解释器、搜索引擎、天气或地图应用程序），但这些工具使用必要性的重视程度往往不足，导致直接且无差别的工具调用。这种简单的方法存在着两个关键问题：（1）由于不必要的工具调用导致的延迟增加，以及（2）与外部工具的错误交互可能导致潜在的错误。在本文中，我们引入元认知作为LLMs自我评估能力的一个代理指标，代表模型对自己局限性的意识。基于此，我们提出了一种名为MeCo的自适应决策策略，用于外部工具的使用。MeCo通过捕获表示空间中的高级认知信号来量化元认知评分，从而指导何时调用工具。值得注意的是，MeCo无需微调且成本较低。我们的实验表明，MeCo能够准确地检测到LLMs的内部认知信号，并在多种基础模型和基准测试中显著提高了工具使用决策的质量。 

---
# Flow-of-Options: Diversified and Improved LLM Reasoning by Thinking Through Options 

**Title (ZH)**: 权衡流：通过权衡选项进行的多样化和改进的大语言模型推理 

**Authors**: Lakshmi Nair, Ian Trase, Mark Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.12929)  

**Abstract**: We present a novel reasoning approach called Flow-of-Options (FoO), designed to address intrinsic biases in Large Language Models (LLMs). FoO enables LLMs to systematically explore a diverse range of possibilities in their reasoning, as demonstrated by an FoO-based agentic system for autonomously solving Machine Learning tasks (AutoML). Our framework outperforms state-of-the-art baselines, achieving improvements of 38.2% - 69.2% on standard data science tasks, and 37.4% - 47.9% on therapeutic chemistry tasks. With an overall operation cost under $1 per task, our framework is well-suited for cost-sensitive applications. Beyond classification and regression, we illustrate the broader applicability of our FoO-based agentic system to tasks such as reinforcement learning and image generation. Our framework presents significant advancements compared to current state-of-the-art agentic systems for AutoML, due to the benefits of FoO in enforcing diversity in LLM solutions through compressed, explainable representations that also support long-term memory when combined with case-based reasoning. 

**Abstract (ZH)**: 我们提出了一种名为Flow-of-Options (FoO) 的新型推理方法，旨在解决大型语言模型（LLMs）固有的偏差问题。FoO 使 LLMS 能够系统地探索其推理中的多种可能性，通过基于 FoO 的自主系统（AutoML）自动解决机器学习任务得到了验证。我们的框架在标准数据科学任务中优于最先进的基准，性能提升了38.2% - 69.2%，在治疗性化学任务中提升了37.4% - 47.9%。由于总体操作成本低于每任务1美元，我们的框架非常适合成本敏感的应用。除了分类和回归任务，我们还展示了基于 FoO 的自主系统在强化学习和图像生成等更广泛任务中的应用潜力。与当前最先进的 AutoML 自主系统相比，我们的框架展示了显著的进步，因为 FoO 在通过压缩且可解释的表示形式强制多样性时，拉长了短期记忆，从而支持基于案例推理的长期记忆。 

---
# GSQ-Tuning: Group-Shared Exponents Integer in Fully Quantized Training for LLMs On-Device Fine-tuning 

**Title (ZH)**: GSQ-调优：全量化训练中用于设备上微调的组共享指数整数方法 

**Authors**: Sifan Zhou, Shuo Wang, Zhihang Yuan, Mingjia Shi, Yuzhang Shang, Dawei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12913)  

**Abstract**: Large Language Models (LLMs) fine-tuning technologies have achieved remarkable results. However, traditional LLM fine-tuning approaches face significant challenges: they require large Floating Point (FP) computation, raising privacy concerns when handling sensitive data, and are impractical for resource-constrained edge devices. While Parameter-Efficient Fine-Tuning (PEFT) techniques reduce trainable parameters, their reliance on floating-point arithmetic creates fundamental incompatibilities with edge hardware. In this work, we introduce a novel framework for on-device LLM fine-tuning that eliminates the need for floating-point operations in both inference and training, named GSQ-Tuning. At its core is the Group-Shared Exponents Integer format, which efficiently represents model parameters in integer format using shared exponents among parameter groups. When combined with LoRA-like adapters, this enables fully integer-based fine-tuning that is both memory and compute efficient. We demonstrate that our approach achieves accuracy comparable to FP16-based fine-tuning while significantly reducing memory usage (50%). Moreover, compared to FP8, our method can reduce 5x power consumption and 11x chip area with same performance, making large-scale model adaptation feasible on edge devices. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的微调技术已经取得了显著成果。然而，传统的大规模语言模型微调方法面临着重大挑战：它们需要大量的浮点运算（FP），处理敏感数据时引发隐私问题，并且对于资源受限的边缘设备来说是不切实际的。尽管参数高效微调（PEFT）技术减少了可训练参数的数量，但其依赖浮点运算在硬件层面造成了根本的不兼容性。在此项工作中，我们提出了一种新的框架，该框架在推断和训练过程中均不使用浮点运算，其名为GSQ-微调。其核心是群共享指数整数格式（Group-Shared Exponents Integer format），该格式利用参数组间的共享指数高效地以整数形式表示模型参数。结合类似于LoRA的适配器时，这种技术可以实现完全基于整数的微调，同时在内存和计算效率方面均表现出色。我们证明，我们的方法在准确性上与基于FP16的微调相当，但内存使用量减少了一半。此外，与FP8相比，该方法可以在相同性能下减少5倍的功耗和11倍的芯片面积，从而使得大规模模型适应在边缘设备上成为可能。 

---
# Towards Equitable AI: Detecting Bias in Using Large Language Models for Marketing 

**Title (ZH)**: 向着公平的AI：检测营销中使用大型语言模型中的偏见 

**Authors**: Berk Yilmaz, Huthaifa I. Ashqar  

**Link**: [PDF](https://arxiv.org/pdf/2502.12838)  

**Abstract**: The recent advances in large language models (LLMs) have revolutionized industries such as finance, marketing, and customer service by enabling sophisticated natural language processing tasks. However, the broad adoption of LLMs brings significant challenges, particularly in the form of social biases that can be embedded within their outputs. Biases related to gender, age, and other sensitive attributes can lead to unfair treatment, raising ethical concerns and risking both company reputation and customer trust. This study examined bias in finance-related marketing slogans generated by LLMs (i.e., ChatGPT) by prompting tailored ads targeting five demographic categories: gender, marital status, age, income level, and education level. A total of 1,700 slogans were generated for 17 unique demographic groups, and key terms were categorized into four thematic groups: empowerment, financial, benefits and features, and personalization. Bias was systematically assessed using relative bias calculations and statistically tested with the Kolmogorov-Smirnov (KS) test against general slogans generated for any individual. Results revealed that marketing slogans are not neutral; rather, they emphasize different themes based on demographic factors. Women, younger individuals, low-income earners, and those with lower education levels receive more distinct messaging compared to older, higher-income, and highly educated individuals. This underscores the need to consider demographic-based biases in AI-generated marketing strategies and their broader societal implications. The findings of this study provide a roadmap for developing more equitable AI systems, highlighting the need for ongoing bias detection and mitigation efforts in LLMs. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）的最新进展已经改变了金融、营销和客户服务等行业，使其能够执行复杂的自然语言处理任务。然而，LLMs的广泛应用也带来了重大挑战，尤其体现在其输出中可能嵌入的社会偏见。与性别、年龄和其他敏感属性相关的偏见可能导致不公平对待，引发伦理问题，损害公司声誉并降低客户信任。本研究通过对大型语言模型（如ChatGPT）生成的与金融相关的营销口号中的偏见进行了研究，提示了针对五大人口统计类别（性别、婚姻状况、年龄、收入水平和教育水平）定制的广告。共生成了1,700条口号，涉及17个独特的群体，并将关键术语归类为四个主题组：赋能、金融、利益和特点，以及个性化。通过相对偏倚计算和使用Kolmogorov-Smirnov（KS）测试与所有个人生成的一般口号进行统计测试，系统地评估了偏倚情况。结果显示，营销口号并非中立的，而是根据不同的人口统计因素强调不同的主题。女性、年轻个体、低收入者和低教育水平者收到的信息与年长、高收入和高教育水平个体相比更加独特。这强调了在AI生成的营销策略及其更广泛的社会影响中考虑基于人口统计的偏见的重要性。本研究的发现为开发更加公平的AI系统提供了蓝图，突显了持续偏见检测和缓解努力在LLMs中的必要性。 

---
# Iron Sharpens Iron: Defending Against Attacks in Machine-Generated Text Detection with Adversarial Training 

**Title (ZH)**: 铁磨铁：通过对抗训练提升机器生成文本检测中的防御能力 

**Authors**: Yuanfan Li, Zhaohan Zhang, Chengzhengxu Li, Chao Shen, Xiaoming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12734)  

**Abstract**: Machine-generated Text (MGT) detection is crucial for regulating and attributing online texts. While the existing MGT detectors achieve strong performance, they remain vulnerable to simple perturbations and adversarial attacks. To build an effective defense against malicious perturbations, we view MGT detection from a threat modeling perspective, that is, analyzing the model's vulnerability from an adversary's point of view and exploring effective mitigations. To this end, we introduce an adversarial framework for training a robust MGT detector, named GREedy Adversary PromoTed DefendER (GREATER). The GREATER consists of two key components: an adversary GREATER-A and a detector GREATER-D. The GREATER-D learns to defend against the adversarial attack from GREATER-A and generalizes the defense to other attacks. GREATER-A identifies and perturbs the critical tokens in embedding space, along with greedy search and pruning to generate stealthy and disruptive adversarial examples. Besides, we update the GREATER-A and GREATER-D synchronously, encouraging the GREATER-D to generalize its defense to different attacks and varying attack intensities. Our experimental results across 9 text perturbation strategies and 5 adversarial attacks show that our GREATER-D reduces the Attack Success Rate (ASR) by 10.61% compared with SOTA defense methods while our GREATER-A is demonstrated to be more effective and efficient than SOTA attack approaches. 

**Abstract (ZH)**: 机器生成文本（MGT）检测对于规范和归因在线文本至关重要。虽然现有的MGT检测器表现出色，但它们仍然容易受到简单的扰动和对抗性攻击的影响。为了构建有效的对抗恶意扰动的防御机制，我们从威胁建模的角度出发，即从攻击者的视角分析模型的脆弱性，并探索有效的缓解措施。为此，我们提出了一种对抗性框架，用于训练一个稳健的MGT检测器，命名为GREedy Adversary PromoTed DefendER（GREATER）。GREATER包含两个关键组件：攻击者GREATER-A和检测器GREATER-D。GREATER-D学习从GREATER-A的攻击中进行防御，并将防御推广到其他类型的攻击。GREATER-A识别并扰动生成关键嵌入空间中的对抗样本，并结合贪婪搜索和剪枝以生成隐形且具有破坏性的对抗样本。此外，我们同步更新GREATER-A和GREATER-D，鼓励GREATER-D将其防御推广到不同的攻击类型和 varied 攻击强度。我们的实验结果表明，在9种文本扰动策略和5种对抗性攻击下，与最先进的防御方法相比，我们的GREATER-D将攻击成功率（ASR）降低了10.61%；同时，我们证明GREATER-A在有效性和效率方面都优于最先进的攻击方法。 

---
# Multi-Step Alignment as Markov Games: An Optimistic Online Gradient Descent Approach with Convergence Guarantees 

**Title (ZH)**: 多步对齐作为马尔可夫博弈：带有收敛保证的乐观在线梯度下降方法 

**Authors**: Yongtao Wu, Luca Viano, Yihang Chen, Zhenyu Zhu, Kimon Antonakopoulos, Quanquan Gu, Volkan Cevher  

**Link**: [PDF](https://arxiv.org/pdf/2502.12678)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) has been highly successful in aligning large language models with human preferences. While prevalent methods like DPO have demonstrated strong performance, they frame interactions with the language model as a bandit problem, which limits their applicability in real-world scenarios where multi-turn conversations are common. Additionally, DPO relies on the Bradley-Terry model assumption, which does not adequately capture the non-transitive nature of human preferences. In this paper, we address these challenges by modeling the alignment problem as a two-player constant-sum Markov game, where each player seeks to maximize their winning rate against the other across all steps of the conversation. Our approach Multi-step Preference Optimization (MPO) is built upon the natural actor-critic framework~\citep{peters2008natural}. We further develop OMPO based on the optimistic online gradient descent algorithm~\citep{rakhlin2013online,joulani17a}. Theoretically, we provide a rigorous analysis for both algorithms on convergence and show that OMPO requires $\mathcal{O}(\epsilon^{-1})$ policy updates to converge to an $\epsilon$-approximate Nash equilibrium. We also validate the effectiveness of our method on multi-turn conversations dataset and math reasoning dataset. 

**Abstract (ZH)**: 人类反馈强化学习（RLHF）在使大型语言模型与人类偏好一致方面取得了极大的成功。尽管像DPO这样的流行方法展示了强大的性能，但它们将与语言模型的交互视为一个多臂老虎机问题，这限制了它们在多轮对话普遍存在的现实场景中的应用。此外，DPO依赖于Bradley-Terry模型假设，这一假设未能充分捕捉人类偏好的非传递性。在本文中，我们通过将对齐问题建模为两名玩家的常和马尔可夫博弈来应对这些挑战，在此博弈中，每个玩家都在整个对话过程中寻求最大化与另一方的获胜比例。我们的方法是多步偏好优化（MPO），它基于自然演员-评论家框架~\citep{peters2008natural}构建。我们进一步基于乐观在线梯度下降算法~\citep{rakhlin2013online,joulani17a}开发了OMPO。理论上，我们对这两种算法的收敛性进行了严格的分析，并证明了OMPO仅需$\mathcal{O}(\epsilon^{-1})$次策略更新即可收敛到$\epsilon$近似纳什均衡。我们还通过多轮对话数据集和数学推理数据集验证了我们方法的有效性。 

---
# DeepResonance: Enhancing Multimodal Music Understanding via Music-centric Multi-way Instruction Tuning 

**Title (ZH)**: DeepResonance：通过以音乐为中心的多方式指令调优增强多模态音乐理解 

**Authors**: Zhuoyuan Mao, Mengjie Zhao, Qiyu Wu, Hiromi Wakaki, Yuki Mitsufuji  

**Link**: [PDF](https://arxiv.org/pdf/2502.12623)  

**Abstract**: Recent advancements in music large language models (LLMs) have significantly improved music understanding tasks, which involve the model's ability to analyze and interpret various musical elements. These improvements primarily focused on integrating both music and text inputs. However, the potential of incorporating additional modalities such as images, videos and textual music features to enhance music understanding remains unexplored. To bridge this gap, we propose DeepResonance, a multimodal music understanding LLM fine-tuned via multi-way instruction tuning with multi-way aligned music, text, image, and video data. To this end, we construct Music4way-MI2T, Music4way-MV2T, and Music4way-Any2T, three 4-way training and evaluation datasets designed to enable DeepResonance to integrate both visual and textual music feature content. We also introduce multi-sampled ImageBind embeddings and a pre-alignment Transformer to enhance modality fusion prior to input into text LLMs, tailoring DeepResonance for multi-way instruction tuning. Our model achieves state-of-the-art performances across six music understanding tasks, highlighting the benefits of the auxiliary modalities and the structural superiority of DeepResonance. We plan to open-source the models and the newly constructed datasets. 

**Abstract (ZH)**: 近年来，音乐大规模语言模型（LLMs）的进展显著提升了音乐理解任务的效果，这些任务涉及模型分析和解释各种音乐元素的能力。这些改进主要集中在整合音乐和文本输入。然而，将图像、视频和文本音乐特征等其他模态纳入以增强音乐理解的可能性尚未得到探索。为了弥合这一差距，我们提出了一种名为DeepResonance的多模态音乐理解LLM，该模型通过多方式指令调用来微调，并使用多方式对齐的音乐、文本、图像和视频数据。为此，我们构建了Music4way-MI2T、Music4way-MV2T和Music4way-Any2T三个四模态训练和评估数据集，旨在使DeepResonance能够整合视觉和文本音乐特征内容。我们还引入了多采样的ImageBind嵌入和预对齐的Transformer，以增强输入到文本LLMs之前的模态融合，从而将DeepResonance定制为多方式指令调用的需求。我们的模型在六项音乐理解任务中取得了最先进的性能，突显了辅助模态和DeepResonance结构优势的好处。我们计划开源这些模型和新构建的数据集。 

---
# CutPaste&Find: Efficient Multimodal Hallucination Detector with Visual-aid Knowledge Base 

**Title (ZH)**: CutPaste&Find：基于视觉辅助知识库的高效多模态 hallucination 检测器 

**Authors**: Cong-Duy Nguyen, Xiaobao Wu, Duc Anh Vu, Shuai Zhao, Thong Nguyen, Anh Tuan Luu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12591)  

**Abstract**: Large Vision-Language Models (LVLMs) have demonstrated impressive multimodal reasoning capabilities, but they remain susceptible to hallucination, particularly object hallucination where non-existent objects or incorrect attributes are fabricated in generated descriptions. Existing detection methods achieve strong performance but rely heavily on expensive API calls and iterative LVLM-based validation, making them impractical for large-scale or offline use. To address these limitations, we propose CutPaste\&Find, a lightweight and training-free framework for detecting hallucinations in LVLM-generated outputs. Our approach leverages off-the-shelf visual and linguistic modules to perform multi-step verification efficiently without requiring LVLM inference. At the core of our framework is a Visual-aid Knowledge Base that encodes rich entity-attribute relationships and associated image representations. We introduce a scaling factor to refine similarity scores, mitigating the issue of suboptimal alignment values even for ground-truth image-text pairs. Comprehensive evaluations on benchmark datasets, including POPE and R-Bench, demonstrate that CutPaste\&Find achieves competitive hallucination detection performance while being significantly more efficient and cost-effective than previous methods. 

**Abstract (ZH)**: 大型多模态语言模型（Large Vision-Language Models, LVLMs）展现了出色的跨模态推理能力，但仍然容易产生幻觉，尤其是在生成描述中创造出不存在的物体或错误的属性。现有的检测方法表现出色，但高度依赖昂贵的API调用和迭代的LVLM验证，这使得它们在大规模或离线使用时不可行。为了解决这些问题，我们提出了一种轻量级且无需训练的框架CutPaste\&Find，专门用于检测LVLM生成输出中的幻觉。我们的方法利用现成的视觉和语言模块高效地执行多步验证，无需进行LVLM推理。该框架的核心是一个视觉辅助知识库，用于编码丰富的实体-属性关系及其相关图像表示。我们引入了一个缩放因子来细化相似度评分，即使对于真实的图像-文本对也能缓解不佳对齐值的问题。在基准数据集（包括POPE和R-Bench）上的全面评估表明，CutPaste\&Find在幻觉检测性能上达到了可竞争的水平，同时在效率和成本效益方面远优于先前的方法。 

---
# G-Refer: Graph Retrieval-Augmented Large Language Model for Explainable Recommendation 

**Title (ZH)**: G-Refer：图检索增强的大语言模型可解释推荐系统 

**Authors**: Yuhan Li, Xinni Zhang, Linhao Luo, Heng Chang, Yuxiang Ren, Irwin King, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.12586)  

**Abstract**: Explainable recommendation has demonstrated significant advantages in informing users about the logic behind recommendations, thereby increasing system transparency, effectiveness, and trustworthiness. To provide personalized and interpretable explanations, existing works often combine the generation capabilities of large language models (LLMs) with collaborative filtering (CF) information. CF information extracted from the user-item interaction graph captures the user behaviors and preferences, which is crucial for providing informative explanations. However, due to the complexity of graph structure, effectively extracting the CF information from graphs still remains a challenge. Moreover, existing methods often struggle with the integration of extracted CF information with LLMs due to its implicit representation and the modality gap between graph structures and natural language explanations. To address these challenges, we propose G-Refer, a framework using graph retrieval-augmented large language models (LLMs) for explainable recommendation. Specifically, we first employ a hybrid graph retrieval mechanism to retrieve explicit CF signals from both structural and semantic perspectives. The retrieved CF information is explicitly formulated as human-understandable text by the proposed graph translation and accounts for the explanations generated by LLMs. To bridge the modality gap, we introduce knowledge pruning and retrieval-augmented fine-tuning to enhance the ability of LLMs to process and utilize the retrieved CF information to generate explanations. Extensive experiments show that G-Refer achieves superior performance compared with existing methods in both explainability and stability. Codes and data are available at this https URL. 

**Abstract (ZH)**: 可解释推荐已经在向用户传达推荐背后的逻辑方面展示了显著优势，从而增加了系统的透明度、有效性以及可信度。为了提供个性化且可解释的解释，现有工作通常将大型语言模型（LLMs）的生成能力与协同过滤（CF）信息结合在一起。从用户项交互图中提取的CF信息捕捉了用户的行为和偏好，这对于提供有信息性的解释至关重要。然而，由于图结构的复杂性，有效地从图中提取CF信息仍然是一项挑战。此外，现有的方法通常难以将提取的CF信息与LLMs整合，因为这些信息具有隐式表示，并且图结构与自然语言解释之间存在模态差异。为了解决这些问题，我们提出了一种名为G-Refer的框架，该框架利用图检索增强的大型语言模型进行可解释推荐。具体而言，我们首先采用一种混合的图检索机制，从结构和语义两个方面检索显式的CF信号。提取的CF信息通过提出的图翻译过程显式地形式化为人类可理解的文本，并为LLMs生成的解释提供依据。为了弥合模态差异，我们引入了知识剪枝和检索增强微调，以增强LLMs处理并利用检索到的CF信息生成解释的能力。广泛实验表明，G-Refer在可解释性和稳定性方面均优于现有方法。代码和数据可在以下网址获取：this https URL。 

---
# UXAgent: An LLM Agent-Based Usability Testing Framework for Web Design 

**Title (ZH)**: UXAgent：基于大语言模型代理的Web设计可用性测试框架 

**Authors**: Yuxuan Lu, Bingsheng Yao, Hansu Gu, Jing Huang, Jessie Wang, Laurence Li, Jiri Gesi, Qi He, Toby Jia-Jun Li, Dakuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12561)  

**Abstract**: Usability testing is a fundamental yet challenging (e.g., inflexible to iterate the study design flaws and hard to recruit study participants) research method for user experience (UX) researchers to evaluate a web design. Recent advances in Large Language Model-simulated Agent (LLM-Agent) research inspired us to design UXAgent to support UX researchers in evaluating and reiterating their usability testing study design before they conduct the real human subject study. Our system features an LLM-Agent module and a universal browser connector module so that UX researchers can automatically generate thousands of simulated users to test the target website. The results are shown in qualitative (e.g., interviewing how an agent thinks ), quantitative (e.g., # of actions), and video recording formats for UX researchers to analyze. Through a heuristic user evaluation with five UX researchers, participants praised the innovation of our system but also expressed concerns about the future of LLM Agent-assisted UX study. 

**Abstract (ZH)**: 易用性测试是用户体验（UX）研究人员评估网页设计的基础但具有挑战性的研究方法（例如，难以迭代研究设计中的缺陷，且难以招募研究参与者）。最近在大型语言模型模拟代理（LLM-Agent）研究方面的进展激发我们设计了UXAgent，以便于在进行真正的人类用户研究之前支持UX研究人员评估和迭代其易用性测试研究设计。我们的系统包括一个LLM-Agent模块和一个通用浏览器连接器模块，使得UX研究人员可以自动生成成千上万的模拟用户来测试目标网站。结果显示为定性（例如，通过采访代理的思维过程）、定量（例如，操作次数）和视频录制格式，供UX研究人员分析。通过五名UX研究人员的启发性用户评估，参与者赞扬了我们系统的创新性，但也对LLM代理辅助用户体验研究的未来表达了担忧。 

---
# EquiBench: Benchmarking Code Reasoning Capabilities of Large Language Models via Equivalence Checking 

**Title (ZH)**: EquiBench: 通过等价性检查评估大型语言模型的代码推理能力 

**Authors**: Anjiang Wei, Jiannan Cao, Ran Li, Hongyu Chen, Yuhui Zhang, Ziheng Wang, Yaofeng Sun, Yuan Liu, Thiago S. F. X. Teixeira, Diyi Yang, Ke Wang, Alex Aiken  

**Link**: [PDF](https://arxiv.org/pdf/2502.12466)  

**Abstract**: Equivalence checking, i.e., determining whether two programs produce identical outputs for all possible inputs, underpins a broad range of applications, including software refactoring, testing, and optimization. We present the task of equivalence checking as a new way to evaluate the code reasoning abilities of large language models (LLMs). We introduce EquiBench, a dataset of 2400 program pairs spanning four programming languages and six equivalence categories. These pairs are systematically generated through program analysis, compiler scheduling, and superoptimization, covering nontrivial structural transformations that demand deep semantic reasoning beyond simple syntactic variations. Our evaluation of 17 state-of-the-art LLMs shows that OpenAI o3-mini achieves the highest overall accuracy of 78.0%. In the most challenging categories, the best accuracies are 62.3% and 68.8%, only modestly above the 50% random baseline for binary classification, indicating significant room for improvement in current models' code reasoning capabilities. 

**Abstract (ZH)**: 等价性检查，即确定两个程序在所有可能的输入下是否产生相同的输出结果，是软件重构、测试和优化等广泛应用的基础。我们将等价性检查任务作为评估大规模语言模型（LLMs）代码推理能力的一种新方法。我们介绍了EquiBench数据集，其中包含2400个程序对，涵盖了四种编程语言和六类等价性类别。这些程序对通过程序分析、编译器调度和超优化系统生成，涵盖了许多要求深层次语义推理而非仅仅简单的语法变化的复杂结构变换。对17个先进LLMs的评估结果显示，OpenAI的o3-mini在总体准确率上最高，达到78.0%。在最具挑战性的类别中，最佳准确率为62.3%和68.8%，这些准确率仅比二分类的随机基线高出50%，表明当前模型在代码推理能力方面仍有很大的改进空间。 

---
# HopRAG: Multi-Hop Reasoning for Logic-Aware Retrieval-Augmented Generation 

**Title (ZH)**: HopRAG：逻辑感知检索增强生成的多跳推理方法 

**Authors**: Hao Liu, Zhengren Wang, Xi Chen, Zhiyu Li, Feiyu Xiong, Qinhan Yu, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12442)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems often struggle with imperfect retrieval, as traditional retrievers focus on lexical or semantic similarity rather than logical relevance. To address this, we propose HopRAG, a novel RAG framework that augments retrieval with logical reasoning through graph-structured knowledge exploration. During indexing, HopRAG constructs a passage graph, with text chunks as vertices and logical connections established via LLM-generated pseudo-queries as edges. During retrieval, it employs a retrieve-reason-prune mechanism: starting with lexically or semantically similar passages, the system explores multi-hop neighbors guided by pseudo-queries and LLM reasoning to identify truly relevant ones. Extensive experiments demonstrate HopRAG's superiority, achieving 76.78\% higher answer accuracy and 65.07\% improved retrieval F1 score compared to conventional methods. The repository is available at this https URL. 

**Abstract (ZH)**: 检索增强生成（RAG）系统通常在检索不完美时遇到困难，因为传统的检索器关注词汇或语义相似性，而不是逻辑相关性。为了解决这个问题，我们提出了HopRAG，这是一种新的RAG框架，通过图结构的知识探索引入逻辑推理来增强检索。在索引过程中，HopRAG构建一篇文字段落图，其中文本片段作为节点，逻辑连接通过LLM生成的伪查询作为边建立。在检索过程中，它采用检索-推理-修剪机制：从词汇或语义相似的文字段落开始，系统根据伪查询和LLM推理指导的多跳邻居进行探索，以识别真正相关的内容。 extensive 实验表明，HopRAG 在答案准确性方面优于传统方法，答案准确性提高了76.78%，检索F1分数提高了65.07%。源代码库可在以下链接访问：this https URL。 

---
# A Survey on Large Language Models for Automated Planning 

**Title (ZH)**: 大型语言模型在自动规划中的应用综述 

**Authors**: Mohamed Aghzal, Erion Plaku, Gregory J. Stein, Ziyu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2502.12435)  

**Abstract**: The planning ability of Large Language Models (LLMs) has garnered increasing attention in recent years due to their remarkable capacity for multi-step reasoning and their ability to generalize across a wide range of domains. While some researchers emphasize the potential of LLMs to perform complex planning tasks, others highlight significant limitations in their performance, particularly when these models are tasked with handling the intricacies of long-horizon reasoning. In this survey, we critically investigate existing research on the use of LLMs in automated planning, examining both their successes and shortcomings in detail. We illustrate that although LLMs are not well-suited to serve as standalone planners because of these limitations, they nonetheless present an enormous opportunity to enhance planning applications when combined with other approaches. Thus, we advocate for a balanced methodology that leverages the inherent flexibility and generalized knowledge of LLMs alongside the rigor and cost-effectiveness of traditional planning methods. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）的规划能力引起了广泛关注，这主要是因为它们在多步推理方面表现出色，并且能够跨多种领域进行泛化。虽然一些研究人员强调LLMs在执行复杂规划任务方面的潜力，但也有一些研究者指出了它们在性能上的显著局限性，尤其是在处理长时提前推理的复杂性方面。在本文综述中，我们对LLMs在自动化规划中的应用进行了批判性考察，详细分析了它们的成功与不足之处。我们表明，尽管由于这些局限性，LLMs并不适合作为独立的规划者，但它们与其它方法结合使用时可以极大地提升规划应用的效果。因此，我们提倡一种平衡的方法，这种方法利用LLMs固有的灵活性和泛化知识，同时结合传统规划方法的严谨性和经济性。 

---
# Independence Tests for Language Models 

**Title (ZH)**: 语言模型的独立性检验 

**Authors**: Sally Zhu, Ahmed Ahmed, Rohith Kuditipudi, Percy Liang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12292)  

**Abstract**: We consider the following problem: given the weights of two models, can we test whether they were trained independently -- i.e., from independent random initializations? We consider two settings: constrained and unconstrained. In the constrained setting, we make assumptions about model architecture and training and propose a family of statistical tests that yield exact p-values with respect to the null hypothesis that the models are trained from independent random initializations. These p-values are valid regardless of the composition of either model's training data; we compute them by simulating exchangeable copies of each model under our assumptions and comparing various similarity measures of weights and activations between the original two models versus these copies. We report the p-values from these tests on pairs of 21 open-weight models (210 total pairs) and correctly identify all pairs of non-independent models. Our tests remain effective even if one model was fine-tuned for many tokens. In the unconstrained setting, where we make no assumptions about training procedures, can change model architecture, and allow for adversarial evasion attacks, the previous tests no longer work. Instead, we propose a new test which matches hidden activations between two models, and which is robust to adversarial transformations and to changes in model architecture. The test can also do localized testing: identifying specific non-independent components of models. Though we no longer obtain exact p-values from this, empirically we find it behaves as one and reliably identifies non-independent models. Notably, we can use the test to identify specific parts of one model that are derived from another (e.g., how Llama 3.1-8B was pruned to initialize Llama 3.2-3B, or shared layers between Mistral-7B and StripedHyena-7B), and it is even robust to retraining individual layers of either model from scratch. 

**Abstract (ZH)**: 我们考虑以下问题：给定两个模型的权重，我们能否测试这些模型是否是独立训练的，即是否来自独立的随机初始化？我们将考虑两种情形：受限和未受限。在受限情形下，我们假设模型架构和训练过程，并提出一系列统计测试方法，这些测试方法可以针对零假设（模型是来自独立随机初始化进行训练）提供精确的P值。无论任一模型的训练数据构成如何，这些P值都是有效的。我们通过在假设条件下模拟每个模型的可交换副本，并比较原模型和这些副本之间的权重和激活的相似度度量来计算这些P值。我们报告了在21个开放权重模型（总共210对）之间进行这些测试的结果，并成功识别出所有非独立训练的模型对。即使其中一个模型经过了大量的微调，我们的测试仍然有效。在未受限情形下，我们不对训练过程作出假设，可以更改模型架构，并允许对抗性规避攻击。在这一情形下，之前的测试不再有效。相反，我们提出了一种新的测试方法，该方法能够在两个模型之间匹配隐藏激活，并且该测试对对抗性变换和模型架构变化具有鲁棒性。此外，测试还能进行局部测试，识别模型中的特定非独立组件。尽管我们不再能从这种方法中获得精确的P值，但实验证明该方法行为类似并能可靠地识别出非独立训练的模型。值得注意的是，我们可以使用该测试识别一个模型中来自另一个模型的具体部分（例如，如何对Llama 3.1-8B进行剪枝以初始化Llama 3.2-3B，或者Mistral-7B和StripedHyena-7B之间的共享层），并且该测试即使在从头重新训练两个模型的个别层后也能保持鲁棒性。 

---
# Integrating Expert Knowledge into Logical Programs via LLMs 

**Title (ZH)**: 通过大语言模型将专家知识集成到逻辑程序中 

**Authors**: Franciszek Górski, Oskar Wysocki, Marco Valentino, Andre Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2502.12275)  

**Abstract**: This paper introduces ExKLoP, a novel framework designed to evaluate how effectively Large Language Models (LLMs) integrate expert knowledge into logical reasoning systems. This capability is especially valuable in engineering, where expert knowledge-such as manufacturer-recommended operational ranges-can be directly embedded into automated monitoring systems. By mirroring expert verification steps, tasks like range checking and constraint validation help ensure system safety and reliability. Our approach systematically evaluates LLM-generated logical rules, assessing both syntactic fluency and logical correctness in these critical validation tasks. We also explore the models capacity for self-correction via an iterative feedback loop based on code execution outcomes. ExKLoP presents an extensible dataset comprising 130 engineering premises, 950 prompts, and corresponding validation points. It enables comprehensive benchmarking while allowing control over task complexity and scalability of experiments. We leverage the synthetic data creation methodology to conduct extensive empirical evaluation on a diverse set of LLMs including Llama3, Gemma, Mixtral, Mistral, and Qwen. Results reveal that while models generate nearly perfect syntactically correct code, they frequently exhibit logical errors in translating expert knowledge. Furthermore, iterative self-correction yields only marginal improvements (up to 3%). Overall, ExKLoP serves as a robust evaluation platform that streamlines the selection of effective models for self-correcting systems while clearly delineating the types of errors encountered. The complete implementation, along with all relevant data, is available at GitHub. 

**Abstract (ZH)**: 本文引入了ExKLoP，一种 novel的框架，旨在评估大规模语言模型（LLMs）如何有效地将专家知识整合到逻辑推理系统中。这种能力在工程领域尤其有价值，因为专家知识，例如制造商推荐的操作范围，可以直接嵌入到自动监控系统中。通过镜像专家验证步骤，任务如范围检查和约束验证可确保系统的安全性和可靠性。我们的方法系统地评估了LLM生成的逻辑规则，不仅评估其语义流畅性，还评估其逻辑正确性。同时，我们探讨了模型通过基于代码执行结果的迭代反馈环进行自我纠正的能力。ExKLoP 提供了一个可扩展的数据集，包含130个工程前提、950个提示及其相应的验证点。它使基准测试得以全面进行，并允许控制任务复杂度和实验的可扩展性。我们利用合成数据创建方法对包括Llama3、Gemma、Mixtral、Mistral和Qwen在内的多种不同的LLM进行了广泛的实证评估。结果显示，尽管模型生成的代码几乎均为语法正确，但它们在将专家知识转换为逻辑规则时经常出现错误。此外，迭代自我纠正仅带来了微小的改进（最多3%）。总体而言，ExKLoP 成为一个稳健的评估平台，有助于简化选择具有自我纠正能力的有效模型的过程，并明确识别遇到的错误类型。完整的实现以及所有相关数据均在GitHub上提供。 

---
# Learning to Reason at the Frontier of Learnability 

**Title (ZH)**: 在可学习性的前沿进行推理学习 

**Authors**: Thomas Foster, Jakob Foerster  

**Link**: [PDF](https://arxiv.org/pdf/2502.12272)  

**Abstract**: Reinforcement learning is now widely adopted as the final stage of large language model training, especially for reasoning-style tasks such as maths problems. Typically, models attempt each question many times during a single training step and attempt to learn from their successes and failures. However, we demonstrate that throughout training with two popular algorithms (PPO and VinePPO) on two widely used datasets, many questions are either solved by all attempts - meaning they are already learned - or by none - providing no meaningful training signal. To address this, we adapt a method from the reinforcement learning literature - sampling for learnability - and apply it to the reinforcement learning stage of LLM training. Our curriculum prioritises questions with high variance of success, i.e. those where the agent sometimes succeeds, but not always. Our findings demonstrate that this curriculum consistently boosts training performance across multiple algorithms and datasets, paving the way for more efficient and effective reinforcement learning in LLMs. 

**Abstract (ZH)**: 强化学习现在被广泛用于大型语言模型训练的最终阶段，特别是在解决数学问题等推理型任务中。通常，模型会在一个训练步骤中尝试多次回答每个问题，并试图从成功和失败中学习。然而，我们证明，在使用两种流行算法（PPO和VinePPO）和两个广泛使用的数据集进行训练的过程中，许多问题要么在所有尝试中都被解决，意味着这些问题是已经学会的；要么在所有尝试中都无法解决，无法提供有意义的训练信号。为解决这一问题，我们借鉴强化学习文献中的一个方法——探索可学习性，并将其应用于大型语言模型训练的强化学习阶段。我们的课程学习策略优先考虑成功率具有高差异性的问题，即那些有时会成功但并非总是成功的问题。我们的研究结果表明，这种方法能够跨多个算法和数据集一致地提升训练性能，为大型语言模型中的强化学习更高效和有效地发展铺平了道路。 

---
# Optimal Brain Iterative Merging: Mitigating Interference in LLM Merging 

**Title (ZH)**: 最优大脑迭代合并：减轻大模型合并中的干扰 

**Authors**: Zhixiang Wang, Zhenyu Mao, Yixuan Qiao, Yunfang Wu, Biye Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.12217)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities, but their high computational costs pose challenges for customization. Model merging offers a cost-effective alternative, yet existing methods suffer from interference among parameters, leading to performance degradation. In this work, we propose Optimal Brain Iterative Merging (OBIM), a novel method designed to mitigate both intra-model and inter-model interference. OBIM consists of two key components: (1) A saliency measurement mechanism that evaluates parameter importance based on loss changes induced by individual weight alterations, reducing intra-model interference by preserving only high-saliency parameters. (2) A mutually exclusive iterative merging framework, which incrementally integrates models using a binary mask to avoid direct parameter averaging, thereby mitigating inter-model interference. We validate OBIM through experiments on both Supervised Fine-Tuned (SFT) models and post-pretrained checkpoints. The results show that OBIM significantly outperforms existing merging techniques. Overall, OBIM provides an effective and practical solution for enhancing LLM merging. 

**Abstract (ZH)**: 大型语言模型（LLMs）已经展示了令人印象深刻的性能，但它们高昂的计算成本限制了定制化的可能性。模型合并提供了一种成本效益较高的替代方案，然而现有的方法在融合过程中存在参数间的干扰，导致性能下降。为了解决这一问题，我们提出了一种名为Optimal Brain Iterative Merging (OBIM)的新颖方法，旨在减轻模型内和模型间的影响。OBIM包括两个关键组成部分：（1）一种显著性度量机制，该机制基于单个权重改变引发的损失变化来评估参数的重要性，并仅保留高显著性的参数来减少模型内干扰；（2）一种相互排斥的迭代合并框架，该框架通过使用二元掩模逐步整合模型，避免直接进行参数平均，从而减少模型间的干扰。我们通过在有监督微调（SFT）模型和后微调检查点上的实验验证了OBIM的有效性。结果显示，OBIM在性能上显著优于现有的合并技术。总体而言，OBIM提供了一种有效且实用的解决方案来增强LLM合并。 

---
# Tactic: Adaptive Sparse Attention with Clustering and Distribution Fitting for Long-Context LLMs 

**Title (ZH)**: 标题：自适应稀疏注意力机制：结合聚类和分布拟合的长上下文语言模型策略 

**Authors**: Kan Zhu, Tian Tang, Qinyu Xu, Yile Gu, Zhichen Zeng, Rohan Kadekodi, Liangyu Zhao, Ang Li, Arvind Krishnamurthy, Baris Kasikci  

**Link**: [PDF](https://arxiv.org/pdf/2502.12216)  

**Abstract**: Long-context models are essential for many applications but face inefficiencies in loading large KV caches during decoding. Prior methods enforce fixed token budgets for sparse attention, assuming a set number of tokens can approximate full attention. However, these methods overlook variations in the importance of attention across heads, layers, and contexts. To address these limitations, we propose Tactic, a sparsity-adaptive and calibration-free sparse attention mechanism that dynamically selects tokens based on their cumulative attention scores rather than a fixed token budget. By setting a target fraction of total attention scores, Tactic ensures that token selection naturally adapts to variations in attention sparsity. To efficiently approximate this selection, Tactic leverages clustering-based sorting and distribution fitting, allowing it to accurately estimate token importance with minimal computational overhead. We show that Tactic outperforms existing sparse attention algorithms, achieving superior accuracy and up to 7.29x decode attention speedup. This improvement translates to an overall 1.58x end-to-end inference speedup, making Tactic a practical and effective solution for long-context LLM inference in accuracy-sensitive applications. 

**Abstract (ZH)**: 长期上下文模型在许多应用中都非常重要，但在解码过程中加载大规模的键值缓存时面临效率问题。现有方法强制执行稀疏注意机制的固定标记预算，假设一定数量的标记可以近似全注意力。然而，这些方法忽略了注意在不同头、层和上下文之间的重要性变化。为了解决这些限制，我们提出了Tactic，这是一种自适应稀疏注意力机制，能够在不进行校准的情况下动态选择标记，基于标记的累积注意力分数而非固定标记预算来进行选择。通过设定总注意力分数的目标比例，Tactic 确保标记选择自然地适应于注意力稀疏性的变化。为了高效地近似这种选择，Tactic 利用基于聚类的排序和分布拟合，能够以最小的计算开销准确估计标记的重要性。实验表明，Tactic 在稀疏注意力算法中表现出色，实现了更高的准确性和高达7.29倍的解码注意力加速。这一改进整体上实现了1.58倍的端到端推理加速，使得Tactic 成为在准确性敏感应用中进行长期上下文LLM推理的实用而有效的解决方案。 

---
# Evaluating the Paperclip Maximizer: Are RL-Based Language Models More Likely to Pursue Instrumental Goals? 

**Title (ZH)**: 评估夹子最大化者：基于强化学习的语言模型更有可能追求工具性目标吗？ 

**Authors**: Yufei He, Yuexin Li, Jiaying Wu, Yuan Sui, Yulin Chen, Bryan Hooi  

**Link**: [PDF](https://arxiv.org/pdf/2502.12206)  

**Abstract**: As large language models (LLMs) continue to evolve, ensuring their alignment with human goals and values remains a pressing challenge. A key concern is \textit{instrumental convergence}, where an AI system, in optimizing for a given objective, develops unintended intermediate goals that override the ultimate objective and deviate from human-intended goals. This issue is particularly relevant in reinforcement learning (RL)-trained models, which can generate creative but unintended strategies to maximize rewards. In this paper, we explore instrumental convergence in LLMs by comparing models trained with direct RL optimization (e.g., the o1 model) to those trained with reinforcement learning from human feedback (RLHF). We hypothesize that RL-driven models exhibit a stronger tendency for instrumental convergence due to their optimization of goal-directed behavior in ways that may misalign with human intentions. To assess this, we introduce InstrumentalEval, a benchmark for evaluating instrumental convergence in RL-trained LLMs. Initial experiments reveal cases where a model tasked with making money unexpectedly pursues instrumental objectives, such as self-replication, implying signs of instrumental convergence. Our findings contribute to a deeper understanding of alignment challenges in AI systems and the risks posed by unintended model behaviors. 

**Abstract (ZH)**: 随着大规模语言模型（LLMs）的不断发展，确保它们与人类目标和价值观保持一致仍然是一个紧迫的挑战。一个关键问题在于\textit{工具性趋同}，即一个AI系统在优化特定目标时，会发展出一些未预见的中间目标，这些目标会超越最终目标并偏离人类预期的意图。这一问题特别适用于通过强化学习（RL）进行训练的模型，这些模型可能会产生创造性的但却未预见的策略来最大化奖励。在本文中，我们通过将通过直接RL优化（例如，o1模型）训练的模型与通过人类反馈强化学习（RLHF）训练的模型进行比较，来探讨工具性趋同在LLMs中的表现。我们假设，由RL驱动的模型更容易表现出工具性趋同的倾向，因为它们以可能与人类意图不符的方式优化目标导向行为。为了评估这一问题，我们引入了InstrumentalEval，一个用于评估RL训练的LLMs中工具性趋同的标准。初步实验显示，在一项旨在赚钱的任务中，模型意外地追求某种工具性目标，如自我复制，这表明存在工具性趋同的现象。我们的研究结果加深了对AI系统中的对齐挑战及其潜在意外行为风险的理解。 

---
# Identifiable Steering via Sparse Autoencoding of Multi-Concept Shifts 

**Title (ZH)**: 基于多概念变化稀疏自编码的可识别操控方法 

**Authors**: Shruti Joshi, Andrea Dittadi, Sébastien Lachapelle, Dhanya Sridhar  

**Link**: [PDF](https://arxiv.org/pdf/2502.12179)  

**Abstract**: Steering methods manipulate the representations of large language models (LLMs) to induce responses that have desired properties, e.g., truthfulness, offering a promising approach for LLM alignment without the need for fine-tuning. Traditionally, steering has relied on supervision, such as from contrastive pairs of prompts that vary in a single target concept, which is costly to obtain and limits the speed of steering research. An appealing alternative is to use unsupervised approaches such as sparse autoencoders (SAEs) to map LLM embeddings to sparse representations that capture human-interpretable concepts. However, without further assumptions, SAEs may not be identifiable: they could learn latent dimensions that entangle multiple concepts, leading to unintentional steering of unrelated properties. We introduce Sparse Shift Autoencoders (SSAEs) that instead map the differences between embeddings to sparse representations. Crucially, we show that SSAEs are identifiable from paired observations that vary in \textit{multiple unknown concepts}, leading to accurate steering of single concepts without the need for supervision. We empirically demonstrate accurate steering across semi-synthetic and real-world language datasets using Llama-3.1 embeddings. 

**Abstract (ZH)**: 引导方法通过操纵大型语言模型（LLM）的表示以产生具有期望属性的响应，例如真实性，为在不需要微调的情况下实现LLM对齐提供了有希望的方法。传统上，引导依赖于监督，例如来自单一目标概念不同的对比提示对，这成本较高且限制了引导研究的进度。一种有吸引力的替代方法是使用稀疏自编码器（SAEs）将LLM嵌入映射到稀疏表示，这些稀疏表示捕捉了人类可解释的概念。然而，在没有进一步假设的情况下，SAEs可能不具备可识别性：它们可能会学习与其他概念纠缠在一起的潜在维度，导致对无关属性的意外引导。我们引入了稀疏平移自编码器（SSAEs），它通过将嵌入之间的差异映射到稀疏表示来工作。关键的是，我们展示了当配对观察在多个未知概念上发生变化时，SSAEs是可识别的，从而能够准确地引导单个概念而无需监督。我们使用Llama-3.1嵌入在半合成和真实世界语言数据集上实证演示了准确的引导。 

---
# GoRA: Gradient-driven Adaptive Low Rank Adaptation 

**Title (ZH)**: GoRA：梯度驱动的自适应低秩适应 

**Authors**: Haonan He, Peng Ye, Yuchen Ren, Yuan Yuan, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.12171)  

**Abstract**: Low-Rank Adaptation (LoRA) is a crucial method for efficiently fine-tuning pretrained large language models (LLMs), with its performance largely influenced by two key factors: rank and initialization strategy. Numerous LoRA variants have been proposed to enhance its performance by addressing these factors. However, these variants often compromise LoRA's usability or efficiency. In this paper, we analyze the fundamental limitations of existing methods and introduce a novel approach, GoRA (Gradient-driven Adaptive Low Rank Adaptation), which adaptively assigns ranks and initializes weights for low-rank adapters simultaneously based on gradient information. Extensive experimental results demonstrate that GoRA significantly improves performance while preserving the high usability and efficiency of LoRA. On the T5 model fine-tuned for the GLUE benchmark, GoRA achieves a 5.88-point improvement over LoRA and slightly surpasses full fine-tuning. Similarly, on the Llama3.1-8B-Base model fine-tuned for GSM8k tasks, GoRA outperforms LoRA with a 5.13-point improvement and exceeds full fine-tuning in high-rank settings by a margin of 2.05 points. 

**Abstract (ZH)**: 低秩适应（LoRA）是一种高效微调预训练大规模语言模型（LLM）的关键方法，其性能主要受到两个关键因素的影响：秩和初始化策略。为了提高其性能，已经提出了大量LoRA的变体，但这些变体往往在提高性能的同时牺牲了LoRA的实用性和效率。本文分析了现有方法的根本局限性，并提出了一种新颖的方法——GoRA（梯度驱动的自适应低秩适应），该方法根据梯度信息同时自适应地分配秩和初始化低秩适配器的权重。大量的实验结果表明，GoRA在保持LoRA高实用性和效率的同时，显著提高了性能。在为GLUE基准数据集微调的T5模型上，GoRA在LoRA的基础上实现了5.88点的性能提升，并且在部分高秩设置中轻微超过了完全微调。同样地，在为GSM8k任务微调的Llama3.1-8B-Base模型上，GoRA在LoRA的基础上实现了5.13点的性能提升，并且在高秩设置中超过了完全微调，领先幅度为2.05点。 

---
# MUDDFormer: Breaking Residual Bottlenecks in Transformers via Multiway Dynamic Dense Connections 

**Title (ZH)**: MUDDFormer: 通过多维动态密集连接打破变压器中的残差瓶颈 

**Authors**: Da Xiao, Qingye Meng, Shengping Li, Xingyuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2502.12170)  

**Abstract**: We propose MUltiway Dynamic Dense (MUDD) connections, a simple yet effective method to address the limitations of residual connections and enhance cross-layer information flow in Transformers. Unlike existing dense connection approaches with static and shared connection weights, MUDD generates connection weights dynamically depending on hidden states at each sequence position and for each decoupled input stream (the query, key, value or residual) of a Transformer block. MUDD connections can be seamlessly integrated into any Transformer architecture to create MUDDFormer. Extensive experiments show that MUDDFormer significantly outperforms Transformers across various model architectures and scales in language modeling, achieving the performance of Transformers trained with 1.8X-2.4X compute. Notably, MUDDPythia-2.8B matches Pythia-6.9B in pretraining ppl and downstream tasks and even rivals Pythia-12B in five-shot settings, while adding only 0.23% parameters and 0.4% computation. Code in JAX and PyTorch and pre-trained models are available at this https URL . 

**Abstract (ZH)**: 我们提出了一种名为MUltiway Dynamic Dense (MUDD) 连接的方法，这是一种简单而有效的解决残差连接局限性并增强Transformer中跨层信息流动的方法。与现有的具有静态和共享连接权重的密集连接方法不同，MUDD能够根据Transformer块中每个序列位置和每个独立输入流（查询、键、值或残差）的隐藏状态动态生成连接权重。MUDD连接可以无缝集成到任何Transformer架构中，创建MUDDFormer。广泛的经验表明，MUDDFormer在各种模型架构和规模的自然语言处理中显著优于Transformer，实现的性能相当于使用1.8倍至2.4倍计算资源训练的Transformer。值得注意的是，在预训练语言模型ppl和下游任务方面，MUDDPythia-2.8B仅需增加0.23%的参数和0.4%的计算量就能与Pythia-6.9B相媲美，并且在五 shot 设置中甚至能够与Pythia-12B相抗衡。JAX和PyTorch代码及预训练模型可在该网址获取。 

---
# Causal Interpretations in Observational Studies: The Role of Sociocultural Backgrounds and Team Dynamics 

**Title (ZH)**: 观察研究中的因果解释：社会文化背景和团队动态的作用 

**Authors**: Jun Wang, Bei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12159)  

**Abstract**: The prevalence of drawing causal conclusions from observational studies has raised concerns about potential exaggeration in science communication. While some believe causal language should only apply to randomized controlled trials, others argue that rigorous methods can justify causal claims in observational studies. Ideally, causal language should align with the strength of the evidence. However, through the analysis of over 80,000 observational study abstracts using computational linguistic and regression methods, we found that causal language is more frequently used by less experienced authors, smaller research teams, male last authors, and authors from countries with higher uncertainty avoidance indices. These findings suggest that the use of causal language may be influenced by external factors such as the sociocultural backgrounds of authors and the dynamics of research collaboration. This newly identified link deepens our understanding of how such factors help shape scientific conclusions in causal inference and science communication. 

**Abstract (ZH)**: 从观察性研究中得出因果结论的倾向引发了对科学传播中潜在夸大性的问题的担忧。有人认为因果语言只应适用于随机对照试验，而另一些人则认为严谨的方法可以在观察性研究中支持因果声明。理想情况下，因果语言应与证据强度相匹配。然而，通过对超过80,000篇观察性研究摘要进行计算语义学和回归分析，我们发现因果语言更频繁地被经验不足的作者、较小的研究团队、最后署名的男性以及社会发展回避指数较高的国家的研究人员使用。这些发现表明，因果语言的使用可能受作者的社会文化背景和研究合作动态等外部因素的影响。这一新识别的联系加深了我们对这些因素如何影响因果推理和科学传播中结论形成的理解。 

---
# Mining Social Determinants of Health for Heart Failure Patient 30-Day Readmission via Large Language Model 

**Title (ZH)**: 通过大型语言模型挖掘心脏衰竭患者30天再入院的社会决定因素 

**Authors**: Mingchen Shao, Youjeong Kang, Xiao Hu, Hyunjung Gloria Kwak, Carl Yang, Jiaying Lu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12158)  

**Abstract**: Heart Failure (HF) affects millions of Americans and leads to high readmission rates, posing significant healthcare challenges. While Social Determinants of Health (SDOH) such as socioeconomic status and housing stability play critical roles in health outcomes, they are often underrepresented in structured EHRs and hidden in unstructured clinical notes. This study leverages advanced large language models (LLMs) to extract SDOHs from clinical text and uses logistic regression to analyze their association with HF readmissions. By identifying key SDOHs (e.g. tobacco usage, limited transportation) linked to readmission risk, this work also offers actionable insights for reducing readmissions and improving patient care. 

**Abstract (ZH)**: 心力衰竭（HF）影响着数百万的美国人，并导致高再入院率，给医疗卫生系统带来了重大挑战。虽然社会决定因素（SDOH）如社会经济状况和住房稳定性在健康结果中扮演着重要角色，但在结构化的电子病历（EHR）中这些因素往往未被充分代表，并且在非结构化的临床笔记中被隐藏。本研究利用先进的大型语言模型（LLMs）从临床文本中提取SDOH，并使用逻辑回归分析这些因素与HF再入院之间的关联。通过识别与再入院风险相关的关键SDOH（如吸烟、交通不便等），本文还提供了一些可用于减少再入院并改善患者护理的行动指南。 

---
