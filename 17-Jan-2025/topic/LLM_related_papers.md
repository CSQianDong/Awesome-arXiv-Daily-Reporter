# Evaluating Conversational Recommender Systems with Large Language Models: A User-Centric Evaluation Framework 

**Title (ZH)**: 用大型语言模型评估对话推荐系统：一种以用户为中心的评估框架 

**Authors**: Nuo Chen, Quanyu Dai, Xiaoyu Dong, Xiao-Ming Wu, Zhenhua Dong  

**Link**: [PDF](https://arxiv.org/pdf/2501.09493)  

**Abstract**: Conversational recommender systems (CRS) involve both recommendation and dialogue tasks, which makes their evaluation a unique challenge. Although past research has analyzed various factors that may affect user satisfaction with CRS interactions from the perspective of user studies, few evaluation metrics for CRS have been proposed. Recent studies have shown that LLMs can align with human preferences, and several LLM-based text quality evaluation measures have been introduced. However, the application of LLMs in CRS evaluation remains relatively limited. To address this research gap and advance the development of user-centric conversational recommender systems, this study proposes an automated LLM-based CRS evaluation framework, building upon existing research in human-computer interaction and psychology. The framework evaluates CRS from four dimensions: dialogue behavior, language expression, recommendation items, and response content. We use this framework to evaluate four different conversational recommender systems. 

**Abstract (ZH)**: 会话推荐系统（CRS）既涉及推荐任务，也涉及对话任务，这使得对其的评估成为一项独特的挑战。尽管过往的研究从用户研究的角度分析了可能影响用户对CRS交互满意度的各种因素，但针对CRS的评价指标却很少提出。最近的研究表明，大型语言模型（LLMs）能够与人类偏好对齐，并且已经引入了多种基于LLM的文本质量评估指标。然而，LLM在CRS评估中的应用仍然相对有限。为了解决这一研究缺口并促进以用户为中心的会话推荐系统的开发，本研究提出了一种基于LLM的自动化CRS评估框架，该框架借鉴了人机交互和心理学领域的现有研究成果。该框架从四个维度评估CRS：对话行为、语言表达、推荐项目和响应内容。我们使用该框架评估了四种不同的会话推荐系统。 

---
# Guiding Retrieval using LLM-based Listwise Rankers 

**Title (ZH)**: 使用基于LLM的列表式排名器指导检索 

**Authors**: Mandeep Rathee, Sean MacAvaney, Avishek Anand  

**Link**: [PDF](https://arxiv.org/pdf/2501.09186)  

**Abstract**: Large Language Models (LLMs) have shown strong promise as rerankers, especially in ``listwise'' settings where an LLM is prompted to rerank several search results at once. However, this ``cascading'' retrieve-and-rerank approach is limited by the bounded recall problem: relevant documents not retrieved initially are permanently excluded from the final ranking. Adaptive retrieval techniques address this problem, but do not work with listwise rerankers because they assume a document's score is computed independently from other documents. In this paper, we propose an adaptation of an existing adaptive retrieval method that supports the listwise setting and helps guide the retrieval process itself (thereby overcoming the bounded recall problem for LLM rerankers). Specifically, our proposed algorithm merges results both from the initial ranking and feedback documents provided by the most relevant documents seen up to that point. Through extensive experiments across diverse LLM rerankers, first stage retrievers, and feedback sources, we demonstrate that our method can improve nDCG@10 by up to 13.23% and recall by 28.02%--all while keeping the total number of LLM inferences constant and overheads due to the adaptive process minimal. The work opens the door to leveraging LLM-based search in settings where the initial pool of results is limited, e.g., by legacy systems, or by the cost of deploying a semantic first-stage. 

**Abstract (ZH)**: 大语言模型（LLM）在重新排序方面表现出强大的潜力，尤其是在“列表级”设置中，即LLM被提示一次性对多个搜索结果进行重新排序。然而，这种“级联”检索和重新排序的方法受到“回忆限制”问题的限制：未在最初检索到的相关文档将被永久排除在最终排名之外。自适应检索技术可以解决这一问题，但它们不能与列表级重新排序器一起使用，因为它们假设一个文档的分数与其他文档独立计算。本文中，我们提出了一种现有自适应检索方法的改进版本，该版本适用于列表级设置，并有助于引导检索过程本身（从而克服LLM重新排序器的回忆限制问题）。具体来说，我们提出的算法将初始排序结果与迄今为止最相关的文档提供的反馈文档结果合并。通过在各种LLM重新排序器、第一阶段检索器和反馈源上的广泛实验，我们证明了我们的方法可以在保持LLM推断次数不变的同时，将nDCG@10提高多达13.23%，召回率提高28.02%，并且自适应过程导致的额外开销极少。这项工作为在初始结果池受限的场景下利用基于LLM的搜索打开了大门，例如由传统系统限制或部署语义第一阶段的成本限制的场景。 

---
# Enhancing Lexicon-Based Text Embeddings with Large Language Models 

**Title (ZH)**: 使用大型语言模型增强基于词典的文本嵌入 

**Authors**: Yibin Lei, Tao Shen, Yu Cao, Andrew Yates  

**Link**: [PDF](https://arxiv.org/pdf/2501.09749)  

**Abstract**: Recent large language models (LLMs) have demonstrated exceptional performance on general-purpose text embedding tasks. While dense embeddings have dominated related research, we introduce the first Lexicon-based EmbeddiNgS (LENS) leveraging LLMs that achieve competitive performance on these tasks. Regarding the inherent tokenization redundancy issue and unidirectional attention limitations in traditional causal LLMs, LENS consolidates the vocabulary space through token embedding clustering, and investigates bidirectional attention and various pooling strategies. Specifically, LENS simplifies lexicon matching by assigning each dimension to a specific token cluster, where semantically similar tokens are grouped together, and unlocking the full potential of LLMs through bidirectional attention. Extensive experiments demonstrate that LENS outperforms dense embeddings on the Massive Text Embedding Benchmark (MTEB), delivering compact feature representations that match the sizes of dense counterparts. Notably, combining LENSE with dense embeddings achieves state-of-the-art performance on the retrieval subset of MTEB (i.e. BEIR). 

**Abstract (ZH)**: 近期的大规模语言模型（LLMs）在通用文本嵌入任务中展现了卓越的表现。尽管密集嵌入在过去相关研究中占主导地位，我们首次引入了基于词典的嵌入（Lexicon-based EmbeddiNgS，LENS），并通过LLMs实现了这些任务的竞争力。针对传统因果LLMs中固有的 token 化冗余问题和单向注意限制，LENS 通过 token 嵌入聚类缩小了词典空间，并研究了双向注意和各种池化策略。具体而言，LENS 通过将每个维度分配给特定的 token 聚类来简化词典匹配，使 semantically 相似的 token 被分组在一起，并通过双向注意充分发挥了LLMs的潜力。大规模文本嵌入基准测试（MTEB）的广泛实验表明，LENS 在密集嵌入上表现出优越性，提供了与密集嵌入相当大小的紧凑特征表示。值得注意的是，将LENS与密集嵌入结合起来，在MTEB的检索子集（即BEIR）上达到了最先进的性能。 

---
# Evaluating LLM Abilities to Understand Tabular Electronic Health Records: A Comprehensive Study of Patient Data Extraction and Retrieval 

**Title (ZH)**: 评估大型语言模型理解电子健康记录表的能力：关于患者数据提取与检索的全面研究 

**Authors**: Jesus Lovon, Martin Mouysset, Jo Oleiwan, Jose G. Moreno, Christine Damase-Michel, Lynda Tamine  

**Link**: [PDF](https://arxiv.org/pdf/2501.09384)  

**Abstract**: Electronic Health Record (EHR) tables pose unique challenges among which is the presence of hidden contextual dependencies between medical features with a high level of data dimensionality and sparsity. This study presents the first investigation into the abilities of LLMs to comprehend EHRs for patient data extraction and retrieval. We conduct extensive experiments using the MIMICSQL dataset to explore the impact of the prompt structure, instruction, context, and demonstration, of two backbone LLMs, Llama2 and Meditron, based on task performance. Through quantitative and qualitative analyses, our findings show that optimal feature selection and serialization methods can enhance task performance by up to 26.79% compared to naive approaches. Similarly, in-context learning setups with relevant example selection improve data extraction performance by 5.95%. Based on our study findings, we propose guidelines that we believe would help the design of LLM-based models to support health search. 

**Abstract (ZH)**: 电子健康记录（EHR）表在数据维度和稀疏性方面具有较高的特征，其中隐藏的上下文依赖关系是一个独特挑战。本研究首次探讨了大型语言模型（LLMs）在理解EHR以提取和检索病人数据方面的能力。我们使用MIMICSQL数据集进行大量的实验，以研究两种主干模型——Llama2和Meditron——在基于任务性能的不同提示结构、指令、上下文和示范的影响。通过定量和定性分析，我们的研究发现，最优特征选择和序列化方法可以将任务性能提高多达26.79%，相较于朴素方法。同样，具有相关示例选择的上下文学习设置可以将数据提取性能提高5.95%。基于我们的研究结果，我们提出了一些建议，我们相信这些建议将有助于设计支持健康搜索的LLM基模型。 

---
# To Retrieve or Not to Retrieve? Uncertainty Detection for Dynamic Retrieval Augmented Generation 

**Title (ZH)**: 是取用还是不取用？动态检索增强生成中的不确定性检测 

**Authors**: Kaustubh D. Dhole  

**Link**: [PDF](https://arxiv.org/pdf/2501.09292)  

**Abstract**: Retrieval-Augmented Generation equips large language models with the capability to retrieve external knowledge, thereby mitigating hallucinations by incorporating information beyond the model's intrinsic abilities. However, most prior works have focused on invoking retrieval deterministically, which makes it unsuitable for tasks such as long-form question answering. Instead, dynamically performing retrieval by invoking it only when the underlying LLM lacks the required knowledge can be more efficient. In this context, we delve deeper into the question, "To Retrieve or Not to Retrieve?" by exploring multiple uncertainty detection methods. We evaluate these methods for the task of long-form question answering, employing dynamic retrieval, and present our comparisons. Our findings suggest that uncertainty detection metrics, such as Degree Matrix Jaccard and Eccentricity, can reduce the number of retrieval calls by almost half, with only a slight reduction in question-answering accuracy. 

**Abstract (ZH)**: 检索增强生成使大规模语言模型具备检索外部知识的能力，从而通过结合模型内在能力之外的信息来减轻幻觉现象。然而，大多数前期工作都集中在以确定性方式触发检索上，这使其不适合长文本问答等任务。相反，仅在基础语言模型缺乏所需知识时动态地触发检索可能是更有效的。在此背景下，我们深入探讨了“检索还是不检索？”这一问题，探索了多种不确定性检测方法。我们使用动态检索对长文本问答任务进行了评估，并展示了我们的比较结果。我们的研究发现，诸如度矩阵雅卡尔距离和偏心率等不确定性检测指标，可以将检索调用次数减少近一半，同时仅轻微降低问答准确性。 

---
# OmniThink: Expanding Knowledge Boundaries in Machine Writing through Thinking 

**Title (ZH)**: OmniThink：通过思考扩展机器写作的知识边界 

**Authors**: Zekun Xi, Wenbiao Yin, Jizhan Fang, Jialong Wu, Runnan Fang, Ningyu Zhang, Jiang Yong, Pengjun Xie, Fei Huang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.09751)  

**Abstract**: Machine writing with large language models often relies on retrieval-augmented generation. However, these approaches remain confined within the boundaries of the model's predefined scope, limiting the generation of content with rich information. Specifically, vanilla-retrieved information tends to lack depth, utility, and suffers from redundancy, which negatively impacts the quality of generated articles, leading to shallow, repetitive, and unoriginal outputs. To address these issues, we propose OmniThink, a machine writing framework that emulates the human-like process of iterative expansion and reflection. The core idea behind OmniThink is to simulate the cognitive behavior of learners as they progressively deepen their knowledge of the topics. Experimental results demonstrate that OmniThink improves the knowledge density of generated articles without compromising metrics such as coherence and depth. Human evaluations and expert feedback further highlight the potential of OmniThink to address real-world challenges in the generation of long-form articles. 

**Abstract (ZH)**: 大规模语言模型进行机器写作往往依赖于检索增强生成。然而，这些方法仍然受限于模型预定义的范围之内，限制了丰富信息内容的生成。具体来说，直接检索的信息往往缺乏深度、实用性和易冗余，这会负面影响生成文章的质量，导致输出内容浅薄、重复且缺乏原创性。为了解决这些问题，我们提出了一种名为OmniThink的机器写作框架，该框架模拟了人类逐步深化知识掌握过程中的迭代扩展和反思机制。OmniThink的核心思想是模拟学习者在逐渐加深对主题理解过程中的认知行为。实验结果表明，OmniThink能够在不牺牲一致性和深度等指标的情况下，提高生成文章的知识密度。进一步的人类评估和专家反馈凸显了OmniThink在应对长篇文章生成中的现实挑战方面的潜力。 

---
# From Scarcity to Capability: Empowering Fake News Detection in Low-Resource Languages with LLMs 

**Title (ZH)**: 从稀缺到能力：利用大语言模型在低资源语言中赋能假新闻检测 

**Authors**: Hrithik Majumdar Shibu, Shrestha Datta, Md. Sumon Miah, Nasrullah Sami, Mahruba Sharmin Chowdhury, Md. Saiful Islam  

**Link**: [PDF](https://arxiv.org/pdf/2501.09604)  

**Abstract**: The rapid spread of fake news presents a significant global challenge, particularly in low-resource languages like Bangla, which lack adequate datasets and detection tools. Although manual fact-checking is accurate, it is expensive and slow to prevent the dissemination of fake news. Addressing this gap, we introduce BanFakeNews-2.0, a robust dataset to enhance Bangla fake news detection. This version includes 11,700 additional, meticulously curated fake news articles validated from credible sources, creating a proportional dataset of 47,000 authentic and 13,000 fake news items across 13 categories. In addition, we created a manually curated independent test set of 460 fake and 540 authentic news items for rigorous evaluation. We invest efforts in collecting fake news from credible sources and manually verified while preserving the linguistic richness. We develop a benchmark system utilizing transformer-based architectures, including fine-tuned Bidirectional Encoder Representations from Transformers variants (F1-87\%) and Large Language Models with Quantized Low-Rank Approximation (F1-89\%), that significantly outperforms traditional methods. BanFakeNews-2.0 offers a valuable resource to advance research and application in fake news detection for low-resourced languages. We publicly release our dataset and model on Github to foster research in this direction. 

**Abstract (ZH)**: 假新闻的迅速传播构成了一个重要的全球性挑战，尤其是在像孟加拉语这样的低资源语言中尤为突出，这些语言缺乏足够的数据集和检测工具。尽管人工事实核查很准确，但它成本高昂且速度慢，难以预防假新闻的传播。为应对这一缺口，我们介绍了BanFakeNews-2.0，这是一个增强孟加拉语假新闻检测的数据集。该版本新增了11,700篇经过仔细筛选和验证的假新闻文章，来自可信的来源，从而形成了一个同比例的数据集，包含47,000条真实新闻和13,000条假新闻，涵盖13个类别。此外，我们还创建了一个包含460条假新闻和540条真实新闻项目的独立手动筛选测试集，以进行严格的评估。我们在收集假新闻时，特别注重从可信来源获取，并进行人工验证，同时保持语言的丰富性。我们开发了一个基准系统，利用基于变换器的架构，包括微调双向编码器表示（F1-87%）的变换器变体和量化低秩近似的大语言模型（F1-89%），这些模型显著优于传统方法。BanFakeNews-2.0为低资源语言中的假新闻检测研究和应用提供了宝贵的资源。我们已在GitHub上公开发布了我们的数据集和模型，以促进该领域的研究。 

---
# Solving the unsolvable: Translating case law in Hong Kong 

**Title (ZH)**: 解决不可解之题：香港判例法的翻译 

**Authors**: King-kui Sin, Xi Xuan, Chunyu Kit, Clara Ho-yan Chan, Honic Ho-kin Ip  

**Link**: [PDF](https://arxiv.org/pdf/2501.09444)  

**Abstract**: This paper addresses the challenges translating case law under Hong Kong's bilingual legal system. It highlights the initial success of translating all written statutes into Chinese before the 1997 handover, a task mandated by the Basic Law. The effort involved significant collaboration among legal, linguistic, and translation experts, resulting in a comprehensive and culturally appropriate bilingual legal system. However, translating case law remains a significant challenge due to the sheer volume and continuous growth of judicial decisions. The paper critiques the governments and judiciarys sporadic and uncoordinated efforts to translate case law, contrasting it with the thorough approach previously taken for statute translation. Although the government acknowledges the importance of legal bilingualism, it lacks a sustainable strategy for translating case law. The Judiciarys position that translating all judgments is unnecessary, unrealistic, and not cost-effectiveis analyzed and critiqued for its impact on legal transparency and public trust. A proposed solution involves leveraging machine translation technology through a human-machine interactive translation platform, which undergoes two major transitions. Initially based on a neural model, the platform transitions to using a large language model for improved translation accuracy. Furthermore, it evolves from a single-agent system to a multi-agent system, incorporating Translator, Annotator, and Proofreader agents. This multi-agent approach, supported by a grant, aims to facilitate efficient, high-quality translation of judicial judgments by integrating advanced artificial intelligence and continuous feedback mechanisms, thus better meeting the needs of a bilingual legal system. 

**Abstract (ZH)**: 本文探讨了在港英文双语法律体系下进行案例法翻译所面临的挑战。在1997年主权移交前，所有书面法例均已成功翻译成中文，这是根据基本法的要求完成的任务。这项工作涉及法律、语言和翻译专家的密切合作，形成了全面且文化适应性的双语法律体系。然而，由于司法判决的数量庞大且持续增长，翻译案例法仍然是一项重大挑战。文章批评了政府和司法机构在翻译案例法方面反复无常且缺乏协调的努力，与早期对法例翻译的全面方法形成了对比。尽管政府承认法律双语化的重要性，但仍缺乏一种可持续的案例法翻译策略。司法机构认为，将所有判决翻译成中文是不必要的、不现实且不经济的做法，这一观点被分析并批评，认为这将对司法透明度和公众信任产生负面影响。本文提出的一项解决方案是通过人机交互翻译平台利用机器翻译技术，该平台经历了两大转变。最初基于神经模型，平台转向使用大型语言模型以提高翻译准确性。此外，它从单个代理系统演变为多代理系统，加入了翻译代理、注释代理和审校代理。在资助下，这种多代理方法旨在通过整合先进的人工智能技术和持续反馈机制，促进高质量的司法判决翻译，更好地满足双语法律体系的需求。 

---
# Perspective Transition of Large Language Models for Solving Subjective Tasks 

**Title (ZH)**: 大型语言模型视角转换在解决主观任务中的应用研究 

**Authors**: Xiaolong Wang, Yuanchi Zhang, Ziyue Wang, Yuzhuang Xu, Fuwen Luo, Yile Wang, Peng Li, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.09265)  

**Abstract**: Large language models (LLMs) have revolutionized the field of natural language processing, enabling remarkable progress in various tasks. Different from objective tasks such as commonsense reasoning and arithmetic question-answering, the performance of LLMs on subjective tasks is still limited, where the perspective on the specific problem plays crucial roles for better interpreting the context and giving proper response. For example, in certain scenarios, LLMs may perform better when answering from an expert role perspective, potentially eliciting their relevant domain knowledge. In contrast, in some scenarios, LLMs may provide more accurate responses when answering from a third-person standpoint, enabling a more comprehensive understanding of the problem and potentially mitigating inherent biases. In this paper, we propose Reasoning through Perspective Transition (RPT), a method based on in-context learning that enables LLMs to dynamically select among direct, role, and third-person perspectives for the best way to solve corresponding subjective problem. Through extensive experiments on totally 12 subjective tasks by using both closed-source and open-source LLMs including GPT-4, GPT-3.5, Llama-3, and Qwen-2, our method outperforms widely used single fixed perspective based methods such as chain-of-thought prompting and expert prompting, highlights the intricate ways that LLMs can adapt their perspectives to provide nuanced and contextually appropriate responses for different problems. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在自然语言处理领域引发了一场革命，推动了各种任务取得了显著的进步。与常识推理和算术问答等客观任务相比，LLMs 在处理主观任务时的表现仍然有限，特定问题的视角对于更好地解释上下文和给予适当回应起着关键作用。例如，在某些场景中，当LLMs 以专家视角回答问题时，可能会表现出更好的性能，潜在地利用其相关领域的知识。相反，在某些场景中，以第三方视角回答问题可能会使LLMs 提供更准确的答案，从而实现对问题的更全面理解，并有可能减轻固有的偏见。在这篇论文中，我们提出了一种基于上下文学习的方法——视角转换推理（RPT），该方法使LLMs 能够动态选择直接、角色和第三方视角，以最适合的方式解决相应的主观问题。通过在包括GPT-4、GPT-3.5、Llama-3 和 Qwen-2在内的12项主观任务上进行广泛的实验，我们的方法在使用闭源和开源LLMs时均优于广泛使用的基于单一固定视角的方法，如链式思维提示和专家提示。我们的研究突显了LLMs 如何灵活地调整视角以提供精细且上下文适配的回复，以应对不同问题的复杂方式。 

---
# Delayed Fusion: Integrating Large Language Models into First-Pass Decoding in End-to-end Speech Recognition 

**Title (ZH)**: 延迟融合：将大型语言模型集成到端到端语音识别的首轮解码中 

**Authors**: Takaaki Hori, Martin Kocour, Adnan Haider, Erik McDermott, Xiaodan Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2501.09258)  

**Abstract**: This paper presents an efficient decoding approach for end-to-end automatic speech recognition (E2E-ASR) with large language models (LLMs). Although shallow fusion is the most common approach to incorporate language models into E2E-ASR decoding, we face two practical problems with LLMs. (1) LLM inference is computationally costly. (2) There may be a vocabulary mismatch between the ASR model and the LLM. To resolve this mismatch, we need to retrain the ASR model and/or the LLM, which is at best time-consuming and in many cases not feasible. We propose "delayed fusion," which applies LLM scores to ASR hypotheses with a delay during decoding and enables easier use of pre-trained LLMs in ASR tasks. This method can reduce not only the number of hypotheses scored by the LLM but also the number of LLM inference calls. It also allows re-tokenizion of ASR hypotheses during decoding if ASR and LLM employ different tokenizations. We demonstrate that delayed fusion provides improved decoding speed and accuracy compared to shallow fusion and N-best rescoring using the LibriHeavy ASR corpus and three public LLMs, OpenLLaMA 3B & 7B and Mistral 7B. 

**Abstract (ZH)**: 本文提出了一个高效解码方法，用于端到端自动语音识别（E2E-ASR）系统，结合了大规模语言模型（LLMs）。尽管浅融合是最常见的将语言模型融入E2E-ASR解码的方法，但在使用LLMs时我们面临两个实际问题：(1) LLM推理计算开销大；(2) ASR模型与LLM之间可能存在词汇表不匹配。为了解决这一不匹配问题，我们需要重新训练ASR模型和/或LLM，这在最理想的情况下耗时长，而在许多情况下是不切实际的。我们提出了一种“延迟融合”的方法，在解码过程中以延迟的方式应用LLM分数，从而在ASR任务中更方便地使用预训练的LLMs。该方法不仅能减少需要LLM评分的假设数量，还能减少LLM推理调用次数。此外，该方法还允许在解码过程中进行ASR假设的重新分词，如果ASR和LLM使用不同的分词方式时。我们使用LibriHeavy ASR语料库以及三个公共LLM（OpenLLaMA 3B & 7B 和 Mistral 7B）证明了延迟融合相比浅融合和N-best评分具有更高的解码速度和准确率。 

---
# FineMedLM-o1: Enhancing the Medical Reasoning Ability of LLM from Supervised Fine-Tuning to Test-Time Training 

**Title (ZH)**: FineMedLM-o1：从监督微调到测试时训练，增强大语言模型的医学推理能力 

**Authors**: Hongzhou Yu, Tianhao Cheng, Ying Cheng, Rui Feng  

**Link**: [PDF](https://arxiv.org/pdf/2501.09213)  

**Abstract**: Recent advancements in large language models (LLMs) have shown promise in medical applications such as disease diagnosis and treatment planning. However, most existing medical LLMs struggle with the advanced reasoning required for complex clinical scenarios, such as differential diagnosis or personalized treatment suggestions. We proposed FineMedLM-o1, which leverages high-quality synthetic medical data and long-form reasoning data for Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO), enabling advanced dialogue and deep reasoning capabilities. Additionally, we introduced Test-Time Training (TTT) in the medical domain for the first time, facilitating domain adaptation and ensuring reliable, accurate reasoning. Experimental results demonstrate that FineMedLM-o1 achieves a 23% average performance improvement over prior models on key medical benchmarks. Furthermore, the introduction of TTT provides an additional 14% performance boost, highlighting its effectiveness in enhancing medical reasoning capabilities. To support this process, we also proposed a novel method for synthesizing medical dialogue. Compared to other open-source datasets, our dataset stands out as superior in both quality and complexity. The project and data will be released on GitHub. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）在医学应用方面取得了进展，如疾病诊断和治疗规划。然而，现有的大多数医学LLMs在处理复杂临床场景所需的高级推理方面存在不足，例如鉴别诊断或个性化治疗建议。我们提出了FineMedLM-o1，它利用高质量的合成医学数据和长篇推理数据进行监督微调（SFT）和直接偏好优化（DPO），从而实现高级对话和深入的推理能力。此外，我们首次在医学领域引入了测试时训练（TTT），促进领域适应并确保可靠的准确推理。实验结果表明，FineMedLM-o1在关键医学基准测试上的平均性能改进达到了23%，而TTT的引入又额外提供了14%的性能提升，突显其在增强医学推理能力方面的有效性。为了支持这一过程，我们还提出了一种新的合成医学对话方法。与现有的开源数据集相比，我们的数据集在质量和复杂性方面都更为优越。该项目和数据将发布在GitHub上。 

---
# The Veln(ia)s is in the Details: Evaluating LLM Judgment on Latvian and Lithuanian Short Answer Matching 

**Title (ZH)**: velnis(ia) 的细节决定一切：评估大规模语言模型对拉脱维亚语和立陶宛语文本简答题匹配判断的能力 

**Authors**: Yevhen Kostiuk, Oxana Vitman, Łukasz Gagała, Artur Kiulian  

**Link**: [PDF](https://arxiv.org/pdf/2501.09164)  

**Abstract**: In this work, we address the challenge of evaluating large language models (LLMs) on the short answer matching task for Latvian and Lithuanian languages. We introduce novel datasets consisting of 502 Latvian and 690 Lithuanian question-answer pairs. For each question-answer pair, we generated matched and non-matched answers using a set of alteration rules specifically designed to introduce small but meaningful changes in the text. These generated answers serve as test cases to assess the ability of LLMs to detect subtle differences in matching of the original answers. A subset of the datasets was manually verified for quality and accuracy. Our results show that while larger LLMs, such as QWEN2.5 72b and LLaMa3.1 70b, demonstrate near-perfect performance in distinguishing matched and non-matched answers, smaller models show more variance. For instance, LLaMa3.1 8b and EuroLLM 9b benefited from few-shot examples, while Mistral Nemo 12b underperformed on detection of subtle text alteration, particularly in Lithuanian, even with additional examples. QWEN2.5 7b and Mistral 7b were able to obtain a strong and comparable performance to the larger 70b models in zero and few shot experiments. Moreover, the performance of Mistral 7b was weaker in few shot experiments. 

**Abstract (ZH)**: 在本研究中，我们针对拉脱维亚语和立陶宛语的短答案匹配任务，探讨了评估大规模语言模型（LLMs）的挑战。我们引入了新的数据集，其中包含502个拉脱维亚语和690个立陶宛语的问题-答案对。对于每个问题-答案对，我们使用一组特定设计的修改规则生成了匹配和不匹配的答案，以引入细微但有意义的文本更改。这些生成的答案作为测试案例，用于评估LLMs检测原始答案细微差异的能力。部分数据集的高质量和准确性由人工验证。结果显示，尽管较大的LLMs（如QWEN2.5 72b和LLaMa3.1 70b）在区分匹配和不匹配的答案方面表现出近乎完美的性能，较小的模型则显示出更大的差异。例如，LLaMa3.1 8b和EuroLLM 9b受益于少量示例，而Mistral Nemo 12b在检测细微文本更改方面表现不佳，尤其是在立陶宛语中，即使有额外的示例也是如此。QWEN2.5 7b和Mistral 7b在零样本和少量样本实验中能够取得与较大的70b模型相当甚至更强的表现。此外，Mistral 7b在少量样本实验中的表现较弱。 

---
# Evaluating GenAI for Simplifying Texts for Education: Improving Accuracy and Consistency for Enhanced Readability 

**Title (ZH)**: 评估GenAI在教育中简化文本的应用：提高准确性和一致性以增强可读性 

**Authors**: Stephanie L. Day, Jacapo Cirica, Steven R. Clapp, Veronika Penkova, Amy E. Giroux, Abbey Banta, Catherine Bordeau, Poojitha Mutteneni, Ben D. Sawyer  

**Link**: [PDF](https://arxiv.org/pdf/2501.09158)  

**Abstract**: Generative artificial intelligence (GenAI) holds great promise as a tool to support personalized learning. Teachers need tools to efficiently and effectively enhance content readability of educational texts so that they are matched to individual students reading levels, while retaining key details. Large Language Models (LLMs) show potential to fill this need, but previous research notes multiple shortcomings in current approaches. In this study, we introduced a generalized approach and metrics for the systematic evaluation of the accuracy and consistency in which LLMs, prompting techniques, and a novel multi-agent architecture to simplify sixty informational reading passages, reducing each from the twelfth grade level down to the eighth, sixth, and fourth grade levels. We calculated the degree to which each LLM and prompting technique accurately achieved the targeted grade level for each passage, percentage change in word count, and consistency in maintaining keywords and key phrases (semantic similarity). One-sample t-tests and multiple regression models revealed significant differences in the best performing LLM and prompt technique for each of the four metrics. Both LLMs and prompting techniques demonstrated variable utility in grade level accuracy and consistency of keywords and key phrases when attempting to level content down to the fourth grade reading level. These results demonstrate the promise of the application of LLMs for efficient and precise automated text simplification, the shortcomings of current models and prompting methods in attaining an ideal balance across various evaluation criteria, and a generalizable method to evaluate future systems. 

**Abstract (ZH)**: 生成式人工智能（GenAI）作为个性化学习工具持有巨大潜力。教师需要工具来高效有效地提高教育文本的可读性，使其与学生个体的阅读水平相匹配，同时保留关键细节。大型语言模型（LLMs）显示出满足这一需求的潜力，但先前的研究指出当前方法存在多方面不足。在本研究中，我们引入了一种通用的方法和指标，用于系统地评估LLMs、提示技术以及一种新颖的多代理架构，这些方法和技术简化了六十篇信息性阅读段落，将每个段落的难度等级从高中降低到初中八年级、六年级和四年级。我们计算了每个LLM和提示技术在每篇段落中准确达到目标年级水平的程度、词数变化百分比，以及保持关键词和关键短语（语义相似性）的一致性。单因素t检验和多种线性回归模型表明，在四个指标中，最佳性能的LLM和提示技术存在显著差异。无论是LLM还是提示技术，在试图将内容简化到四年级阅读水平时，都能在年级水平准确性和关键词和关键短语的一致性方面展现出不同的适用性。这些结果表明，在简化文本方面应用LLMs的高效和精准自动化的前景，当前模型和提示方法在满足各种评价标准的综合平衡方面存在的不足，以及评估未来系统的一种通用方法。 

---
# Multilingual LLMs Struggle to Link Orthography and Semantics in Bilingual Word Processing 

**Title (ZH)**: 多语言大型语言模型在双语单词处理中难以链接拼写与语义 

**Authors**: Eshaan Tanwar, Gayatri Oke, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2501.09127)  

**Abstract**: Bilingual lexical processing is shaped by the complex interplay of phonological, orthographic, and semantic features of two languages within an integrated mental lexicon. In humans, this is evident in the ease with which cognate words - words similar in both orthographic form and meaning (e.g., blind, meaning "sightless" in both English and German) - are processed, compared to the challenges posed by interlingual homographs, which share orthographic form but differ in meaning (e.g., gift, meaning "present" in English but "poison" in German). We investigate how multilingual Large Language Models (LLMs) handle such phenomena, focusing on English-Spanish, English-French, and English-German cognates, non-cognate, and interlingual homographs. Specifically, we evaluate their ability to disambiguate meanings and make semantic judgments, both when these word types are presented in isolation or within sentence contexts. Our findings reveal that while certain LLMs demonstrate strong performance in recognizing cognates and non-cognates in isolation, they exhibit significant difficulty in disambiguating interlingual homographs, often performing below random baselines. This suggests LLMs tend to rely heavily on orthographic similarities rather than semantic understanding when interpreting interlingual homographs. Further, we find LLMs exhibit difficulty in retrieving word meanings, with performance in isolative disambiguation tasks having no correlation with semantic understanding. Finally, we study how the LLM processes interlingual homographs in incongruent sentences. We find models to opt for different strategies in understanding English and non-English homographs, highlighting a lack of a unified approach to handling cross-lingual ambiguities. 

**Abstract (ZH)**: 双语词汇处理是由两种语言在整合的心理词库中复杂的音韵、拼写和语义特征之间的相互作用所塑造的。在人类中，这一点体现在歧义词——这两个语言中拼写形式和意义相似的单词（例如，英语和德语中“blind”的意思是“失明”）在处理上相对容易，与那些拼写形式相同而意义不同的跨语言同形词（例如，英语中的“gift”意思是“礼物”，而德语中的“gift”意思是“毒药”）所面临的挑战形成了对比。我们调查了多语言大型语言模型（LLMs）如何处理这些现象，并重点关注英语-西班牙语、英语-法语和英语-德语的同形词、非同形词和跨语言同形词。具体而言，我们评估它们在单词单独呈现或在句子上下文呈现时区分意义和进行语义判断的能力。我们的研究结果表明，尽管某些LLM在单独呈现同形词和非同形词时表现出较强的识别能力，但在区分跨语言同形词时却表现出显著困难，经常表现低于随机基线的水平。这表明在解释跨语言同形词时，LLM们倾向于依赖拼写相似性而非语义理解。此外，我们发现LLM在检索词义方面存在困难，孤立消歧任务的表现与语义理解之间没有关联。最后，我们研究了LLM如何处理不一致句中的跨语言同形词。我们发现模型在理解英语和非英语同形词时采取了不同的策略，这突显了处理跨语言歧义时缺乏统一的方法。 

---
# Augmenting Human-Annotated Training Data with Large Language Model Generation and Distillation in Open-Response Assessment 

**Title (ZH)**: 在开放响应评估中通过大规模语言模型生成和精简增强人类标注训练数据 

**Authors**: Conrad Borchers, Danielle R. Thomas, Jionghao Lin, Ralph Abboud, Kenneth R. Koedinger  

**Link**: [PDF](https://arxiv.org/pdf/2501.09126)  

**Abstract**: Large Language Models (LLMs) like GPT-4o can help automate text classification tasks at low cost and scale. However, there are major concerns about the validity and reliability of LLM outputs. By contrast, human coding is generally more reliable but expensive to procure at scale. In this study, we propose a hybrid solution to leverage the strengths of both. We combine human-coded data and synthetic LLM-produced data to fine-tune a classical machine learning classifier, distilling both into a smaller BERT model. We evaluate our method on a human-coded test set as a validity measure for LLM output quality. In three experiments, we systematically vary LLM-generated samples' size, variety, and consistency, informed by best practices in LLM tuning. Our findings indicate that augmenting datasets with synthetic samples improves classifier performance, with optimal results achieved at an 80% synthetic to 20% human-coded data ratio. Lower temperature settings of 0.3, corresponding to less variability in LLM generations, produced more stable improvements but also limited model learning from augmented samples. In contrast, higher temperature settings (0.7 and above) introduced greater variability in performance estimates and, at times, lower performance. Hence, LLMs may produce more uniform output that classifiers overfit to earlier or produce more diverse output that runs the risk of deteriorating model performance through information irrelevant to the prediction task. Filtering out inconsistent synthetic samples did not enhance performance. We conclude that integrating human and LLM-generated data to improve text classification models in assessment offers a scalable solution that leverages both the accuracy of human coding and the variety of LLM outputs. 

**Abstract (ZH)**: 大型语言模型（LLMs）如GPT-4可以在低成本和大规模的情况下自动化文本分类任务。然而，LLMs输出的有效性和可靠性存在重大关切。相比之下，虽然人工编码通常更可靠，但在大规模应用中却成本高昂。在本研究中，我们提出了一种混合解决方案，结合人工编码数据和合成LLM生成的数据，用于微调经典机器学习分类器，并将其精简为一个更小的BERT模型。我们通过人工编码的数据集进行评估，用作衡量LLM输出质量的标准。在三个实验中，我们系统地调整LLM生成样本的数量、多样性和一致性，以遵循LLM调优的最佳实践。研究结果表明，将合成样本添加到数据集中可以提高分类器性能，最优效果出现在合成样本占80%，人工编码数据占20%的比例下。较低的温度设置（0.3）产生更稳定的改进，但限制了模型从增强样本中学习的能力。相反，较高的温度设置（0.7及以上）增加了性能估计的变异性，并在某些情况下降低了性能。因此，LLMs可能会生成模型过拟合的数据，导致输出更加统一；或者生成更多样化的输出，增加模型性能受损的风险，因为这些输出与预测任务无关的信息过多。排除不一致的合成样本并未提升性能。我们得出结论，结合人工和LLM生成的数据以改进评估中的文本分类模型提供了一种可扩展的解决方案，该解决方案充分利用了人工编码的准确性以及LLM输出的多样性。 

---
# SteLLA: A Structured Grading System Using LLMs with RAG 

**Title (ZH)**: SteLLA：一种基于LLM和RAG的结构化评分系统

注释：在这句话中，“SteLLA”看起来像是一个专有名词或系统名称，因此保持不变。"LLM"是指大型语言模型（Large Language Model），"RAG"是指检索增强生成（Retrieval-Augmented Generation），这些都是当前学术和工业界讨论的热点话题。 

**Authors**: Hefei Qiu, Brian White, Ashley Ding, Reinaldo Costa, Ali Hachem, Wei Ding, Ping Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.09092)  

**Abstract**: Large Language Models (LLMs) have shown strong general capabilities in many applications. However, how to make them reliable tools for some specific tasks such as automated short answer grading (ASAG) remains a challenge. We present SteLLA (Structured Grading System Using LLMs with RAG) in which a) Retrieval Augmented Generation (RAG) approach is used to empower LLMs specifically on the ASAG task by extracting structured information from the highly relevant and reliable external knowledge based on the instructor-provided reference answer and rubric, b) an LLM performs a structured and question-answering-based evaluation of student answers to provide analytical grades and feedback. A real-world dataset that contains students' answers in an exam was collected from a college-level Biology course. Experiments show that our proposed system can achieve substantial agreement with the human grader while providing break-down grades and feedback on all the knowledge points examined in the problem. A qualitative and error analysis of the feedback generated by GPT4 shows that GPT4 is good at capturing facts while may be prone to inferring too much implication from the given text in the grading task which provides insights into the usage of LLMs in the ASAG system. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在许多应用中展示了强大的通用能力。然而，如何使它们成为某些特定任务（如自动化简答评分，ASAG）的可靠工具仍然是一个挑战。我们提出了SteLLA（基于LLM与RAG的结构化评分系统），其中：
a) 使用检索增强生成（RAG）方法，通过从与参考答案和评分标准高度相关且可靠的外部知识中提取结构化信息，具体提升LLM在ASAG任务上的能力；
b) 一个LLM通过结构化和基于问题的答案评估来对学生答案进行评分和反馈，从而提供分析性评分和反馈。
我们从一门大学生物学课程的考试中收集了一个包含学生答案的现实世界数据集。实验结果表明，我们提出系统的评分结果与人类评分者之间具有显著的一致性，同时提供了所有知识点的详细评分和反馈。对GPT4生成的反馈进行定性和错误分析后发现，GPT4在捕捉事实方面表现良好，但在评分任务中可能会从给定文本中过分推断含义，这为LLMs在ASAG系统中的应用提供了见解。 

---
# Decompose-ToM: Enhancing Theory of Mind Reasoning in Large Language Models through Simulation and Task Decomposition 

**Title (ZH)**: Decompose-ToM：通过模拟和任务分解增强大型语言模型的理论心智推理能力 

**Authors**: Sneheel Sarangi, Maha Elgarf, Hanan Salam  

**Link**: [PDF](https://arxiv.org/pdf/2501.09056)  

**Abstract**: Theory of Mind (ToM) is the ability to understand and reflect on the mental states of others. Although this capability is crucial for human interaction, testing on Large Language Models (LLMs) reveals that they possess only a rudimentary understanding of it. Although the most capable closed-source LLMs have come close to human performance on some ToM tasks, they still perform poorly on complex variations of the task that involve more structured reasoning. In this work, we utilize the concept of "pretend-play", or ``Simulation Theory'' from cognitive psychology to propose ``Decompose-ToM'': an LLM-based inference algorithm that improves model performance on complex ToM tasks. We recursively simulate user perspectives and decompose the ToM task into a simpler set of functions: subject identification, question-reframing, world model updation, and knowledge availability. We test the algorithm on higher-order ToM tasks and a task testing for ToM capabilities in a conversational setting, demonstrating that our approach shows significant improvement across models compared to baseline methods while requiring minimal prompt tuning across tasks and no additional model training. 

**Abstract (ZH)**: 理论心理（ToM）是指理解并反思他人心理状态的能力。尽管这一能力对于人类的互动至关重要，但在对大型语言模型（LLMs）的测试中显示，它们仅具备初级的ToM理解能力。尽管最强大的闭源LLMs已经在某些ToM任务上接近了人类的表现，但在涉及更复杂推理的结构化任务中仍然表现不佳。在本研究中，我们利用认知心理学中的“假装游戏”概念，即“模拟理论”，提出了一种基于LLM的推理算法——“分解ToM”：该算法能够提高模型在复杂ToM任务中的性能。我们递归地模拟用户视角，并将ToM任务分解为更简单的一组功能：主体识别、问题重新表述、世界模型更新以及知识可用性。我们测试了该算法在高阶ToM任务和对话情境中测试ToM能力的表现，结果显示，相较于基线方法，我们的方法在各个模型上表现出了显著的改进，同时仅需最少的提示调整，并且无需额外的模型训练。 

---
# Suggesting Code Edits in Interactive Machine Learning Notebooks Using Large Language Models 

**Title (ZH)**: 使用大型语言模型在交互式机器学习笔记本中建议代码编辑 

**Authors**: Bihui Jin, Jiayue Wang, Pengyu Nie  

**Link**: [PDF](https://arxiv.org/pdf/2501.09745)  

**Abstract**: Machine learning developers frequently use interactive computational notebooks, such as Jupyter notebooks, to host code for data processing and model training. Jupyter notebooks provide a convenient tool for writing machine learning pipelines and interactively observing outputs, however, maintaining Jupyter notebooks, e.g., to add new features or fix bugs, can be challenging due to the length and complexity of the notebooks. Moreover, there is no existing benchmark related to developer edits on Jupyter notebooks. To address this, we present the first dataset of 48,398 Jupyter notebook edits derived from 20,095 revisions of 792 machine learning repositories on GitHub, and perform the first study of the using LLMs to predict code edits in Jupyter notebooks. Our dataset captures granular details of cell-level and line-level modifications, offering a foundation for understanding real-world maintenance patterns in machine learning workflows. We observed that the edits on Jupyter notebooks are highly localized, with changes averaging only 166 lines of code in repositories. While larger models outperform smaller counterparts in code editing, all models have low accuracy on our dataset even after finetuning, demonstrating the complexity of real-world machine learning maintenance tasks. Our findings emphasize the critical role of contextual information in improving model performance and point toward promising avenues for advancing large language models' capabilities in engineering machine learning code. 

**Abstract (ZH)**: 机器学习开发人员经常使用交互式计算笔记本（如Jupyter笔记本）来托管数据处理和模型训练的代码。Jupyter笔记本为编写机器学习管道和实时观察输出结果提供了一种方便的工具，但是维护Jupyter笔记本，例如添加新功能或修复错误，可能会因笔记本的长度和复杂性而变得颇具挑战性。此外，目前尚无关于开发人员对Jupyter笔记本所做的修改的基准测试。为了解决这一问题，我们首次创建了一个包含48,398个Jupyter笔记本编辑的数据集，这些编辑是从GitHub上792个机器学习仓库的20,095个修订版本中提取出来的，并首次使用大语言模型（LLM）来预测Jupyter笔记本中的代码编辑。我们的数据集涵盖了单元级别和行级别的详细修改信息，为了解机器学习工作流中的实际维护模式提供了基础。我们观察到，Jupyter笔记本中的修改具有高度局部化的特点，平均每个仓库修改仅涉及166行代码。虽然更大的模型在代码编辑方面优于较小的模型，但在我们数据集上的所有模型即使经过微调后仍具有较低的准确率，这表明现实世界的机器学习维护任务具有较高的复杂性。我们的研究结果强调了改进模型性能中上下文信息的关键作用，并指出了提升大语言模型在工程机器学习代码方面能力的前景。 

---
# Towards Large Reasoning Models: A Survey of Reinforced Reasoning with Large Language Models 

**Title (ZH)**: 面向大规模推理模型：大规模语言模型强化推理综述 

**Authors**: Fengli Xu, Qianyue Hao, Zefang Zong, Jingwei Wang, Yunke Zhang, Jingyi Wang, Xiaochong Lan, Jiahui Gong, Tianjian Ouyang, Fanjin Meng, Chenyang Shao, Yuwei Yan, Qinglong Yang, Yiwen Song, Sijian Ren, Xinyuan Hu, Yu Li, Jie Feng, Chen Gao, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.09686)  

**Abstract**: Language has long been conceived as an essential tool for human reasoning. The breakthrough of Large Language Models (LLMs) has sparked significant research interest in leveraging these models to tackle complex reasoning tasks. Researchers have moved beyond simple autoregressive token generation by introducing the concept of "thought" -- a sequence of tokens representing intermediate steps in the reasoning process. This innovative paradigm enables LLMs' to mimic complex human reasoning processes, such as tree search and reflective thinking. Recently, an emerging trend of learning to reason has applied reinforcement learning (RL) to train LLMs to master reasoning processes. This approach enables the automatic generation of high-quality reasoning trajectories through trial-and-error search algorithms, significantly expanding LLMs' reasoning capacity by providing substantially more training data. Furthermore, recent studies demonstrate that encouraging LLMs to "think" with more tokens during test-time inference can further significantly boost reasoning accuracy. Therefore, the train-time and test-time scaling combined to show a new research frontier -- a path toward Large Reasoning Model. The introduction of OpenAI's o1 series marks a significant milestone in this research direction. In this survey, we present a comprehensive review of recent progress in LLM reasoning. We begin by introducing the foundational background of LLMs and then explore the key technical components driving the development of large reasoning models, with a focus on automated data construction, learning-to-reason techniques, and test-time scaling. We also analyze popular open-source projects at building large reasoning models, and conclude with open challenges and future research directions. 

**Abstract (ZH)**: 语言长期以来一直被认为是人类推理的关键工具。大型语言模型（LLMs）的突破引发了利用这些模型解决复杂推理任务的研究兴趣。研究人员已经超越了简单的自回归标记生成，引入了“思考”的概念——一系列表示推理过程中间步骤的标记序列。这一创新范式使LLMs能够模仿复杂的类人推理过程，例如树搜索和反思性思考。最近，一种新兴的学习推理趋势应用强化学习（RL）来训练LLMs掌握推理过程。这种方法通过试错搜索算法自动生成高质量的推理轨迹，大大扩展了LLMs的推理能力，提供更多的训练数据。此外，最近的研究表明，在测试时促使LLMs生成更多的标记进行推理可以进一步显著提高推理准确性。因此，训练时间和测试时间的扩展共同展示了一条新的研究前沿——通往大型推理模型的道路。OpenAI的o1系列的推出标志着这一研究方向的重要里程碑。在本文综述中，我们将全面回顾近年来LLMs推理的发展。首先，我们介绍LLMs的基础背景，然后探讨推动大型推理模型发展的关键技术组件，重点在于自动化数据构建、学习推理技术以及测试时的扩展。我们还将分析用于构建大型推理模型的流行开源项目，并总结存在的挑战和未来的研究方向。 

---
# Augmenting a Large Language Model with a Combination of Text and Visual Data for Conversational Visualization of Global Geospatial Data 

**Title (ZH)**: 将一大型语言模型与文本和视觉数据相结合，以实现全球地理空间数据的对话可视化 

**Authors**: Omar Mena, Alexandre Kouyoumdjian, Lonni Besançon, Michael Gleicher, Ivan Viola, Anders Ynnerman  

**Link**: [PDF](https://arxiv.org/pdf/2501.09521)  

**Abstract**: We present a method for augmenting a Large Language Model (LLM) with a combination of text and visual data to enable accurate question answering in visualization of scientific data, making conversational visualization possible. LLMs struggle with tasks like visual data interaction, as they lack contextual visual information. We address this problem by merging a text description of a visualization and dataset with snapshots of the visualization. We extract their essential features into a structured text file, highly compact, yet descriptive enough to appropriately augment the LLM with contextual information, without any fine-tuning. This approach can be applied to any visualization that is already finally rendered, as long as it is associated with some textual description. 

**Abstract (ZH)**: 我们提出了一种方法，将大型语言模型（LLM）与文本和视觉数据结合起来，以实现科学数据可视化中的准确问题回答，从而使对话式可视化成为可能。LLM 在处理如视觉数据交互等任务时存在困难，因为它们缺乏上下文视觉信息。我们通过将可视化和数据集的文本描述与可视化快照相结合来解决这一问题。我们将这些信息的关键特征提取到一个结构化的文本文件中，该文件虽高度紧凑，但描述足够详细，能够为LLM提供必要的上下文信息，而无需进行任何微调。这种方法可以应用于任何已最终渲染的可视化，只要它与某些文本描述相关联即可。 

---
# Generative Visual Commonsense Answering and Explaining with Generative Scene Graph Constructing 

**Title (ZH)**: 生成式视觉常识回答与生成场景图构建解释 

**Authors**: Fan Yuan, Xiaoyuan Fang, Rong Quan, Jing Li, Wei Bi, Xiaogang Xu, Piji Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.09041)  

**Abstract**: Visual Commonsense Reasoning, which is regarded as one challenging task to pursue advanced visual scene comprehension, has been used to diagnose the reasoning ability of AI systems. However, reliable reasoning requires a good grasp of the scene's details. Existing work fails to effectively exploit the real-world object relationship information present within the scene, and instead overly relies on knowledge from training memory. Based on these observations, we propose a novel scene-graph-enhanced visual commonsense reasoning generation method named \textit{\textbf{G2}}, which first utilizes the image patches and LLMs to construct a location-free scene graph, and then answer and explain based on the scene graph's information. We also propose automatic scene graph filtering and selection strategies to absorb valuable scene graph information during training. Extensive experiments are conducted on the tasks and datasets of scene graph constructing and visual commonsense answering and explaining, respectively. Experimental results and ablation analysis demonstrate the effectiveness of our proposed framework. 

**Abstract (ZH)**: 视觉常识推理（Visual Commonsense Reasoning）被视为追求高级视觉场景理解的一项具有挑战性的任务，已被用于评估AI系统的推理能力。然而，可靠推理需要对场景细节有较好的掌握。现有工作未能有效地利用场景内真实世界对象之间的关系信息，而是过度依赖训练记忆中的知识。基于这些观察，我们提出了一种新颖的场景图增强视觉常识推理生成方法，命名为**G2**。该方法首先利用图像片段和LLMs构建一个位置无关的场景图，然后基于场景图的信息进行回答和解释。我们还提出了自动场景图筛选和选择策略，以在训练过程中吸收有价值的信息。我们在场景图构建和视觉常识回答与解释的任务和数据集上进行了广泛的实验。实验结果和消融分析证明了我们提出的框架的有效性。 

---
# Aligning Instruction Tuning with Pre-training 

**Title (ZH)**: 将指令调优与预训练对齐 

**Authors**: Yiming Liang, Tianyu Zheng, Xinrun Du, Ge Zhang, Xingwei Qu, Xiang Yue, Chujie Zheng, Jiaheng Liu, Lei Ma, Wenhu Chen, Guoyin Wang, Zhaoxiang Zhang, Wenhao Huang, Jiajun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.09368)  

**Abstract**: Instruction tuning enhances large language models (LLMs) to follow human instructions across diverse tasks, relying on high-quality datasets to guide behavior. However, these datasets, whether manually curated or synthetically generated, are often narrowly focused and misaligned with the broad distributions captured during pre-training, limiting LLM generalization and effective use of pre-trained knowledge. We propose *Aligning Instruction Tuning with Pre-training* (AITP), a method that bridges this gap by identifying coverage shortfalls in instruction-tuning datasets and rewriting underrepresented pre-training data into high-quality instruction-response pairs. This approach enriches dataset diversity while preserving task-specific objectives. Evaluations on three fully open LLMs across eight benchmarks demonstrate consistent performance improvements with AITP. Ablations highlight the benefits of adaptive data selection, controlled rewriting, and balanced integration, emphasizing the importance of aligning instruction tuning with pre-training distributions to unlock the full potential of LLMs. 

**Abstract (ZH)**: 指令微调增强了大规模语言模型（LLMs）在各种任务中遵循人类指令的能力，依赖于高质量数据集的指导。然而，这些数据集，无论是手工策划的还是合成生成的，往往聚焦狭窄且与预训练期间捕捉到的广泛分布不一致，限制了LLM的泛化能力和预训练知识的有效利用。我们提出了一种名为*预训练与指令微调对齐*（AITP，Alignment of Instruction Tuning with Pre-training）的方法，该方法通过识别指令微调数据集中的覆盖率不足，并将未充分代表的预训练数据改写为高质量的指令-响应对，从而弥合了这一差距。这种方法丰富了数据集的多样性，同时保留了特定任务的目标。在三个完全开放的LLM上，针对八个基准进行的评估表明，AITP能够一致地提高性能。消融实验突显了自适应数据选择、可控改写和平衡整合的好处，强调了将指令微调与预训练分布对齐的重要性，以充分利用LLM的潜力。 

---
# CyberMentor: AI Powered Learning Tool Platform to Address Diverse Student Needs in Cybersecurity Education 

**Title (ZH)**: CyberMentor：一种基于人工智能的学习工具平台，旨在满足网络安全教育中多样化学生需求 

**Authors**: Tianyu Wang, Nianjun Zhou, Zhixiong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.09709)  

**Abstract**: Many non-traditional students in cybersecurity programs often lack access to advice from peers, family members and professors, which can hinder their educational experiences. Additionally, these students may not fully benefit from various LLM-powered AI assistants due to issues like content relevance, locality of advice, minimum expertise, and timing. This paper addresses these challenges by introducing an application designed to provide comprehensive support by answering questions related to knowledge, skills, and career preparation advice tailored to the needs of these students. We developed a learning tool platform, CyberMentor, to address the diverse needs and pain points of students majoring in cybersecurity. Powered by agentic workflow and Generative Large Language Models (LLMs), the platform leverages Retrieval-Augmented Generation (RAG) for accurate and contextually relevant information retrieval to achieve accessibility and personalization. We demonstrated its value in addressing knowledge requirements for cybersecurity education and for career marketability, in tackling skill requirements for analytical and programming assignments, and in delivering real time on demand learning support. Using three use scenarios, we showcased CyberMentor in facilitating knowledge acquisition and career preparation and providing seamless skill-based guidance and support. We also employed the LangChain prompt-based evaluation methodology to evaluate the platform's impact, confirming its strong performance in helpfulness, correctness, and completeness. These results underscore the system's ability to support students in developing practical cybersecurity skills while improving equity and sustainability within higher education. Furthermore, CyberMentor's open-source design allows for adaptation across other disciplines, fostering educational innovation and broadening its potential impact. 

**Abstract (ZH)**: 在网络安全项目中，许多非传统学生往往缺乏来自同学、家庭成员和导师的建议，这可能会阻碍他们的学习体验。此外，这些学生可能无法充分利用各种基于LLM的AI助手，原因包括内容相关性不足、建议的地域性限制、最低专业要求和时间性因素。本文通过引入一个旨在为这些学生提供全面支持的应用程序来解决这些问题。该应用能够回答与知识、技能和职业准备建议相关的问题，提供符合这些学生需求的定制化建议。我们开发了一个名为CyberMentor的学习工具平台，以满足网络安全专业学生多样化的需求和痛点。该平台借助代理型工作流程和生成型大规模语言模型（LLMs），利用检索增强生成（RAG）实现准确且上下文相关的信息检索，从而实现可访问性和个性化。我们通过网络安全教育的知识需求、职业市场可塑性、技能需求解决分析和编程作业、以及实时按需学习支持等方面的价值展示，证明了CyberMentor的有效性。我们通过三个使用场景展示了CyberMentor在促进知识获取、职业准备和提供无缝技能导向支持方面的应用。同时，我们采用LangChain基于提示的评估方法来评估平台的影响，确认了其在有用性、准确性、完整性的强大表现。这些结果强调了该系统的支持能力，能够在提高教育公平性和可持续性方面帮助学生发展实用的网络安全技能。此外，CyberMentor的开源设计使其能够在其他学科中进行调整，从而促进教育创新并扩大其潜在影响。 

---
# Beyond Reward Hacking: Causal Rewards for Large Language Model Alignment 

**Title (ZH)**: 超越奖励作弊：大型语言模型对齐的因果奖励 

**Authors**: Chaoqi Wang, Zhuokai Zhao, Yibo Jiang, Zhaorun Chen, Chen Zhu, Yuxin Chen, Jiayi Liu, Lizhu Zhang, Xiangjun Fan, Hao Ma, Sinong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.09620)  

**Abstract**: Recent advances in large language models (LLMs) have demonstrated significant progress in performing complex tasks. While Reinforcement Learning from Human Feedback (RLHF) has been effective in aligning LLMs with human preferences, it is susceptible to spurious correlations in reward modeling. Consequently, it often introduces biases-such as length bias, sycophancy, conceptual bias, and discrimination that hinder the model's ability to capture true causal relationships. To address this, we propose a novel causal reward modeling approach that integrates causal inference to mitigate these spurious correlations. Our method enforces counterfactual invariance, ensuring reward predictions remain consistent when irrelevant variables are altered. Through experiments on both synthetic and real-world datasets, we show that our approach mitigates various types of spurious correlations effectively, resulting in more reliable and fair alignment of LLMs with human preferences. As a drop-in enhancement to the existing RLHF workflow, our causal reward modeling provides a practical way to improve the trustworthiness and fairness of LLM finetuning. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）在执行复杂任务方面取得了显著进展。尽管基于人类反馈的强化学习（RLHF）在使LLMs与人类偏好对齐方面效果显著，但它在奖励建模中容易受到虚假相关性的影响。因此，它常常引入一些偏差，如长度偏差、奉承、概念偏差和歧视，这些偏差妨碍了模型捕捉真实因果关系的能力。为解决这一问题，我们提出了一种新的因果奖励建模方法，该方法结合因果推断以减轻这些虚假相关性的影响。我们的方法确保了反事实不变性，即在改变无关变量时，奖励预测保持一致。通过在合成数据集和真实世界数据集上的实验，我们展示了我们的方法有效地减轻了各种类型的虚假相关性，从而更可靠和公平地使LLMs与人类偏好对齐。作为一种针对现有RLHF工作流的即插即用增强方法，我们的因果奖励建模提供了提高LLM微调可信度和公平性的实用途径。 

---
# MoE$^2$: Optimizing Collaborative Inference for Edge Large Language Models 

**Title (ZH)**: MoE\(^2\): 优化边缘大型语言模型协作推理 

**Authors**: Lyudong Jin, Yanning Zhang, Yanhan Li, Shurong Wang, Howard H. Yang, Jian Wu, Meng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.09410)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across a wide range of natural language processing tasks. Exploiting the heterogeneous capabilities of edge LLMs is crucial for diverse emerging applications, as it enables greater cost-effectiveness and reduced latency. In this work, we introduce \textit{Mixture-of-Edge-Experts (MoE$^2$)}, a novel collaborative inference framework for edge LLMs. We formulate the joint gating and expert selection problem to optimize inference performance under energy and latency constraints. Unlike conventional MoE problems, LLM expert selection is significantly more challenging due to the combinatorial nature and the heterogeneity of edge LLMs across various attributes. To this end, we propose a two-level expert selection mechanism through which we uncover an optimality-preserving property of gating parameters across expert selections. This property enables the decomposition of the training and selection processes, significantly reducing complexity. Furthermore, we leverage the objective's monotonicity and design a discrete monotonic optimization algorithm for optimal expert selection. We implement edge servers with NVIDIA Jetson AGX Orins and NVIDIA RTX 4090 GPUs, and perform extensive experiments. Our results validate that performance improvements of various LLM models and show that our MoE$^2$ method can achieve optimal trade-offs among different delay and energy budgets, and outperforms baselines under various system resource constraints. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在广泛自然语言处理任务中展现了显著的能力。充分利用边缘端LLMs的异质能力对于多种新兴应用至关重要，因为它能提高成本效益并减少延迟。在这项工作中，我们介绍了**边缘端专家混合（MoE²）**，这是一种新颖的边缘端LLMs协作推理框架。我们提出了联合门控和专家选择问题的建模方法，以在能量和延迟约束下优化推理性能。与传统的MoE问题不同，由于边缘端LLMs在各个属性上的组合特性和异质性，LLM专家选择具有显著的挑战性。为了解决这一问题，我们提出了一种两层专家选择机制，通过该机制揭示了门控参数在不同专家选择下的保持最优性属性。这一属性使得训练和选择过程得以分解，显著降低了复杂度。此外，我们利用目标函数的单调性并设计了离散单调优化算法来进行最优专家选择。我们使用NVIDIA Jetson AGX Orin和NVIDIA RTX 4090 GPU构建边缘服务器，并进行了广泛实验。实验结果证明了各种LLM模型性能的提升，并表明我们的MoE²方法能够在不同的延迟和能量预算下实现最优权衡，并在各种系统资源约束下优于基准方法。 

---
# Rational Tuning of LLM Cascades via Probabilistic Modeling 

**Title (ZH)**: 通过概率建模对大规模语言模型级联进行合理的调优 

**Authors**: Michael J. Zellinger, Matt Thomson  

**Link**: [PDF](https://arxiv.org/pdf/2501.09345)  

**Abstract**: Understanding the reliability of large language models (LLMs) has recently garnered significant attention. Given LLMs' propensity to hallucinate, as well as their high sensitivity to prompt design, it is already challenging to predict the performance of an individual LLM. However, the problem becomes more complex for compound LLM systems such as cascades, where in addition to each model's standalone performance, we must understand how the error rates of different models interact. In this paper, we present a probabilistic model for the joint performance distribution of a sequence of LLMs, which enables a framework for rationally tuning the confidence thresholds of a LLM cascade using continuous optimization. Compared to selecting confidence thresholds using grid search, our parametric Markov-copula model significantly improves runtime scaling with respect to the length of the cascade and the desired resolution of the cost-error curve, turning them from intractable into low-order polynomial. In addition, the optimal thresholds computed using our continuous optimization-based algorithm increasingly outperform those found via grid search as cascade length grows, improving the area under the cost-error curve by 1.9% on average for cascades consisting of at least three models. Overall, our Markov-copula model provides a rational basis for tuning LLM cascade performance and points to the potential of probabilistic methods in analyzing LLM systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）的可靠性近年来得到了广泛关注。鉴于LLMs具有生成幻觉的倾向，以及其对提示设计的高度敏感性，预测单个LLM的性能已经颇具挑战性。而在复合LLM系统，如级联系统中，问题变得更加复杂，我们需要不仅考虑每个模型的独立性能，还要理解不同模型错误率之间的相互作用。在这篇论文中，我们提出了一种概率模型来描述LLM序列的联合性能分布，该模型可以作为一种框架来有理由地使用连续优化调整LLM级联的置信阈值。与使用网格搜索选择置信阈值相比，我们的参数马尔可夫-柯皮亚模型在级联长度和所需的成本-错误曲线分辨率方面显著改善了运行时扩展性，使它们从不可解变成低次多项式。另外，基于连续优化算法计算的最佳阈值在级联长度增长时逐渐优于网格搜索找到的阈值，在长度至少为三个模型的级联中，平均提高了成本-错误曲线下的面积1.9%。总体而言，我们的马尔可夫-柯皮亚模型为调整LLM级联性能提供了一个理性的基础，并指出概率方法在分析LLM系统方面具有潜在优势。 

---
# LAVCap: LLM-based Audio-Visual Captioning using Optimal Transport 

**Title (ZH)**: LAVCap：基于最优 transport 的大型语言模型驱动的音视频描述生成 

**Authors**: Kyeongha Rho, Hyeongkeun Lee, Valentio Iverson, Joon Son Chung  

**Link**: [PDF](https://arxiv.org/pdf/2501.09291)  

**Abstract**: Automated audio captioning is a task that generates textual descriptions for audio content, and recent studies have explored using visual information to enhance captioning quality. However, current methods often fail to effectively fuse audio and visual data, missing important semantic cues from each modality. To address this, we introduce LAVCap, a large language model (LLM)-based audio-visual captioning framework that effectively integrates visual information with audio to improve audio captioning performance. LAVCap employs an optimal transport-based alignment loss to bridge the modality gap between audio and visual features, enabling more effective semantic extraction. Additionally, we propose an optimal transport attention module that enhances audio-visual fusion using an optimal transport assignment map. Combined with the optimal training strategy, experimental results demonstrate that each component of our framework is effective. LAVCap outperforms existing state-of-the-art methods on the AudioCaps dataset, without relying on large datasets or post-processing. Code is available at this https URL. 

**Abstract (ZH)**: 自动音频字幕生成是一项生成音频内容文本描述的任务，近年来的研究探讨了利用视觉信息来提高字幕质量的方法。然而，当前的方法往往无法有效地融合音频和视觉数据，从而错过了每种模态中的重要语义线索。为解决这一问题，我们引入了LAVCap，这是一种基于大规模语言模型（LLM）的音频-视觉字幕生成框架，能够有效地将视觉信息与音频融合，以提高音频字幕生成性能。LAVCap 采用基于运筹学的对齐损失来弥合音频和视觉特征之间的模态差异，从而更好地提取语义信息。此外，我们提出了一个基于运筹学注意力模块，该模块使用运筹学分配图增强了音频-视觉融合能力。结合最佳训练策略，实验结果表明，框架中的每个组件都是有效的。LAVCap 在 AudioCaps 数据集上的性能超过了现有最先进的方法，且无需依赖大规模数据集或后续处理。相关代码可从此链接访问：[点击访问代码] 

---
# Large Language Model is Secretly a Protein Sequence Optimizer 

**Title (ZH)**: 大型语言模型实际上是蛋白质序列优化器 

**Authors**: Yinkai Wang, Jiaxing He, Yuanqi Du, Xiaohui Chen, Jianan Canal Li, Li-Ping Liu, Xiaolin Xu, Soha Hassoun  

**Link**: [PDF](https://arxiv.org/pdf/2501.09274)  

**Abstract**: We consider the protein sequence engineering problem, which aims to find protein sequences with high fitness levels, starting from a given wild-type sequence. Directed evolution has been a dominating paradigm in this field which has an iterative process to generate variants and select via experimental feedback. We demonstrate large language models (LLMs), despite being trained on massive texts, are secretly protein sequence optimizers. With a directed evolutionary method, LLM can perform protein engineering through Pareto and experiment-budget constrained optimization, demonstrating success on both synthetic and experimental fitness landscapes. 

**Abstract (ZH)**: 我们将研究蛋白质序列工程问题，其目标是从给定的野生型序列出发，找到具有高适应度水平的蛋白质序列。定向进化一直是该领域的主导范式，它通过迭代过程生成变异体，并通过实验反馈进行选择。我们证明，尽管大型语言模型（LLMs）是通过大量文本训练的，但它们实际上是蛋白质序列优化器。通过使用定向进化方法，LLMs可以通过帕累托优化和实验预算约束进行蛋白质工程，展示出在合成性和实验性适应度景观中均取得成功的能力。 

---
# Towards Multilingual LLM Evaluation for Baltic and Nordic languages: A study on Lithuanian History 

**Title (ZH)**: 面向波罗的海和北欧语言的多语言LLM评估：立陶宛历史研究 

**Authors**: Yevhen Kostiuk, Oxana Vitman, Łukasz Gagała, Artur Kiulian  

**Link**: [PDF](https://arxiv.org/pdf/2501.09154)  

**Abstract**: In this work, we evaluated Lithuanian and general history knowledge of multilingual Large Language Models (LLMs) on a multiple-choice question-answering task. The models were tested on a dataset of Lithuanian national and general history questions translated into Baltic, Nordic, and other languages (English, Ukrainian, Arabic) to assess the knowledge sharing from culturally and historically connected groups. We evaluated GPT-4o, LLaMa3.1 8b and 70b, QWEN2.5 7b and 72b, Mistral Nemo 12b, LLaMa3 8b, Mistral 7b, LLaMa3.2 3b, and Nordic fine-tuned models (GPT-SW3 and LLaMa3 8b).
Our results show that GPT-4o consistently outperformed all other models across language groups, with slightly better results for Baltic and Nordic languages. Larger open-source models like QWEN2.5 72b and LLaMa3.1 70b performed well but showed weaker alignment with Baltic languages. Smaller models (Mistral Nemo 12b, LLaMa3.2 3b, QWEN 7B, LLaMa3.1 8B, and LLaMa3 8b) demonstrated gaps with LT-related alignment with Baltic languages while performing better on Nordic and other languages. The Nordic fine-tuned models did not surpass multilingual models, indicating that shared cultural or historical context alone does not guarantee better performance. 

**Abstract (ZH)**: 在本研究中，我们通过多项选择题作答任务评估了多语言大型语言模型（LLMs）对立陶宛历史和一般历史知识的理解。模型被测试在一个包含立陶宛国家历史和一般历史问题的数据集中，这些问题已经被翻译成波罗的海语系、北欧语系以及其他语言（英语、乌克兰语、阿拉伯语），以此来评估文化或历史联系紧密的群体之间的知识共享情况。我们评估了GPT-4o、LLaMa3.1 8b和70b、QWEN2.5 7b和72b、Mistral Nemo 12b、LLaMa3 8b、Mistral 7b、LLaMa3.2 3b以及北欧语系微调的模型（GPT-SW3和LLaMa3 8b）。

研究结果显示，GPT-4o在各个语言组中始终保持领先地位，特别是在波罗的海语系和北欧语系的语言中表现更佳。大型开源模型如QWEN2.5 72b和LLaMa3.1 70b表现良好，但与波罗的海语系语言的匹配度较弱。较小的模型（Mistral Nemo 12b、LLaMa3.2 3b、QWEN 7B、LLaMa3.1 8B和LLaMa3 8b）在立陶宛语相关性匹配度上存在差距，但在这北欧语系和其他语言中的表现较好。北欧语系微调的模型未能超越多语言模型，这表明共享的文化或历史背景并不足以确保更好的模型性能。 

---
