# MedRAG: Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilot 

**Title (ZH)**: 医卫伴侣增强版：基于知识图谱启发推理的检索增强生成技术（MedRAG） 

**Authors**: Xuejiao Zhao, Siyan Liu, Su-Yin Yang, Chunyan Miao  

**Link**: [PDF](https://arxiv.org/pdf/2502.04413)  

**Abstract**: Retrieval-augmented generation (RAG) is a well-suited technique for retrieving privacy-sensitive Electronic Health Records (EHR). It can serve as a key module of the healthcare copilot, helping reduce misdiagnosis for healthcare practitioners and patients. However, the diagnostic accuracy and specificity of existing heuristic-based RAG models used in the medical domain are inadequate, particularly for diseases with similar manifestations. This paper proposes MedRAG, a RAG model enhanced by knowledge graph (KG)-elicited reasoning for the medical domain that retrieves diagnosis and treatment recommendations based on manifestations. MedRAG systematically constructs a comprehensive four-tier hierarchical diagnostic KG encompassing critical diagnostic differences of various diseases. These differences are dynamically integrated with similar EHRs retrieved from an EHR database, and reasoned within a large language model. This process enables more accurate and specific decision support, while also proactively providing follow-up questions to enhance personalized medical decision-making. MedRAG is evaluated on both a public dataset DDXPlus and a private chronic pain diagnostic dataset (CPDD) collected from Tan Tock Seng Hospital, and its performance is compared against various existing RAG methods. Experimental results show that, leveraging the information integration and relational abilities of the KG, our MedRAG provides more specific diagnostic insights and outperforms state-of-the-art models in reducing misdiagnosis rates. Our code will be available at this https URL 

**Abstract (ZH)**: 检索增强生成（RAG）是一种适用于检索敏感电子健康记录（EHR）的技术，非常适合医疗领域的应用。它可以作为医疗copilot的关键模块，帮助减少医疗工作者和患者的误诊。然而，现有基于启发式的医疗领域的RAG模型在疾病表现相似的情况下诊断准确性和特异性不足。本文提出了一种名为MedRAG的模型，该模型通过知识图谱（KG）驱动的推理增强了RAG技术，用于基于症状检索诊断和治疗建议。MedRAG系统性地构建了一个包含各种疾病关键诊断差异的多层次诊断KG。这些差异与从EHR数据库中检索到的相似EHR动态集成，并在大型语言模型中进行推理。这一过程不仅使诊断决策支持更为准确和具体，还能主动提供后续问题以增强个性化的医疗决策。MedRAG在公共数据集DDXPlus和Tan Tock Seng医院收集的慢性疼痛诊断私有数据集（CPDD）上进行了评估，并将其性能与多种现有RAG方法进行了比较。实验结果表明，通过利用知识图谱的信息整合和关系处理能力，MedRAG提供了更具体的诊断见解，并在降低误诊率方面优于现有最先进的模型。我们的代码将在以下网址提供：[提供链接的网址] 

---
# MARAGE: Transferable Multi-Model Adversarial Attack for Retrieval-Augmented Generation Data Extraction 

**Title (ZH)**: MARAGE：适用于检索增强生成数据提取的可迁移多模型对抗攻击方法 

**Authors**: Xiao Hu, Eric Liu, Weizhou Wang, Xiangyu Guo, David Lie  

**Link**: [PDF](https://arxiv.org/pdf/2502.04360)  

**Abstract**: Retrieval-Augmented Generation (RAG) offers a solution to mitigate hallucinations in Large Language Models (LLMs) by grounding their outputs to knowledge retrieved from external sources. The use of private resources and data in constructing these external data stores can expose them to risks of extraction attacks, in which attackers attempt to steal data from these private databases. Existing RAG extraction attacks often rely on manually crafted prompts, which limit their effectiveness. In this paper, we introduce a framework called MARAGE for optimizing an adversarial string that, when appended to user queries submitted to a target RAG system, causes outputs containing the retrieved RAG data verbatim. MARAGE leverages a continuous optimization scheme that integrates gradients from multiple models with different architectures simultaneously to enhance the transferability of the optimized string to unseen models. Additionally, we propose a strategy that emphasizes the initial tokens in the target RAG data, further improving the attack's generalizability. Evaluations show that MARAGE consistently outperforms both manual and optimization-based baselines across multiple LLMs and RAG datasets, while maintaining robust transferability to previously unseen models. Moreover, we conduct probing tasks to shed light on the reasons why MARAGE is more effective compared to the baselines and to analyze the impact of our approach on the model's internal state. 

**Abstract (ZH)**: 检索增强生成（RAG）通过将大型语言模型（LLMs）的输出与外部来源检索的知识进行结合，提供了一种减轻幻觉的方法。在构建这些外部数据存储时使用私有资源和数据可能会暴露它们受到提取攻击的风险，在这类攻击中，攻击者试图从这些私有数据库中窃取数据。现有的RAG提取攻击通常依赖于手动构建的提示，限制了它们的效果。本文提出了一种名为MARAGE的框架，用于优化一个敌对字符串，当该字符串附加到提交给目标RAG系统的用户查询时，会导致包含摘要检索RAG数据的输出。MARAGE利用了一种连续的优化方案，该方案同时整合了具有不同架构的多个模型的梯度，以增强优化字符串在未见过的模型中的转移性。此外，我们提出了一种策略，强调目标RAG数据中的初始令牌，进一步提高攻击的通用性。评估结果显示，MARAGE在多个LLM和RAG数据集中的一致地优于手动构建的和基于优化的基准，并且在之前未见过的模型上仍然具有稳健的转移性。此外，我们还进行了探究任务，以揭示MARAGE相较于基准模型更有效的原因，并分析了我们方法对模型内部状态的影响。 

---
# Open Foundation Models in Healthcare: Challenges, Paradoxes, and Opportunities with GenAI Driven Personalized Prescription 

**Title (ZH)**: 面向医疗健康的开放基础模型：基于GenAI的个性化处方所面临的挑战、悖论与机遇 

**Authors**: Mahdi Alkaeed, Sofiat Abioye, Adnan Qayyum, Yosra Magdi Mekki, Ilhem Berrou, Mohamad Abdallah, Ala Al-Fuqaha, Muhammad Bilal, Junaid Qadir  

**Link**: [PDF](https://arxiv.org/pdf/2502.04356)  

**Abstract**: In response to the success of proprietary Large Language Models (LLMs) such as OpenAI's GPT-4, there is a growing interest in developing open, non-proprietary LLMs and AI foundation models (AIFMs) for transparent use in academic, scientific, and non-commercial applications. Despite their inability to match the refined functionalities of their proprietary counterparts, open models hold immense potential to revolutionize healthcare applications. In this paper, we examine the prospects of open-source LLMs and AIFMs for developing healthcare applications and make two key contributions. Firstly, we present a comprehensive survey of the current state-of-the-art open-source healthcare LLMs and AIFMs and introduce a taxonomy of these open AIFMs, categorizing their utility across various healthcare tasks. Secondly, to evaluate the general-purpose applications of open LLMs in healthcare, we present a case study on personalized prescriptions. This task is particularly significant due to its critical role in delivering tailored, patient-specific medications that can greatly improve treatment outcomes. In addition, we compare the performance of open-source models with proprietary models in settings with and without Retrieval-Augmented Generation (RAG). Our findings suggest that, although less refined, open LLMs can achieve performance comparable to proprietary models when paired with grounding techniques such as RAG. Furthermore, to highlight the clinical significance of LLMs-empowered personalized prescriptions, we perform subjective assessment through an expert clinician. We also elaborate on ethical considerations and potential risks associated with the misuse of powerful LLMs and AIFMs, highlighting the need for a cautious and responsible implementation in healthcare. 

**Abstract (ZH)**: 针对OpenAI的GPT-4等拥有者专有的大型语言模型（LLMs）的成功，人们日益关注开发开放的、非专有LLMs和人工智能基础模型（AIFMs），以透明地应用于学术、科学和非商业应用中。尽管它们的功能不如专有模型精细，但开放模型在医疗健康应用方面具有巨大的潜力，有可能彻底改变这些领域。本文探讨了开放源代码LLMs和AIFMs在开发医疗健康应用方面的前景，并提出了两个关键贡献。首先，我们对当前最先进的开放源代码医疗LLMs和AIFMs进行了全面的调研，并介绍了这些开放AIFMs的分类法，将它们的应用分类到各种医疗健康任务中。其次，为了评估开放源代码LLMs在医疗健康中的通用应用，我们进行了一个个性化处方的案例研究。这一任务尤为重要，因为个性化处方在提供针对患者的具体建议并显著改善治疗效果方面起着关键作用。此外，我们在有和没有检索增强生成（RAG）的情况下，比较了开放源代码模型与专有模型的性能。研究结果表明，虽然开放模型不如专有模型精细，但在配合使用诸如RAG等对接技术时，它们可以达到与专有模型相当的性能。此外，为了突出LLMs赋能个性化处方的临床价值，我们通过专家临床评估进行了主观测评。同时，我们还探讨了使用LLMs和AIFMs可能带来的伦理考量和潜在风险，强调在医疗健康领域谨慎和负责任地应用这些强大技术的重要性。 

---
# Enhancing Health Information Retrieval with RAG by Prioritizing Topical Relevance and Factual Accuracy 

**Title (ZH)**: 通过优先考虑主题相关性和事实准确性以增强健康信息检索的RAG方法 

**Authors**: Rishabh Uapadhyay, Marco Viviani  

**Link**: [PDF](https://arxiv.org/pdf/2502.04666)  

**Abstract**: The exponential surge in online health information, coupled with its increasing use by non-experts, highlights the pressing need for advanced Health Information Retrieval models that consider not only topical relevance but also the factual accuracy of the retrieved information, given the potential risks associated with health misinformation. To this aim, this paper introduces a solution driven by Retrieval-Augmented Generation (RAG), which leverages the capabilities of generative Large Language Models (LLMs) to enhance the retrieval of health-related documents grounded in scientific evidence. In particular, we propose a three-stage model: in the first stage, the user's query is employed to retrieve topically relevant passages with associated references from a knowledge base constituted by scientific literature. In the second stage, these passages, alongside the initial query, are processed by LLMs to generate a contextually relevant rich text (GenText). In the last stage, the documents to be retrieved are evaluated and ranked both from the point of view of topical relevance and factual accuracy by means of their comparison with GenText, either through stance detection or semantic similarity. In addition to calculating factual accuracy, GenText can offer a layer of explainability for it, aiding users in understanding the reasoning behind the retrieval. Experimental evaluation of our model on benchmark datasets and against baseline models demonstrates its effectiveness in enhancing the retrieval of both topically relevant and factually accurate health information, thus presenting a significant step forward in the health misinformation mitigation problem. 

**Abstract (ZH)**: 互联网健康信息的指数级增长，以及其被非专业人士广泛应用的趋势，突显了迫切需要发展高级的健康信息检索模型的重要性。这些模型不仅要考虑检索信息的相关性，还要确保信息的准确性，因为健康错误信息的风险隐患不容忽视。为了应对这一挑战，本文提出了一种基于检索增强生成（RAG）方法的解决方案，利用生成型大规模语言模型（LLMs）的能力来提升基于科学研究证据的健康相关文档的检索质量。具体而言，我们提出了一种三阶段模型：首先，利用用户的查询从科学文献构成的知识库中检索相关段落及其引用文献；其次，这些段落与初始查询一起，通过LLMs生成上下文相关的内容丰富文本（GenText）；最后，通过与GenText的比较，无论是通过立场检测还是语义相似性，评估并排序待检索的文档，从相关性和准确性的角度进行综合考量。除了计算事实准确性外，GenText还可以提供一层解释性，帮助用户理解检索背后的理由。我们模型在基准数据集上的实验评估和与基线模型的对比表明，它在增强Topically Relevant和Factually Accurate健康信息检索方面具有明显优势，从而在健康错误信息防控问题上迈出了重要一步。 

---
