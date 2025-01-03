# OmniChat: Enhancing Spoken Dialogue Systems with Scalable Synthetic Data for Diverse Scenarios 

**Title (ZH)**: OmniChat：通过可扩展的合成数据增强多场景口语对话系统 

**Authors**: Xize Cheng, Dongjie Fu, Xiaoda Yang, Minghui Fang, Ruofan Hu, Jingyu Lu, Bai Jionghao, Zehan Wang, Shengpeng Ji, Rongjie Huang, Linjun Li, Yu Chen, Tao Jin, Zhou Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2501.01384)  

**Abstract**: With the rapid development of large language models, researchers have created increasingly advanced spoken dialogue systems that can naturally converse with humans. However, these systems still struggle to handle the full complexity of real-world conversations, including audio events, musical contexts, and emotional expressions, mainly because current dialogue datasets are constrained in both scale and scenario diversity. In this paper, we propose leveraging synthetic data to enhance the dialogue models across diverse scenarios. We introduce ShareChatX, the first comprehensive, large-scale dataset for spoken dialogue that spans diverse scenarios. Based on this dataset, we introduce OmniChat, a multi-turn dialogue system with a heterogeneous feature fusion module, designed to optimize feature selection in different dialogue contexts. In addition, we explored critical aspects of training dialogue systems using synthetic data. Through comprehensive experimentation, we determined the ideal balance between synthetic and real data, achieving state-of-the-art results on the real-world dialogue dataset DailyTalk. We also highlight the crucial importance of synthetic data in tackling diverse, complex dialogue scenarios, especially those involving audio and music. For more details, please visit our demo page at \url{this https URL}. 

**Abstract (ZH)**: 随着大型语言模型的快速发展，研究人员已经创建了越来越先进的语音对话系统，能够自然地与人类进行对话。然而，这些系统仍然难以应对真实世界对话的全部复杂性，包括音频事件、音乐背景和情感表达等内容，主要原因在于当前的对话数据集在规模和场景多样性方面存在限制。在本文中，我们提出利用合成数据来增强多场景下的对话模型。我们介绍了ShareChatX，这一首个覆盖多种场景的全面、大规模语音对话数据集。基于此数据集，我们提出了OmniChat，一种具有异构特征融合模块的多轮对话系统，旨在优化不同对话场景下的特征选择。此外，我们还探讨了使用合成数据训练对话系统的关键方面。通过全面的实验，我们确定了合成数据和真实数据之间的理想平衡，在真实世界对话数据集DailyTalk上取得了最先进的结果。我们还强调了合成数据在处理各种复杂对话场景中的关键重要性，尤其是在涉及音频和音乐的场景中。欲了解更多信息，请访问我们的演示页面：\url{this https URL}。 

---
# Training Medical Large Vision-Language Models with Abnormal-Aware Feedback 

**Title (ZH)**: 带有异常感知反馈训练的医疗大规模视觉-语言模型 

**Authors**: Yucheng Zhou, Lingran Song, Jianbing Shen  

**Link**: [PDF](https://arxiv.org/pdf/2501.01377)  

**Abstract**: Existing Medical Large Vision-Language Models (Med-LVLMs), which encapsulate extensive medical knowledge, demonstrate excellent capabilities in understanding medical images and responding to human queries based on these images. However, there remain challenges in visual localization in medical images, which is crucial for abnormality detection and interpretation. To address these issues, we propose a novel UMed-LVLM designed with Unveiling Medical abnormalities. Specifically, we collect a Medical Abnormalities Unveiling (MAU) dataset and propose a two-stage training method for UMed-LVLM training. To collect MAU dataset, we propose a prompt method utilizing the GPT-4V to generate diagnoses based on identified abnormal areas in medical images. Moreover, the two-stage training method includes Abnormal-Aware Instruction Tuning and Abnormal-Aware Rewarding, comprising Abnormal Localization Rewarding and Vision Relevance Rewarding. Experimental results demonstrate that our UMed-LVLM surpasses existing Med-LVLMs in identifying and understanding medical abnormality. In addition, this work shows that enhancing the abnormality detection capabilities of Med-LVLMs significantly improves their understanding of medical images and generalization capability. 

**Abstract (ZH)**: 现有的医疗大规模视觉-语言模型（Med-LVLMs）集成了丰富的医学知识，展示了在理解医学图像和基于这些图像回答人类查询方面的出色能力。然而，在医学图像中的视觉定位方面仍存在挑战，这对于异常检测和解释至关重要。为了解决这些问题，我们提出了一种名为UMed-LVLM的新型模型，专门用于揭示医学异常。具体而言，我们收集了一个医学异常揭示（MAU）数据集，并提出了一种双阶段训练方法来训练UMed-LVLM。为了收集MAU数据集，我们提出了一种利用GPT-4V的提示方法，根据医学图像中识别出的异常区域生成诊断。此外，双阶段训练方法包括异常意识指令调优和异常意识奖励，其中包括异常定位奖励和视觉相关性奖励。实验结果表明，我们的UMed-LVLM在识别和理解医学异常方面优于现有的Med-LVLMs。此外，本文展示了增强Med-LVLMs的异常检测能力显著提高了它们对医学图像的理解能力和泛化能力。 

---
# Aligning Large Language Models for Faithful Integrity Against Opposing Argument 

**Title (ZH)**: 将大型语言模型与反对论点的忠实完整性对齐 

**Authors**: Yong Zhao, Yang Deng, See-Kiong Ng, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2501.01336)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities in complex reasoning tasks. However, they can be easily misled by unfaithful arguments during conversations, even when their original statements are correct. To this end, we investigate the problem of maintaining faithful integrity in LLMs. This involves ensuring that LLMs adhere to their faithful statements in the face of opposing arguments and are able to correct their incorrect statements when presented with faithful arguments. In this work, we propose a novel framework, named Alignment for Faithful Integrity with Confidence Estimation (AFICE), which aims to align the LLM responses with faithful integrity. Specifically, AFICE first designs a Bilateral Confidence Estimation (BCE) approach for estimating the uncertainty of each response generated by the LLM given a specific context, which simultaneously estimate the model's confidence to the question based on the internal states during decoding as well as to the answer based on cumulative probability ratios. With the BCE, we construct a conversational preference dataset composed of context, original statement, and argument, which is adopted for aligning the LLM for faithful integrity using Direct Preference Optimization (DPO). Extensive experimental results on a wide range of benchmarks demonstrate significant improvements in the LLM's ability to maintain faithful responses when encountering opposing arguments, ensuring both the practical utility and trustworthiness of LLMs in complex interactive settings. Code and data will be released via this https URL 

**Abstract (ZH)**: 大型语言模型（LLMs）在复杂推理任务中展现了令人印象深刻的性能。然而，在对话中，它们可能会被不忠实的论证误导，即使它们的原始声明是正确的。为解决这一问题，我们探讨了保持LLM忠实完整性的挑战。这涉及到确保在面对反对方论点时，LLM能够坚守其忠实声明，并在被忠实论点纠正时能够修正其错误声明。在此项研究中，我们提出了一个新颖的框架，称为“基于置信估计的忠实完整性对齐”（AFICE，Alignment for Faithful Integrity with Confidence Estimation），旨在使LLM响应与忠实完整性保持一致。具体来说，AFICE 首先设计了一种双边置信估计（BCE，Bilateral Confidence Estimation）方法，用于估计给定特定上下文中LLM生成的每个响应的不确定性，并同时根据解码过程中模型对问题的置信度和累积概率比来估计模型对答案的置信度。通过BCE，我们构建了一个对话偏好数据集，包含上下文、原始声明和论点，该数据集用于利用直接偏好优化（DPO，Direct Preference Optimization）对齐LLM以保持忠实完整性。广泛基准上的大量实验结果表明，当遇到反对论点时，该框架显著提高了LLM维护忠实响应的能力，确保了LLM在复杂交互环境中的实用性和可信度。代码和数据将在此处发布：https://... 

---
# Decoding Knowledge in Large Language Models: A Framework for Categorization and Comprehension 

**Title (ZH)**: 大型语言模型中的知识解码：一种分类和理解的框架 

**Authors**: Yanbo Fang, Ruixiang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2501.01332)  

**Abstract**: Understanding how large language models (LLMs) acquire, retain, and apply knowledge remains an open challenge. This paper introduces a novel framework, K-(CSA)^2, which categorizes LLM knowledge along two dimensions: correctness and confidence. The framework defines six categories of knowledge, ranging from highly confident correctness to confidently held misconceptions, enabling a nuanced evaluation of model comprehension beyond binary accuracy. Using this framework, we demonstrate how techniques like chain-of-thought prompting and reinforcement learning with human feedback fundamentally alter the knowledge structures of internal (pre-trained) and external (context-dependent) knowledge in LLMs. CoT particularly enhances base model performance and shows synergistic benefits when applied to aligned LLMs. Moreover, our layer-wise analysis reveals that higher layers in LLMs encode more high-confidence knowledge, while low-confidence knowledge tends to emerge in middle-to-lower layers. 

**Abstract (ZH)**: 理解大型语言模型（LLMs）如何获取、保留和应用知识仍然是一项开放性的挑战。本文介绍了一种新的框架，K-(CSA)²，该框架按照正确性和信心两个维度对LLM的知识进行分类。该框架定义了六个知识类别，从高度自信的正确性到有把握的误解，从而实现对模型理解程度的细腻评估，超越了二元准确性的评价。通过使用这种框架，我们展示了诸如思维链提示和带有人类反馈的强化学习等技术如何根本性地改变内部（预训练的）和外部（上下文相关的）知识结构。思维链特别增强了基础模型的表现，并且在应用于对齐的语言模型时显示出协同效益。此外，我们的逐层分析表明，LLM中的较高层编码了更多的高信心知识，而低信心知识容易在中间到较低层出现。 

---
# Think More, Hallucinate Less: Mitigating Hallucinations via Dual Process of Fast and Slow Thinking 

**Title (ZH)**: 思考更多，幻觉更少：通过快速思维和慢速思维双重过程减轻幻觉问题 

**Authors**: Xiaoxue Cheng, Junyi Li, Wayne Xin Zhao, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2501.01306)  

**Abstract**: Large language models (LLMs) demonstrate exceptional capabilities, yet still face the hallucination issue. Typical text generation approaches adopt an auto-regressive generation without deliberate reasoning, which often results in untrustworthy and factually inaccurate responses. In this paper, we propose HaluSearch, a novel framework that incorporates tree search-based algorithms (e.g. MCTS) to enable an explicit slow thinking generation process for mitigating hallucinations of LLMs during inference. Specifically, HaluSearch frames text generation as a step-by-step reasoning process, using a self-evaluation reward model to score each generation step and guide the tree search towards the most reliable generation pathway for fully exploiting the internal knowledge of LLMs. To balance efficiency and quality, we introduce a hierarchical thinking system switch mechanism inspired by the dual process theory in cognitive science, which dynamically alternates between fast and slow thinking modes at both the instance and step levels, adapting to the complexity of questions and reasoning states. We conduct extensive experiments on both English and Chinese datasets and the results show that our approach significantly outperforms baseline approaches. 

**Abstract (ZH)**: 大型语言模型（LLMs）展现出卓越的能力，但仍面临幻觉问题。典型的文字生成方法采用自动回归生成而不进行刻意的推理，这往往导致不可靠且事实不准确的回答。在本文中，我们提出了一种名为HaluSearch的新型框架，该框架结合了基于树搜索的算法（例如MCTS），以在推理过程中通过显式的慢思考生成过程来减轻LLMs的幻觉问题。具体而言，HaluSearch将文本生成构想为逐步推理过程，使用自我评估奖励模型来评估每一步生成，并引导树搜索通往最可靠的生成路径，以便充分利用LLMs的内部知识。为了平衡效率和质量，我们引入了一种受认知科学中的双重过程理论启发的分层思考系统切换机制，在实例和步骤层面上动态地交替使用快速和慢速思考模式，以适应问题和推理状态的复杂性。我们对英文和中文数据集进行了广泛的实验，并且结果显示，我们的方法显著优于基线方法。 

---
# Large Language Models for Mental Health Diagnostic Assessments: Exploring The Potential of Large Language Models for Assisting with Mental Health Diagnostic Assessments -- The Depression and Anxiety Case 

**Title (ZH)**: 大型语言模型在心理健康诊断评估中的应用：探索大型语言模型在辅助心理健康诊断评估中的潜力——以抑郁和焦虑为例 

**Authors**: Kaushik Roy, Harshul Surana, Darssan Eswaramoorthi, Yuxin Zi, Vedant Palit, Ritvik Garimella, Amit Sheth  

**Link**: [PDF](https://arxiv.org/pdf/2501.01305)  

**Abstract**: Large language models (LLMs) are increasingly attracting the attention of healthcare professionals for their potential to assist in diagnostic assessments, which could alleviate the strain on the healthcare system caused by a high patient load and a shortage of providers. For LLMs to be effective in supporting diagnostic assessments, it is essential that they closely replicate the standard diagnostic procedures used by clinicians. In this paper, we specifically examine the diagnostic assessment processes described in the Patient Health Questionnaire-9 (PHQ-9) for major depressive disorder (MDD) and the Generalized Anxiety Disorder-7 (GAD-7) questionnaire for generalized anxiety disorder (GAD). We investigate various prompting and fine-tuning techniques to guide both proprietary and open-source LLMs in adhering to these processes, and we evaluate the agreement between LLM-generated diagnostic outcomes and expert-validated ground truth. For fine-tuning, we utilize the Mentalllama and Llama models, while for prompting, we experiment with proprietary models like GPT-3.5 and GPT-4o, as well as open-source models such as llama-3.1-8b and mixtral-8x7b. 

**Abstract (ZH)**: 大型语言模型（LLMs）日益受到医疗专业人员的关注，它们有可能在诊断评估中提供帮助，从而缓解由于患者数量过多和医疗提供者短缺而导致的医疗系统压力。为了使LLMs在支持诊断评估方面有效，它们必须紧密复制临床医生使用的标准诊断程序。本文具体研究了用于重度抑郁症（MDD）的患者健康问卷-9（PHQ-9）和用于广泛性焦虑障碍（GAD）的一般化焦虑问卷-7（GAD-7）中的诊断评估过程。我们探讨了各种提示和微调技术，以引导自有的和开源的LLMs遵守这些过程，并且评估了LLM生成的诊断结果与专家验证的黄金标准之间的一致性。在微调方面，我们使用了Mentalllama和Llama模型，而在提示方面，我们尝试了诸如GPT-3.5和GPT-4o等自有模型，以及诸如llama-3.1-8b和mixtral-8x7b等开源模型。 

---
# Citations and Trust in LLM Generated Responses 

**Title (ZH)**: LLM生成响应中的引用与信任问题 

**Authors**: Yifan Ding, Matthew Facciani, Amrit Poudel, Ellen Joyce, Salvador Aguinaga, Balaji Veeramani, Sanmitra Bhattacharya, Tim Weninger  

**Link**: [PDF](https://arxiv.org/pdf/2501.01303)  

**Abstract**: Question answering systems are rapidly advancing, but their opaque nature may impact user trust. We explored trust through an anti-monitoring framework, where trust is predicted to be correlated with presence of citations and inversely related to checking citations. We tested this hypothesis with a live question-answering experiment that presented text responses generated using a commercial Chatbot along with varying citations (zero, one, or five), both relevant and random, and recorded if participants checked the citations and their self-reported trust in the generated responses. We found a significant increase in trust when citations were present, a result that held true even when the citations were random; we also found a significant decrease in trust when participants checked the citations. These results highlight the importance of citations in enhancing trust in AI-generated content. 

**Abstract (ZH)**: 问答系统正在迅速发展，但其不透明的性质可能会对用户信任产生影响。我们通过反监控框架探索了信任问题，该框架假设信任与引文的出现正相关，而与验证引文的行为负相关。我们通过一项现场问答实验进行了测试，该实验展示了使用商业聊天机器人生成的文本回答，并提供了不同数量的引文（零篇、一篇或五篇，既有相关引文也有随机引文），记录了参与者是否检查了引文以及他们对生成回答的信任程度。我们发现，当引文存在时，信任显著增加，这一结果即使在引文是随机的情况下依然成立；我们还发现，当参与者检查引文时，信任显著下降。这些结果突显了引文在增强人工智能生成内容的信任方面的重要性。 

---
# ToolComp: A Multi-Tool Reasoning & Process Supervision Benchmark 

**Title (ZH)**: ToolComp：一个多种工具推理与过程监督基准 

**Authors**: Vaskar Nath, Pranav Raja, Claire Yoon, Sean Hendryx  

**Link**: [PDF](https://arxiv.org/pdf/2501.01290)  

**Abstract**: Despite recent advances in AI, the development of systems capable of executing complex, multi-step reasoning tasks involving multiple tools remains a significant challenge. Current benchmarks fall short in capturing the real-world complexity of tool-use reasoning, where verifying the correctness of not only the final answer but also the intermediate steps is important for evaluation, development, and identifying failures during inference time. To bridge this gap, we introduce ToolComp, a comprehensive benchmark designed to evaluate multi-step tool-use reasoning. ToolComp is developed through a collaboration between models and human annotators, featuring human-edited/verified prompts, final answers, and process supervision labels, allowing for the evaluation of both final outcomes and intermediate reasoning. Evaluation across six different model families demonstrates the challenging nature of our dataset, with the majority of models achieving less than 50% accuracy. Additionally, we generate synthetic training data to compare the performance of outcome-supervised reward models (ORMs) with process-supervised reward models (PRMs) to assess their ability to improve complex tool-use reasoning as evaluated by ToolComp. Our results show that PRMs generalize significantly better than ORMs, achieving a 19% and 11% improvement in rank@1 accuracy for ranking base and fine-tuned model trajectories, respectively. These findings highlight the critical role of process supervision in both the evaluation and training of AI models, paving the way for more robust and capable systems in complex, multi-step tool-use tasks. 

**Abstract (ZH)**: 尽管近期人工智能取得了进展，但开发能够执行涉及多种工具的复杂多步推理任务的系统仍然是一个重大挑战。当前的基准测试在捕捉实际场景中工具使用推理的复杂性方面仍显不足，在实际推理过程中，不仅要验证最终答案的正确性，还需要验证中间步骤的正确性，这对于评估、开发和推理时间中的错误识别都至关重要。为应对这一挑战，我们提出了 ToolComp，这是一个全面的基准测试，旨在评估多步骤工具使用推理能力。ToolComp 通过模型与人工标注者的合作开发而成，包含人工编辑/验证的提示、最终答案及过程监控标签，从而能够评估最终结果和中间推理过程。针对六个不同模型家族的评估显示，该数据集具有很高的挑战性，大多数模型的准确率低于50%。此外，我们生成了合成训练数据，比较基于最终结果监督的奖励模型（ORM）和基于过程监督的奖励模型（PRM）的表现，评估它们是否能够通过 ToolComp 评估的复杂工具使用推理任务得到改进。结果表明，PRM 在泛化能力方面明显优于 ORM，在排序基线模型和微调模型轨迹上的排名 @1 准确率分别提高了19%和11%。这些发现突显了过程监督在 AI 模型评估和训练中的关键作用，并为复杂多步骤工具使用任务中更 robust 和强大的系统铺平了道路。 

---
# NeutraSum: A Language Model can help a Balanced Media Diet by Neutralizing News Summaries 

**Title (ZH)**: NeutraSum：语言模型可以通过中性化新闻摘要来帮助实现平衡的媒体饮食 

**Authors**: Xi Luo, Junjie Liu, Sirong Wu, Yuhui Deng  

**Link**: [PDF](https://arxiv.org/pdf/2501.01284)  

**Abstract**: Media bias in news articles arises from the political polarisation of media outlets, which can reinforce societal stereotypes and beliefs. Reporting on the same event often varies significantly between outlets, reflecting their political leanings through polarised language and focus. Although previous studies have attempted to generate bias-free summaries from multiperspective news articles, they have not effectively addressed the challenge of mitigating inherent media bias. To address this gap, we propose \textbf{NeutraSum}, a novel framework that integrates two neutrality losses to adjust the semantic space of generated summaries, thus minimising media bias. These losses, designed to balance the semantic distances across polarised inputs and ensure alignment with expert-written summaries, guide the generation of neutral and factually rich summaries. To evaluate media bias, we employ the political compass test, which maps political leanings based on economic and social dimensions. Experimental results on the Allsides dataset demonstrate that NeutraSum not only improves summarisation performance but also achieves significant reductions in media bias, offering a promising approach for neutral news summarisation. 

**Abstract (ZH)**: 新闻文章中的媒体偏见来源于媒体机构的政治极化，这可能强化社会刻板印象和信念。报道同一事件时，不同机构之间往往存在显著差异，反映出它们的政治倾向通过极化的语言和重点来体现。尽管之前的研究尝试从多视角新闻文章中生成无偏见的摘要，但它们并未有效解决减轻固有媒体偏见的挑战。为此，我们提出了一种新颖的框架\textbf{NeutraSum}，该框架结合了两种中立性损失，以调节生成摘要的语义空间，从而最小化媒体偏见。这些损失旨在平衡极化输入之间的语义距离，并与专家撰写的摘要保持一致，从而引导生成客观且富含事实的摘要。

为了评估媒体偏见，我们采用了政治罗盘测试，该测试根据经济和社会维度映射政治倾向。在Allsides数据集上的实验结果表明，NeutraSum不仅提高了摘要生成性能，还在显著降低媒体偏见方面取得了显著成果，为中立新闻摘要提供了有前景的方法。 

---
# Does a Large Language Model Really Speak in Human-Like Language? 

**Title (ZH)**: 大型语言模型真的能够使用类人类语言表达吗？ 

**Authors**: Mose Park, Yunjin Choi, Jong-June Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2501.01273)  

**Abstract**: Large Language Models (LLMs) have recently emerged, attracting considerable attention due to their ability to generate highly natural, human-like text. This study compares the latent community structures of LLM-generated text and human-written text within a hypothesis testing procedure. Specifically, we analyze three text sets: original human-written texts ($\mathcal{O}$), their LLM-paraphrased versions ($\mathcal{G}$), and a twice-paraphrased set ($\mathcal{S}$) derived from $\mathcal{G}$. Our analysis addresses two key questions: (1) Is the difference in latent community structures between $\mathcal{O}$ and $\mathcal{G}$ the same as that between $\mathcal{G}$ and $\mathcal{S}$? (2) Does $\mathcal{G}$ become more similar to $\mathcal{O}$ as the LLM parameter controlling text variability is adjusted? The first question is based on the assumption that if LLM-generated text truly resembles human language, then the gap between the pair ($\mathcal{O}$, $\mathcal{G}$) should be similar to that between the pair ($\mathcal{G}$, $\mathcal{S}$), as both pairs consist of an original text and its paraphrase. The second question examines whether the degree of similarity between LLM-generated and human text varies with changes in the breadth of text generation. To address these questions, we propose a statistical hypothesis testing framework that leverages the fact that each text has corresponding parts across all datasets due to their paraphrasing relationship. This relationship enables the mapping of one dataset's relative position to another, allowing two datasets to be mapped to a third dataset. As a result, both mapped datasets can be quantified with respect to the space characterized by the third dataset, facilitating a direct comparison between them. Our results indicate that GPT-generated text remains distinct from human-authored text. 

**Abstract (ZH)**: 大型语言模型（LLMs）近年来崭露头角，因其能够生成高度自然、类人语言的文本而备受关注。本研究在假设检验的框架下，比较了LLM生成文本与人类撰写的文本之间的潜在社区结构。具体来说，我们分析了三个文本集：原始的人类撰写的文本集（$\mathcal{O}$），其LLM改写的版本（$\mathcal{G}$），以及由$\mathcal{G}$二次改写的集子（$\mathcal{S}$）。我们的分析旨在回答两个关键问题：（1）$\mathcal{O}$和$\mathcal{G}$之间的潜在社区结构差异是否与$\mathcal{G}$和$\mathcal{S}$之间的差异相同？（2）随着控制文本变异性的LLM参数调整，$\mathcal{G}$是否变得与$\mathcal{O}$更相似？第一个问题基于这样的假设：如果LLM生成的文本确实类似于人类语言，那么$\mathcal{O}$和$\mathcal{G}$这对文本与$\mathcal{G}$和$\mathcal{S}$这对文本之间的差距应该相似，因为这两对都包含一个原始文本及其改写版本。第二个问题探讨的是，LLM生成的文本与人类文本之间的相似度是否随文本生成范围的变化而变化。为了解答这些问题，我们提出了一种统计假设检验框架，利用每个文本集在所有数据集中对应部分的事实，这得益于它们的改写关系。这种关系使得一个数据集可以映射到另一个数据集的位置，允许两个数据集映射到第三个数据集。结果，两个映射后的数据集都可以根据由第三个数据集定义的空间进行量化，从而可以直接比较它们。研究结果表明，GPT生成的文本仍然与人类撰写的文本有所不同。 

---
# ProgCo: Program Helps Self-Correction of Large Language Models 

**Title (ZH)**: ProgCo: 程序辅助大型语言模型自我修正 

**Authors**: Xiaoshuai Song, Yanan Wu, Weixun Wang, Jiaheng Liu, Wenbo Su, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2501.01264)  

**Abstract**: Self-Correction aims to enable large language models (LLMs) to self-verify and self-refine their initial responses without external feedback. However, LLMs often fail to effectively self-verify and generate correct feedback, further misleading refinement and leading to the failure of self-correction, especially in complex reasoning tasks. In this paper, we propose Program-driven Self-Correction (ProgCo). First, program-driven verification (ProgVe) achieves complex verification logic and extensive validation through self-generated, self-executing verification pseudo-programs. Then, program-driven refinement (ProgRe) receives feedback from ProgVe, conducts dual reflection and refinement on both responses and verification programs to mitigate misleading of incorrect feedback in complex reasoning tasks. Experiments on three instruction-following and mathematical benchmarks indicate that ProgCo achieves effective self-correction, and can be further enhance performance when combined with real program tools. 

**Abstract (ZH)**: 自修正的目标在于使大规模语言模型（LLMs）能够在其初始响应无需外部反馈的情况下进行自我验证和自我完善。然而，LLMs 在自我验证和生成正确反馈方面常常无法有效执行，这可能会进一步导致自我修正过程的误导和失败，尤其是在复杂的推理任务中。本文中，我们提出了一种程序驱动的自修正方法（ProgCo）。首先，程序驱动的验证（ProgVe）通过自动生成并自执行的验证伪程序实现了复杂的验证逻辑和广泛的验证。其次，程序驱动的完善（ProgRe）接收ProgVe的反馈，对响应和验证程序进行双重反思和完善，以减轻复杂推理任务中错误反馈带来的误导。在三个指令遵循和数学基准测试中的实验表明，ProgCo 实现了有效的自我修正，并且当与实际程序工具结合时可以进一步提高性能。 

---
# CodeElo: Benchmarking Competition-level Code Generation of LLMs with Human-comparable Elo Ratings 

**Title (ZH)**: CodeElo：使用类人类 Elo 排名Benchmark大规模代码生成竞赛级LLM性能 

**Authors**: Shanghaoran Quan, Jiaxi Yang, Bowen Yu, Bo Zheng, Dayiheng Liu, An Yang, Xuancheng Ren, Bofei Gao, Yibo Miao, Yunlong Feng, Zekun Wang, Jian Yang, Zeyu Cui, Yang Fan, Yichang Zhang, Binyuan Hui, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2501.01257)  

**Abstract**: With the increasing code reasoning capabilities of existing large language models (LLMs) and breakthroughs in reasoning models like OpenAI o1 and o3, there is a growing need to develop more challenging and comprehensive benchmarks that effectively test their sophisticated competition-level coding abilities. Existing benchmarks, like LiveCodeBench and USACO, fall short due to the unavailability of private test cases, lack of support for special judges, and misaligned execution environments. To bridge this gap, we introduce CodeElo, a standardized competition-level code generation benchmark that effectively addresses all these challenges for the first time. CodeElo benchmark is mainly based on the official CodeForces platform and tries to align with the platform as much as possible. We compile the recent six months of contest problems on CodeForces with detailed information such as contest divisions, problem difficulty ratings, and problem algorithm tags. We introduce a unique judging method in which problems are submitted directly to the platform and develop a reliable Elo rating calculation system that aligns with the platform and is comparable with human participants but has lower variance. By testing on our CodeElo, we provide the Elo ratings of 30 existing popular open-source and 3 proprietary LLMs for the first time. The results show that o1-mini and QwQ-32B-Preview stand out significantly, achieving Elo ratings of 1578 and 1261, respectively, while other models struggle even with the easiest problems, placing in the lowest 20 percent among all human participants. Detailed analysis experiments are also conducted to provide insights into performance across algorithms and comparisons between using C++ and Python, which can suggest directions for future studies. 

**Abstract (ZH)**: 随着现有大型语言模型（LLMs）代码推理能力的不断增强以及OpenAI的o1和o3推理模型的重大突破，开发更加具有挑战性和全面性的基准测试的需求也日益增长，以有效测试其复杂的竞争级编程能力。现有的基准测试，如LiveCodeBench和USACO，由于缺乏私有测试案例、不支持特殊裁判以及执行环境不匹配等原因而不足。为解决这些问题，我们引入了CodeElo，这是一种标准化的竞争级代码生成基准测试，首次能够有效应对上述所有挑战。CodeElo基准测试主要基于官方的CodeForces平台，并尽可能地与该平台保持一致。我们编译了CodeForces过去六个月的比赛题目，并详细记录了比赛分区、问题难度评级和算法标签等信息。我们引入了一种独特的评判方法，即将问题直接提交给平台，并开发了一套可靠的Elo等级计算系统，该系统与平台保持一致，并且与人类参赛者具有可比性，但具有较低的方差。通过在CodeElo上进行测试，我们首次提供了30个现有流行的开源和3个专有LLMs的Elo等级。结果表明，o1-mini和QwQ-32B-Preview表现出色，分别获得了1578和1261的Elo等级，而其他模型即使是面对最简单的题目也表现不佳，甚至位于所有人类参赛者的最低20%。我们还进行了详细的分析实验，以提供不同算法性能和使用C++与Python的不同方式之间的见解，这可以为未来的研究提供方向。 

---
# Digital Guardians: Can GPT-4, Perspective API, and Moderation API reliably detect hate speech in reader comments of German online newspapers? 

**Title (ZH)**: 数字守护者：GPT-4、Perspective API和Moderation API能否可靠地检测德国在线报纸读者评论中的仇恨言论？ 

**Authors**: Manuel Weber, Moritz Huber, Maximilian Auch, Alexander Döschl, Max-Emanuel Keller, Peter Mandl  

**Link**: [PDF](https://arxiv.org/pdf/2501.01256)  

**Abstract**: In recent years, toxic content and hate speech have become widespread phenomena on the internet. Moderators of online newspapers and forums are now required, partly due to legal regulations, to carefully review and, if necessary, delete reader comments. This is a labor-intensive process. Some providers of large language models already offer solutions for automated hate speech detection or the identification of toxic content. These include GPT-4o from OpenAI, Jigsaw's (Google) Perspective API, and OpenAI's Moderation API. Based on the selected German test dataset HOCON34k, which was specifically created for developing tools to detect hate speech in reader comments of online newspapers, these solutions are compared with each other and against the HOCON34k baseline. The test dataset contains 1,592 annotated text samples. For GPT-4o, three different promptings are used, employing a Zero-Shot, One-Shot, and Few-Shot approach. The results of the experiments demonstrate that GPT-4o outperforms both the Perspective API and the Moderation API, and exceeds the HOCON34k baseline by approximately 5 percentage points, as measured by a combined metric of MCC and F2-score. 

**Abstract (ZH)**: 近年来，有毒内容和仇恨言论在网络空间中变得越来越普遍。在线报纸和论坛的管理员，部分由于法律法规的要求，现在需要仔细审查读者评论，并在必要时删除有害内容。这一过程是劳动密集型的。一些大型语言模型的提供商已经提出了自动检测仇恨言论或有毒内容的解决方案。这些解决方案包括OpenAI的GPT-4o、Jigsaw（Google）的Perspective API和OpenAI的 Moderation API。基于专门用于开发检测在线报纸读者评论中仇恨言论的工具的德语测试数据集HOCON34k，这些解决方案进行了比较，并与HOCON34k基线进行了对比。测试数据集包含1,592个标注过的文本样本。对于GPT-4o，采用了三种不同的提示方法，分别是零样本（Zero-Shot）、单样本（One-Shot）和少样本（Few-Shot）方法。实验结果表明，GPT-4o在综合MCC和F2分数的衡量标准下，优于Perspective API和Moderation API，并且比HOCON34k基线高出约5个百分点。 

---
# Large Language Model-Enhanced Symbolic Reasoning for Knowledge Base Completion 

**Title (ZH)**: 大型语言模型增强的符号推理在知识库补全中的应用 

**Authors**: Qiyuan He, Jianfei Yu, Wenya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.01246)  

**Abstract**: Integrating large language models (LLMs) with rule-based reasoning offers a powerful solution for improving the flexibility and reliability of Knowledge Base Completion (KBC). Traditional rule-based KBC methods offer verifiable reasoning yet lack flexibility, while LLMs provide strong semantic understanding yet suffer from hallucinations. With the aim of combining LLMs' understanding capability with the logical and rigor of rule-based approaches, we propose a novel framework consisting of a Subgraph Extractor, an LLM Proposer, and a Rule Reasoner. The Subgraph Extractor first samples subgraphs from the KB. Then, the LLM uses these subgraphs to propose diverse and meaningful rules that are helpful for inferring missing facts. To effectively avoid hallucination in LLMs' generations, these proposed rules are further refined by a Rule Reasoner to pinpoint the most significant rules in the KB for Knowledge Base Completion. Our approach offers several key benefits: the utilization of LLMs to enhance the richness and diversity of the proposed rules and the integration with rule-based reasoning to improve reliability. Our method also demonstrates strong performance across diverse KB datasets, highlighting the robustness and generalizability of the proposed framework. 

**Abstract (ZH)**: 将大型语言模型（LLM）与基于规则的推理相结合为提高知识库完成（KBC）的灵活性和可靠性提供了一种强大的解决方案。传统的基于规则的KBC方法能够验证推理过程但缺乏灵活性，而LLM则提供了强大的语义理解能力，但却容易出现幻觉。为了解决这一点，我们旨在结合LLM的理解能力和基于规则方法的逻辑性和严谨性，提出了一种新的框架，该框架包括子图提取器、LLM 提出者和规则推理器。首先，子图提取器从知识库中抽取子图。然后，LLM 使用这些子图提出多样且有意义的规则，这些规则有助于推断缺失的事实。为了有效避免LLM生成过程中出现的幻觉，这些提出的规则将由规则推理器进一步精炼，以指出知识库中最关键的规则，从而促进知识库完成。我们的方法提供了几个关键优势：通过利用LLM增加所提出规则的丰富性和多样性，并与基于规则的推理相结合以提高可靠性。此外，我们的方法在多种不同的知识库数据集上表现出强大性能，这凸显了提出框架的稳健性和通用性。 

---
# Automated Self-Refinement and Self-Correction for LLM-based Product Attribute Value Extraction 

**Title (ZH)**: 基于LLM的产品属性值自动化自我精炼与自我修正 

**Authors**: Alexander Brinkmann, Christian Bizer  

**Link**: [PDF](https://arxiv.org/pdf/2501.01237)  

**Abstract**: Structured product data, in the form of attribute-value pairs, is essential for e-commerce platforms to support features such as faceted product search and attribute-based product comparison. However, vendors often provide unstructured product descriptions, making attribute value extraction necessary to ensure data consistency and usability. Large language models (LLMs) have demonstrated their potential for product attribute value extraction in few-shot scenarios. Recent research has shown that self-refinement techniques can improve the performance of LLMs on tasks such as code generation and text-to-SQL translation. For other tasks, the application of these techniques has resulted in increased costs due to processing additional tokens, without achieving any improvement in performance. This paper investigates applying two self-refinement techniques, error-based prompt rewriting and self-correction, to the product attribute value extraction task. The self-refinement techniques are evaluated across zero-shot, few-shot in-context learning, and fine-tuning scenarios using GPT-4o. The experiments show that both self-refinement techniques have only a marginal impact on the model's performance across the different scenarios, while significantly increasing processing costs. For scenarios with training data, fine-tuning yields the highest performance, while the ramp-up costs of fine-tuning are balanced out as the amount of product descriptions increases. 

**Abstract (ZH)**: 结构化产品数据，以属性-值对的形式存在，对于电商平台支持功能例如分类产品搜索和基于属性的产品比较至关重要。然而，供应商通常提供的是非结构化产品描述，因此需要从非结构化描述中提取属性值以确保数据的一致性和可用性。大规模语言模型（LLMs）已经在少量示例场景下展示了其在产品属性值提取方面的潜力。近期的研究表明，自我修正技术可以提高LLMs在代码生成和文本到SQL翻译等任务上的性能。对于其他任务，这些技术的应用却导致了成本增加，因为处理额外的标记并没有提升性能。本文探讨了应用两种自我修正技术——基于错误的提示重写和自我修正——到产品属性值提取任务中。通过使用GPT-4o的零样本、少量上下文学习和微调场景评估这两大自我修正技术。实验结果显示，在不同的场景中，这两种自我修正技术对模型性能的影响微乎其微，但显著增加了处理成本。在有训练数据的情况下，微调场景获得了最高的性能，而且随着产品描述数量的增加，微调的成本递增效应逐渐被平衡。 

---
# Data Augmentation Techniques for Chinese Disease Name Normalization 

**Title (ZH)**: 中文翻译如下，符合学术规范：

中文标题：中文疾病名称规范化中的数据增强技术

如果这是论文的内容摘要或介绍部分，可以进一步翻译为：

中文内容：本文探讨了中文疾病名称规范化中使用的数据增强技术。 

**Authors**: Wenqian Cui, Xiangling Fu, Shaohui Liu, Mingjun Gu, Xien Liu, Ji Wu, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2501.01195)  

**Abstract**: Disease name normalization is an important task in the medical domain. It classifies disease names written in various formats into standardized names, serving as a fundamental component in smart healthcare systems for various disease-related functions. Nevertheless, the most significant obstacle to existing disease name normalization systems is the severe shortage of training data. Consequently, we present a novel data augmentation approach that includes a series of data augmentation techniques and some supporting modules to help mitigate the problem. Through extensive experimentation, we illustrate that our proposed approach exhibits significant performance improvements across various baseline models and training objectives, particularly in scenarios with limited training data 

**Abstract (ZH)**: 疾病名称规范化是医疗领域的一项重要任务。它将各种格式的疾病名称分类为标准化名称，作为智能医疗系统中多种疾病相关功能的基础组件。然而，现有疾病名称规范化系统的最大障碍是缺乏训练数据。因此，我们提出了一种新的数据增强方法，该方法包括一系列数据增强技术和支持模块，以帮助缓解这一问题。通过广泛的实验，我们表明我们提出的方法在各种基线模型和训练目标上表现出显著的性能提升，特别是在训练数据有限的情况下。 

---
# Blind Men and the Elephant: Diverse Perspectives on Gender Stereotypes in Benchmark Datasets 

**Title (ZH)**: 盲人摸象：基准数据集中性别刻板印象的多元视角 

**Authors**: Mahdi Zakizadeh, Mohammad Taher Pilehvar  

**Link**: [PDF](https://arxiv.org/pdf/2501.01168)  

**Abstract**: The multifaceted challenge of accurately measuring gender stereotypical bias in language models is akin to discerning different segments of a broader, unseen entity. This short paper primarily focuses on intrinsic bias mitigation and measurement strategies for language models, building on prior research that demonstrates a lack of correlation between intrinsic and extrinsic approaches. We delve deeper into intrinsic measurements, identifying inconsistencies and suggesting that these benchmarks may reflect different facets of gender stereotype. Our methodology involves analyzing data distributions across datasets and integrating gender stereotype components informed by social psychology. By adjusting the distribution of two datasets, we achieve a better alignment of outcomes. Our findings underscore the complexity of gender stereotyping in language models and point to new directions for developing more refined techniques to detect and reduce bias. 

**Abstract (ZH)**: 准确测量语言模型中的性别刻板印象偏见是一项多方面的挑战，类似于试图辨别一个更广泛而看不见实体的不同组成部分。本文主要集中在语言模型的内在偏见缓解和测量策略上，基于先前研究显示的内在方法与外在方法之间缺乏相关性的结论。我们深入探讨了内在测量方法，发现了不一致性，并提出这些基准可能反映了不同方面的性别刻板印象。我们的方法包括分析数据集中的数据分布，并结合由社会心理学所提供的性别刻板印象成分。通过调整两个数据集的分布，我们取得了更好的结果对齐。我们的发现强调了语言模型中性别刻板印象复杂性，并指出了开发更精细的检测和减少偏见技术的新方向。 

---
# Leveraging Full Dependency Parsing Graph Information For Biomedical Event Extraction 

**Title (ZH)**: 利用完整依存解析图信息进行生物医学事件抽取 

**Authors**: Farshad Noravesh, Reza Haffari, Ong Huey Fang, Layki Soon, Sailaja Rajalana, Arghya Pal  

**Link**: [PDF](https://arxiv.org/pdf/2501.01158)  

**Abstract**: Many models are proposed in the literature on biomedical event extraction(BEE). Some of them use the shortest dependency path(SDP) information to represent the argument classification task. There is an issue with this representation since even missing one word from the dependency parsing graph may totally change the final prediction. To this end, the full adjacency matrix of the dependency graph is used to embed individual tokens using a graph convolutional network(GCN). An ablation study is also done to show the effect of the dependency graph on the overall performance. The results show a significant improvement when dependency graph information is used. The proposed model slightly outperforms state-of-the-art models on BEE over different datasets. 

**Abstract (ZH)**: 在生物医学事件提取（BEE）文献中，提出了许多模型。一些模型使用最短依赖路径（SDP）信息来表示论元分类任务。这种表示方式存在问题，因为即使依赖解析图中缺失一个单词也可能彻底改变最终预测结果。为此，使用全依赖图的邻接矩阵并通过图卷积网络（GCN）嵌入个体词元。此外，还进行了消融研究以展示依赖图在整体性能中的作用。结果显示，在使用依赖图信息时，性能有了显著提升。提出的模型在不同数据集上的生物医学事件提取中略优于现有最佳模型。 

---
# BlockDialect: Block-wise Fine-grained Mixed Format for Energy-Efficient LLM Inference 

**Title (ZH)**: BlockDialect：面向能效的分块细粒度混合格式大规模语言模型推理 

**Authors**: Wonsuk Jang, Thierry Tambe  

**Link**: [PDF](https://arxiv.org/pdf/2501.01144)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success, but their increasing size poses significant challenges in memory usage and computational costs. Quantizing both weights and activations can address these issues, with fine-grained block-wise quantization emerging as a promising hardware-supported solution to mitigate outliers. However, existing methods struggle to capture nuanced block data distributions. To address this, we propose BlockDialect, a block-wise fine-grained mixed format technique that assigns a per-block optimal number format from formatbook for better data representation. Additionally, we introduce DialectFP4, a formatbook of FP4 variants (akin to dialects) that adapt to diverse data distributions. Importantly, DialectFP4 ensures hardware efficiency by selecting representable values as scaled integers compatible with low-precision integer arithmetic. Furthermore, we propose a two-stage approach for online DialectFP4 activation quantization. BlockDialect achieves 11.40% (6.90%) accuracy gain on the LLaMA3-8B (LLaMA2-7B) model compared to MXFP4 format with a comparable bit usage per data, while being only 5.89% (3.31%) below full precision even when quantizing full-path matrix multiplication. Focusing on how to represent over how to scale, our work presents a promising path for energy-efficient LLM inference. 

**Abstract (ZH)**: 大规模语言模型（LLMs）已经在许多任务上取得了显著的成功，但其不断增加的规模给内存使用和计算成本带来了重大挑战。权重和激活量化的细粒度块化量化已成为一种有前景的硬件支持解决方案，可以缓解这些问题，尤其是针对异常值的处理。然而，现有的方法难以捕捉到精细的块数据分布特性。为解决这一问题，我们提出了一种名为BlockDialect的技术，这是一种细粒度混合格式的块量化方法，能够从formatbook为每个块分配最优的数据表示格式。此外，我们还引入了DialectFP4，这是一种适应不同数据分布的FP4变体（类似于方言）的formatbook。重要的是，DialectFP4通过选择与低精度整数算术兼容的缩放整数来确保硬件效率，从而确保所表示的值是可以表示的。此外，我们提出了一种两阶段的在线DialectFP4激活量化方法。BlockDialect在LLaMA3-8B（LLaMA2-7B）模型上分别相对于MXFP4格式实现了11.40%（6.90%）的准确率提升，且在量化全路径矩阵乘法时，与全精度相比仅低5.89%（3.31%）。我们的工作着重于如何表示而不是如何缩放，这为能效高效的LLM推理提供了一条有前景的道路。 

---
# TED: Turn Emphasis with Dialogue Feature Attention for Emotion Recognition in Conversation 

**Title (ZH)**: TED：利用对话特征注意力突出强调在对话情感识别中的应用 

**Authors**: Junya Ono, Hiromi Wakaki  

**Link**: [PDF](https://arxiv.org/pdf/2501.01123)  

**Abstract**: Emotion recognition in conversation (ERC) has been attracting attention by methods for modeling multi-turn contexts. The multi-turn input to a pretraining model implicitly assumes that the current turn and other turns are distinguished during the training process by inserting special tokens into the input sequence. This paper proposes a priority-based attention method to distinguish each turn explicitly by adding dialogue features into the attention mechanism, called Turn Emphasis with Dialogue (TED). It has a priority for each turn according to turn position and speaker information as dialogue features. It takes multi-head self-attention between turn-based vectors for multi-turn input and adjusts attention scores with the dialogue features. We evaluate TED on four typical benchmarks. The experimental results demonstrate that TED has high overall performance in all datasets and achieves state-of-the-art performance on IEMOCAP with numerous turns. 

**Abstract (ZH)**: 对话中的情绪识别（Emotion Recognition in Conversation, ERC）已成为通过建模多轮上下文的方法引起关注的研究领域。多轮输入到预训练模型中隐含地假设，在训练过程中通过在输入序列中插入特殊标记来区分当前轮次和其他轮次。本文提出了一种基于优先级的注意力方法，通过向注意力机制中添加对话特征来明确地区分每一轮次，这种方法称为对话重点（Turn Emphasis with Dialogue，TED）。每一轮次根据其位置和发言者信息具有优先级。该方法在基于轮次的向量之间采用多头自注意力机制处理多轮输入，并利用对话特征调整注意力分数。我们在四个典型的基准数据集上评估了TED。实验结果表明，TED在所有数据集上总体性能优异，并且在IEMOCAP数据集上（尤其是涉及大量轮次的场景）达到了最先进的性能。 

---
# BeliN: A Novel Corpus for Bengali Religious News Headline Generation using Contextual Feature Fusion 

**Title (ZH)**: BeliN：一种基于上下文特征融合的孟加拉语宗教新闻标题生成新型语料库 

**Authors**: Md Osama, Ashim Dey, Kawsar Ahmed, Muhammad Ashad Kabir  

**Link**: [PDF](https://arxiv.org/pdf/2501.01069)  

**Abstract**: Automatic text summarization, particularly headline generation, remains a critical yet underexplored area for Bengali religious news. Existing approaches to headline generation typically rely solely on the article content, overlooking crucial contextual features such as sentiment, category, and aspect. This limitation significantly hinders their effectiveness and overall performance. This study addresses this limitation by introducing a novel corpus, BeliN (Bengali Religious News) - comprising religious news articles from prominent Bangladeshi online newspapers, and MultiGen - a contextual multi-input feature fusion headline generation approach. Leveraging transformer-based pre-trained language models such as BanglaT5, mBART, mT5, and mT0, MultiGen integrates additional contextual features - including category, aspect, and sentiment - with the news content. This fusion enables the model to capture critical contextual information often overlooked by traditional methods. Experimental results demonstrate the superiority of MultiGen over the baseline approach that uses only news content, achieving a BLEU score of 18.61 and ROUGE-L score of 24.19, compared to baseline approach scores of 16.08 and 23.08, respectively. These findings underscore the importance of incorporating contextual features in headline generation for low-resource languages. By bridging linguistic and cultural gaps, this research advances natural language processing for Bengali and other underrepresented languages. To promote reproducibility and further exploration, the dataset and implementation code are publicly accessible at this https URL. 

**Abstract (ZH)**: 自动文本摘要，特别是标题生成，在孟加拉语宗教新闻领域仍是一个关键但尚未充分探索的领域。现有的标题生成方法通常仅依赖文章内容，而忽视了诸如情感、类别和方面等重要的上下文特征。这一局限性极大地阻碍了其效果和整体性能。本研究通过引入一个新的语料库——BeliN（孟加拉语宗教新闻），以及一个多输入上下文特征融合的标题生成方法——MultiGen，来解决这一局限性。BeliN 包含来自主流孟加拉线上报纸的宗教新闻文章。MultiGen 利用基于变换器的预训练语言模型（如 BanglaT5、mBART、mT5 和 mT0），将类别、方面和情感等额外的上下文特征与新闻内容整合在一起。这种融合使得模型能够捕捉到传统方法经常忽略的关键上下文信息。实验结果表明，相较于只使用新闻内容的基线方法，MultiGen 在 BLEU 分数上提高了 18.61，在 ROUGE-L 分数上提高了 24.19（基线方法的相应分数为 16.08 和 23.08）。这些发现强调了在低资源语言的标题生成中融入上下文特征的重要性。通过弥合语言和文化障碍，本研究促进了孟加拉语和其他未充分代表语言的自然语言处理的发展。为促进可重复性和进一步探索，该数据集和实现代码已在以下链接公开：[请在此处提供链接]。 

---
# Dynamic Attention-Guided Context Decoding for Mitigating Context Faithfulness Hallucinations in Large Language Models 

**Title (ZH)**: 面向大型语言模型减轻上下文忠实性幻觉的动态注意力导向上下文解码 

**Authors**: Yanwen Huang, Yong Zhang, Ning Cheng, Zhitao Li, Shaojun Wang, Jing Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2501.01059)  

**Abstract**: Large language models (LLMs) often suffer from context faithfulness hallucinations, where outputs deviate from retrieved information due to insufficient context utilization and high output uncertainty. Our uncertainty evaluation experiments reveal a strong correlation between high uncertainty and hallucinations. We hypothesize that attention mechanisms encode signals indicative of contextual utilization, validated through probing analysis. Based on these insights, we propose Dynamic Attention-Guided Context Decoding (DAGCD), a lightweight framework that integrates attention distributions and uncertainty signals in a single-pass decoding process. Experiments across QA datasets demonstrate DAGCD's effectiveness, achieving significant improvements in faithfulness and robustness while maintaining computational efficiency. 

**Abstract (ZH)**: 大型语言模型（LLMs）往往会遭受语境忠实性幻觉的问题，即输出与检索到的信息相偏离，这可能是由于上下文利用不足和高输出不确定性所致。我们的不确定性评估实验揭示了高不确定性与幻觉之间存在强烈的相关性。我们假设注意力机制编码了与上下文利用相关的信号，并通过探查分析进行了验证。基于这些见解，我们提出了动态注意力引导的上下文解码（DAGCD），这是一个轻量级框架，将注意力分布和不确定性信号整合到单次解码过程中。通过跨多种问答数据集的实验，证明了DAGCD的有效性，不仅在忠实性和鲁棒性方面取得了显著改进，同时保持了计算效率。 

---
# Risks of Cultural Erasure in Large Language Models 

**Title (ZH)**: 大型语言模型中的文化抹除风险 

**Authors**: Rida Qadri, Aida M. Davani, Kevin Robinson, Vinodkumar Prabhakaran  

**Link**: [PDF](https://arxiv.org/pdf/2501.01056)  

**Abstract**: Large language models are increasingly being integrated into applications that shape the production and discovery of societal knowledge such as search, online education, and travel planning. As a result, language models will shape how people learn about, perceive and interact with global cultures making it important to consider whose knowledge systems and perspectives are represented in models. Recognizing this importance, increasingly work in Machine Learning and NLP has focused on evaluating gaps in global cultural representational distribution within outputs. However, more work is needed on developing benchmarks for cross-cultural impacts of language models that stem from a nuanced sociologically-aware conceptualization of cultural impact or harm. We join this line of work arguing for the need of metricizable evaluations of language technologies that interrogate and account for historical power inequities and differential impacts of representation on global cultures, particularly for cultures already under-represented in the digital corpora. We look at two concepts of erasure: omission: where cultures are not represented at all and simplification i.e. when cultural complexity is erased by presenting one-dimensional views of a rich culture. The former focuses on whether something is represented, and the latter on how it is represented. We focus our analysis on two task contexts with the potential to influence global cultural production. First, we probe representations that a language model produces about different places around the world when asked to describe these contexts. Second, we analyze the cultures represented in the travel recommendations produced by a set of language model applications. Our study shows ways in which the NLP community and application developers can begin to operationalize complex socio-cultural considerations into standard evaluations and benchmarks. 

**Abstract (ZH)**: 大型语言模型越来越多地被整合到搜索引擎、在线教育和旅行规划等应用中，从而影响社会知识的生产和发现。因此，语言模型将塑造人们了解、认知和与全球文化互动的方式，这使得考虑模型中所体现的知识系统和视角变得尤为重要。认识到这一点，机器学习和自然语言处理领域的研究越来越多地致力于评估模型输出中的全球文化代表性分布差距。然而，还需要在基于细致社会意识的概念化文化影响或危害的基础上，发展衡量语言模型跨文化影响的基准。我们加入这一研究行列，强调需要可量化的评估语言技术的指标，这些指标能够探究和考虑历史权力不平等及代表性对全球文化的不同影响，尤其是对于已经在数字化语料库中严重不足的文化。

我们探讨了两种擦除概念：一是省略，指某一文化完全没有被代表；二是简化，指在呈现丰富文化时呈现实单维度的观点，从而消除了文化复杂性。前者关注的是某事物是否被代表，后者关注的是其如何被代表。我们在两个可能影响全球文化生产的任务背景中进行了分析。首先，我们探究了当语言模型被要求描述世界各地的不同场景时，其产生的文化代表情况。其次，我们分析了一组语言模型应用中提供的旅行推荐所体现的文化。我们的研究展示了NLP社区和应用开发者如何开始将复杂的社会文化考量纳入标准评估和基准中的方法。 

---
# Dynamic Scaling of Unit Tests for Code Reward Modeling 

**Title (ZH)**: 代码奖励模型中单元测试的动态缩放 

**Authors**: Zeyao Ma, Xiaokang Zhang, Jing Zhang, Jifan Yu, Sijia Luo, Jie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2501.01054)  

**Abstract**: Current large language models (LLMs) often struggle to produce accurate responses on the first attempt for complex reasoning tasks like code generation. Prior research tackles this challenge by generating multiple candidate solutions and validating them with LLM-generated unit tests. The execution results of unit tests serve as reward signals to identify correct solutions. As LLMs always confidently make mistakes, these unit tests are not reliable, thereby diminishing the quality of reward signals. Motivated by the observation that scaling the number of solutions improves LLM performance, we explore the impact of scaling unit tests to enhance reward signal quality. Our pioneer experiment reveals a positive correlation between the number of unit tests and reward signal quality, with greater benefits observed in more challenging problems. Based on these insights, we propose CodeRM-8B, a lightweight yet effective unit test generator that enables efficient and high-quality unit test scaling. Additionally, we implement a dynamic scaling mechanism that adapts the number of unit tests based on problem difficulty, further improving efficiency. Experimental results show that our approach significantly improves performance across various models on three benchmarks (e.g., with gains of 18.43% for Llama3-8B and 3.42% for GPT-4o-mini on HumanEval Plus). 

**Abstract (ZH)**: 当前的大语言模型（LLMs）在处理复杂的推理任务（如代码生成）时，往往难以在第一次尝试中生成准确的响应。之前的研究所通过生成多个候选解决方案，并使用LLM生成的单元测试进行验证来应对这一挑战。单元测试的执行结果作为奖励信号，用于识别正确的解。然而，由于LLMs总是自信地出错，这些单元测试不够可靠，从而降低了奖励信号的质量。受观察到的数量扩展解决方案能提高LLM性能的启发，我们探索了扩展单元测试数量以提高奖励信号质量的影响。我们的先驱实验揭示了单元测试数量与奖励信号质量之间存在正相关关系，在更具挑战性的问题上观察到了更大的效益。基于这些见解，我们提出了一种轻量级但有效的单元测试生成器CodeRM-8B，该生成器能够实现高效的单元测试数量扩展。此外，我们实现了一种动态的扩展机制，根据问题难度自动调整单元测试的数量，进一步提高了效率。实验结果表明，我们的方法在三个基准测试中显著提高了各种模型的性能（例如，在HumanEval Plus基准测试中，代码量减少了18.43%的Llama3-8B和3.42%的GPT-4o-mini的性能）。 

---
# FED: Fast and Efficient Dataset Deduplication Framework with GPU Acceleration 

**Title (ZH)**: FED：基于GPU加速的快速高效数据集去重框架 

**Authors**: Youngjun Son, Chaewon Kim, Jaejin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2501.01046)  

**Abstract**: Dataset deduplication plays a crucial role in enhancing data quality, ultimately improving training performance and efficiency of LLMs. A commonly used method for data deduplication is the MinHash LSH algorithm. Recently, NVIDIA introduced a GPU-based MinHash LSH deduplication method, but it remains suboptimal, leaving room for further improvement in processing efficiency. This paper proposes a GPU-accelerated deduplication framework \sys that optimizes MinHash LSH for GPU clusters and leverages computationally efficient and partially reusable non-cryptographic hash functions. \sys significantly outperforms the CPU-based deduplication tool included in SlimPajama by up to 58.3 times and the GPU-based deduplication tool included in NVIDIA NeMo Curator by up to 8.6 times when processing 1 million documents with a node of four GPUs. Deduplication of 1.2 trillion tokens is completed in just 5.1 hours in a four-node, 16-GPU environment. The related code is publicly available on GitHub (this https URL). 

**Abstract (ZH)**: 数据集去重在提高数据质量、最终提升大规模语言模型（LLM）训练性能和效率方面发挥着重要作用。一种常用的数据去重方法是MinHash LSH算法。最近，NVIDIA引入了一种基于GPU的MinHash LSH去重方法，但其处理效率仍存在改进空间。本文提出了一种名为\sys的GPU加速去重框架，该框架优化了适用于GPU集群的MinHash LSH，并利用了计算效率高且部分可重用的非密码学哈希函数。在处理100万份文档时，与SlimPajama自带的基于CPU的去重工具相比，\sys的性能提高了最多58.3倍；与NVIDIA NeMo Curator自带的基于GPU的去重工具相比，性能提高了最多8.6倍。在四节点、16个GPU的环境中，1.2万亿个令牌的去重仅需5.1小时即可完成。相关代码已在GitHub上公开发布（请点击此链接：[GitHub链接]）。 

---
# MSWA: Refining Local Attention with Multi-ScaleWindow Attention 

**Title (ZH)**: MSWA：基于多尺度窗口注意力的局部注意力精炼 

**Authors**: Yixing Xu, Shivank Nag, Dong Li, Lu Tian, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2501.01039)  

**Abstract**: Transformer-based LLMs have achieved exceptional performance across a wide range of NLP tasks. However, the standard self-attention mechanism suffers from quadratic time complexity and linearly increased cache size. Sliding window attention (SWA) solves this problem by restricting the attention range to a fixed-size local context window. Nevertheless, SWA employs a uniform window size for each head in each layer, making it inefficient in capturing context of varying scales. To mitigate this limitation, we propose Multi-Scale Window Attention (MSWA) which applies diverse window sizes across heads and layers in the Transformer. It not only allows for different window sizes among heads within the same layer but also progressively increases window size allocation from shallow to deep layers, thus enabling the model to capture contextual information with different lengths and distances. Experimental results on language modeling and common-sense reasoning tasks substantiate that MSWA outperforms traditional local attention in both effectiveness and efficiency. 

**Abstract (ZH)**: 基于Transformer的大型语言模型（LLMs）已经在广泛的自然语言处理（NLP）任务中取得了卓越的性能。然而，标准的自注意力机制具有二次时间复杂度和线性增加的缓存大小。滑动窗口注意力（SWA）通过限制注意力范围到固定大小的局部上下文窗口来解决这一问题。尽管如此，SWA 在每一层中的每个头都采用了统一的窗口大小，这使得其在捕捉不同尺度的上下文时不够高效。为缓解这一局限，我们提出了一种多尺度窗口注意力（MSWA），它在Transformer中的各个头和层应用了不同的窗口大小。它不仅允许在同一层内的各个头之间采用不同的窗口大小，而且还从浅层逐渐增加到深层的窗口大小分配，从而使得模型能够捕获不同长度和距离的上下文信息。在语言建模和常识推理任务上的实验结果证明，MSWA 在有效性与效率方面均优于传统的局部注意力机制。 

---
# Advancing Singlish Understanding: Bridging the Gap with Datasets and Multimodal Models 

**Title (ZH)**: 提升新加坡英语理解：借助数据集和多模态模型缩小差距 

**Authors**: Bin Wang, Xunlong Zou, Shuo Sun, Wenyu Zhang, Yingxu He, Zhuohan Liu, Chengwei Wei, Nancy F. Chen, AiTi Aw  

**Link**: [PDF](https://arxiv.org/pdf/2501.01034)  

**Abstract**: Singlish, a Creole language rooted in English, is a key focus in linguistic research within multilingual and multicultural contexts. However, its spoken form remains underexplored, limiting insights into its linguistic structure and applications. To address this gap, we standardize and annotate the largest spoken Singlish corpus, introducing the Multitask National Speech Corpus (MNSC). These datasets support diverse tasks, including Automatic Speech Recognition (ASR), Spoken Question Answering (SQA), Spoken Dialogue Summarization (SDS), and Paralinguistic Question Answering (PQA). We release standardized splits and a human-verified test set to facilitate further research. Additionally, we propose SingAudioLLM, a multi-task multimodal model leveraging multimodal large language models to handle these tasks concurrently. Experiments reveal our models adaptability to Singlish context, achieving state-of-the-art performance and outperforming prior models by 10-30% in comparison with other AudioLLMs and cascaded solutions. 

**Abstract (ZH)**: 新加坡话，这一源于英语的克里奥耳语言，在多语言和多文化背景下是语言研究的一个重要焦点。然而，其口语形式仍然没有得到充分探索，限制了对其语言结构和应用的理解。为弥补这一空白，我们标准化并标注了最大的新加坡话语料库，引入了多任务国家语音语料库（MNSC）。这些数据集支持多种任务，包括自动语音识别（ASR）、口语问答（SQA）、口语对话摘要（SDS）和副语言问答（PQA）。我们发布标准化的划分和经人工验证的测试集，以促进进一步的研究。此外，我们提出了SingAudioLLM，这是一种利用多模态大型语言模型的多任务多模态模型，能够同时处理这些任务。实验结果显示，我们的模型在新加坡话语境中的适应性更佳，实现了最先进的性能，并在与其他音频LLM和级联解决方案的对比中提升了10%到30%。 

---
# ValuesRAG: Enhancing Cultural Alignment Through Retrieval-Augmented Contextual Learning 

**Title (ZH)**: ValuesRAG：通过检索增强上下文学习提高文化一致性 

**Authors**: Wonduk Seo, Zonghao Yuan, Yi Bu  

**Link**: [PDF](https://arxiv.org/pdf/2501.01031)  

**Abstract**: Cultural values alignment in Large Language Models (LLMs) is a critical challenge due to their tendency to embed Western-centric biases from training data, leading to misrepresentations and fairness issues in cross-cultural contexts. Recent approaches, such as role-assignment and few-shot learning, often struggle with reliable cultural alignment as they heavily rely on pre-trained knowledge, lack scalability, and fail to capture nuanced cultural values effectively. To address these issues, we propose ValuesRAG, a novel and effective framework that applies Retrieval-Augmented Generation (RAG) with in-context learning to integrate cultural and demographic knowledge dynamically during text generation. Leveraging the World Values Survey (WVS) dataset, ValuesRAG first generates summaries of values for each individual. Subsequently, we curated several representative regional datasets to serve as test datasets and retrieve relevant summaries of values based on demographic features, followed by a reranking step to select the top-k relevant summaries. ValuesRAG consistently outperforms baseline methods, both in the main experiment and in the ablation study where only the values summary was provided, highlighting ValuesRAG's potential to foster culturally aligned AI systems and enhance the inclusivity of AI-driven applications. 

**Abstract (ZH)**: 大型语言模型（LLMs）中的文化价值观对齐是一个关键挑战，因为它们倾向于从训练数据中嵌入以西方为中心的偏见，导致跨文化背景下出现误表征和公正性问题。近年来，诸如角色分配和少样本学习等方法在可靠的文化对齐方面往往难以应对，因为这些方法高度依赖预训练知识，缺乏可扩展性，并且难以有效捕捉文化价值观的细微差别。为了解决这些问题，我们提出了一种名为ValuesRAG的新颖而有效的框架，该框架采用检索增强生成（RAG）结合上下文学习，在文本生成过程中动态整合文化与人口统计数据知识。利用世界价值观调查（WVS）数据集，ValuesRAG 首先为每位个体生成价值观摘要。随后，我们精选了几个具有代表性的区域数据集作为测试数据集，并根据人口统计数据特征检索相关的价值观摘要，再通过重新排序步骤选择最相关的前k个摘要。在主要实验和仅提供价值观摘要的消融研究中，ValuesRAG 一致性地优于基线方法，这强调了ValuesRAG 在促进文化对齐的人工智能系统以及增强人工智能驱动应用的包容性方面的潜力。 

---
# Reasoning based on symbolic and parametric knowledge bases: a survey 

**Title (ZH)**: 基于符号知识库和参数知识库的推理：一种综述 

**Authors**: Mayi Xu, Yunfeng Ning, Yongqi Li, Jianhao Chen, Jintao Wen, Yao Xiao, Shen Zhou, Birong Pan, Zepeng Bao, Xin Miao, Hankun Kang, Ke Sun, Tieyun Qian  

**Link**: [PDF](https://arxiv.org/pdf/2501.01030)  

**Abstract**: Reasoning is fundamental to human intelligence, and critical for problem-solving, decision-making, and critical thinking. Reasoning refers to drawing new conclusions based on existing knowledge, which can support various applications like clinical diagnosis, basic education, and financial analysis. Though a good number of surveys have been proposed for reviewing reasoning-related methods, none of them has systematically investigated these methods from the viewpoint of their dependent knowledge base. Both the scenarios to which the knowledge bases are applied and their storage formats are significantly different. Hence, investigating reasoning methods from the knowledge base perspective helps us better understand the challenges and future directions. To fill this gap, this paper first classifies the knowledge base into symbolic and parametric ones. The former explicitly stores information in human-readable symbols, and the latter implicitly encodes knowledge within parameters. Then, we provide a comprehensive overview of reasoning methods using symbolic knowledge bases, parametric knowledge bases, and both of them. Finally, we identify the future direction toward enhancing reasoning capabilities to bridge the gap between human and machine intelligence. 

**Abstract (ZH)**: 推理是人类智能的基础，对于问题解决、决策制定和批判性思维至关重要。推理是指基于现有知识得出新结论的过程，可应用于临床诊断、基础教育和金融分析等多种应用领域。尽管已经提出了许多综述来回顾与推理相关的研究方法，但没有任何一项综述系统地从知识库依赖性的视角来研究这些方法。由于知识库所应用于的场景及其存储格式存在显著差异，因此从知识库视角来研究推理方法有助于我们更好地了解现有挑战和未来方向。为填补这一空白，本文首先将知识库分类为符号性和参数性知识库。符号性知识库明确地以人类可读的符号形式存储信息，而参数性知识库则隐式地将知识编码在参数中。然后，本文提供了对符号性知识库、参数性知识库及其两者结合使用的推理方法的全面综述。最后，本文指出了提升推理能力的未来方向，以缩小人类智能与机器智能之间的差距。 

---
# KaLM-Embedding: Superior Training Data Brings A Stronger Embedding Model 

**Title (ZH)**: KaLM-嵌入：更优质的训练数据带来更强的嵌入模型 

**Authors**: Xinshuo Hu, Zifei Shan, Xinping Zhao, Zetian Sun, Zhenyu Liu, Dongfang Li, Shaolin Ye, Xinyuan Wei, Qian Chen, Baotian Hu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.01028)  

**Abstract**: As retrieval-augmented generation prevails in large language models, embedding models are becoming increasingly crucial. Despite the growing number of general embedding models, prior work often overlooks the critical role of training data quality. In this work, we introduce KaLM-Embedding, a general multilingual embedding model that leverages a large quantity of cleaner, more diverse, and domain-specific training data. Our model has been trained with key techniques proven to enhance performance: (1) persona-based synthetic data to create diversified examples distilled from LLMs, (2) ranking consistency filtering to remove less informative samples, and (3) semi-homogeneous task batch sampling to improve training efficacy. Departing from traditional BERT-like architectures, we adopt Qwen2-0.5B as the pre-trained model, facilitating the adaptation of auto-regressive language models for general embedding tasks. Extensive evaluations of the MTEB benchmark across multiple languages show that our model outperforms others of comparable size, setting a new standard for multilingual embedding models with <1B parameters. 

**Abstract (ZH)**: 随着检索增强生成在大规模语言模型中的应用越来越广泛，嵌入模型变得愈发关键。尽管存在众多通用嵌入模型，先前的工作往往忽视了培训数据质量的关键作用。在本文中，我们介绍了KaLM-Embedding，这是一种利用大量更清洁、更具多样性和领域特定训练数据的通用多语言嵌入模型。我们的模型采用了已被证明能够提升性能的一些关键技术：（1）基于人设的合成数据来创建多样化的例子，这些例子是从大语言模型（LLM）中提炼出来的，（2）排名一致性过滤以去除不具信息性的样本，（3）半同质任务批处理抽样以提高训练效率。不同于传统的BERT架构，我们采用了Qwen2-0.5B作为预训练模型，这有助于自回归语言模型适应通用嵌入任务。通过MTEB基准在多种语言上的广泛评估表明，我们的模型在性能上优于其他同类大小的模型，并且在参数数量少于1亿的情况下，为我们树立了新的多语言嵌入模型标准。 

---
# MDSF: Context-Aware Multi-Dimensional Data Storytelling Framework based on Large language Model 

**Title (ZH)**: MDSF：基于大规模语言模型的认知上下文多维数据叙事框架 

**Authors**: Chengze Zhang, Changshan Li, Shiyang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2501.01014)  

**Abstract**: The exponential growth of data and advancements in big data technologies have created a demand for more efficient and automated approaches to data analysis and storytelling. However, automated data analysis systems still face challenges in leveraging large language models (LLMs) for data insight discovery, augmented analysis, and data storytelling. This paper introduces the Multidimensional Data Storytelling Framework (MDSF) based on large language models for automated insight generation and context-aware storytelling. The framework incorporates advanced preprocessing techniques, augmented analysis algorithms, and a unique scoring mechanism to identify and prioritize actionable insights. The use of fine-tuned LLMs enhances contextual understanding and generates narratives with minimal manual intervention. The architecture also includes an agent-based mechanism for real-time storytelling continuation control. Key findings reveal that MDSF outperforms existing methods across various datasets in terms of insight ranking accuracy, descriptive quality, and narrative coherence. The experimental evaluation demonstrates MDSF's ability to automate complex analytical tasks, reduce interpretive biases, and improve user satisfaction. User studies further underscore its practical utility in enhancing content structure, conclusion extraction, and richness of detail. 

**Abstract (ZH)**: 数据的指数级增长和大数据技术的进步为更高效的自动化数据分析和故事叙述方法提出了需求。然而，现有的自动化数据分析系统在利用大规模语言模型（LLMs）进行数据洞察发现、增强分析和数据故事叙述方面仍然面临挑战。本文介绍了基于大规模语言模型的多维数据故事叙述框架（MDSF），该框架旨在实现自动化洞察生成和情境感知故事叙述。该框架整合了高级预处理技术、增强分析算法以及一种独特的评分机制，用于识别和优先级排序可操作的洞察。微调后的LLMs通过增强上下文理解，在最少人工干预的情况下生成叙述。该架构还包括了基于代理的机制，以实现实时故事叙述延续控制。实验研究结果表明，MDSF在各类数据集上均在洞察排名准确性、描述质量和叙述连贯性方面优于现有方法。实验证明，MDSF能够自动化复杂分析任务、减少解释性偏见并提高用户体验。进一步的用户研究还强调了其在增强内容结构、结论提取和细节丰富性方面的实际应用价值。 

---
# Exploring Information Processing in Large Language Models: Insights from Information Bottleneck Theory 

**Title (ZH)**: 探索大型语言模型中的信息处理：信息瓶颈理论的洞见 

**Authors**: Zhou Yang, Zhengyu Qi, Zhaochun Ren, Zhikai Jia, Haizhou Sun, Xiaofei Zhu, Xiangwen Liao  

**Link**: [PDF](https://arxiv.org/pdf/2501.00999)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of tasks by understanding input information and predicting corresponding outputs. However, the internal mechanisms by which LLMs comprehend input and make effective predictions remain poorly understood. In this paper, we explore the working mechanism of LLMs in information processing from the perspective of Information Bottleneck Theory. We propose a non-training construction strategy to define a task space and identify the following key findings: (1) LLMs compress input information into specific task spaces (e.g., sentiment space, topic space) to facilitate task understanding; (2) they then extract and utilize relevant information from the task space at critical moments to generate accurate predictions. Based on these insights, we introduce two novel approaches: an Information Compression-based Context Learning (IC-ICL) and a Task-Space-guided Fine-Tuning (TS-FT). IC-ICL enhances reasoning performance and inference efficiency by compressing retrieved example information into the task space. TS-FT employs a space-guided loss to fine-tune LLMs, encouraging the learning of more effective compression and selection mechanisms. Experiments across multiple datasets validate the effectiveness of task space construction. Additionally, IC-ICL not only improves performance but also accelerates inference speed by over 40\%, while TS-FT achieves superior results with a minimal strategy adjustment. 

**Abstract (ZH)**: 大型语言模型（LLMs）已在广泛的任务中展示了卓越的性能，通过理解输入信息并预测相应的输出。然而，LLMs 在处理输入信息和作出有效预测时的具体工作机制仍然不甚明了。本文从信息瓶颈理论的角度探讨了LLMs在信息处理过程中的工作机制。我们提出了一种非训练的构建策略来定义任务空间，并发现以下关键发现：（1）LLMs 将输入信息压缩到特定的任务空间（例如情感空间、主题空间），以利于任务理解；（2）然后在关键时刻从任务空间中提取和利用相关信息来生成准确的预测。基于这些见解，我们引入了两种新的方法：信息压缩为基础的上下文学习（IC-ICL）和任务空间导向的微调（TS-FT）。IC-ICL 通过将检索到的示例信息压缩到任务空间中，增强推理性能和推理效率。TS-FT 利用空间导向的损失对LLMs 进行微调，促使模型学习更有效的压缩和选择机制。通过对多个数据集的实验验证了任务空间构建的有效性。此外，IC-ICL 不仅提高了性能，还通过超过40%的加速提升了推理速度，而TS-FT 在几乎没有策略调整的情况下实现了更优的结果。 

---
# Are LLMs effective psychological assessors? Leveraging adaptive RAG for interpretable mental health screening through psychometric practice 

**Title (ZH)**: 大型语言模型在心理评估方面是否有效？通过心理测量实践利用自适应 Retrieval-Augmented Generation 进行可解释的心理健康筛查 

**Authors**: Federico Ravenda, Seyed Ali Bahrainian, Andrea Raballo, Antonietta Mira, Noriko Kando  

**Link**: [PDF](https://arxiv.org/pdf/2501.00982)  

**Abstract**: In psychological practice, standardized questionnaires serve as essential tools for assessing mental constructs (e.g., attitudes, traits, and emotions) through structured questions (aka items). With the increasing prevalence of social media platforms where users share personal experiences and emotions, researchers are exploring computational methods to leverage this data for rapid mental health screening. In this study, we propose a novel adaptive Retrieval-Augmented Generation (RAG) approach that completes psychological questionnaires by analyzing social media posts. Our method retrieves the most relevant user posts for each question in a psychological survey and uses Large Language Models (LLMs) to predict questionnaire scores in a zero-shot setting. Our findings are twofold. First we demonstrate that this approach can effectively predict users' responses to psychological questionnaires, such as the Beck Depression Inventory II (BDI-II), achieving performance comparable to or surpassing state-of-the-art models on Reddit-based benchmark datasets without relying on training data. Second, we show how this methodology can be generalized as a scalable screening tool, as the final assessment is systematically derived by completing standardized questionnaires and tracking how individual item responses contribute to the diagnosis, aligning with established psychometric practices. 

**Abstract (ZH)**: 在心理实践中，标准化量表通过结构化问题（即项目）来评估心理构念（如态度、特质和情绪），发挥着重要作用。随着社交媒体平台的普及，用户分享个人经历和情绪，研究人员正探索利用这些数据进行快速心理健康筛查的计算方法。本研究提出了一种新颖的自适应检索增强生成（RAG）方法，通过分析社交媒体帖子来完成心理健康量表。我们的方法为精神健康调查中的每个问题检索最相关的用户帖子，并使用大规模语言模型（LLMs）在零样本设置中预测量表评分。我们的发现分为两个方面。首先，我们展示了该方法能够有效预测用户对心理健康量表的反应，例如贝克抑郁量表第二版（BDI-II），并且在基于Reddit的数据集上达到或超过最先进的模型的性能，而无需依赖训练数据。其次，我们展示了这种方法可以作为可扩展筛查工具进行泛化，最终评估是通过完成标准化量表并系统地跟踪每个项目响应对诊断的贡献而得出的，与现有的心理测量学实践相一致。 

---
# Incremental Dialogue Management: Survey, Discussion, and Implications for HRI 

**Title (ZH)**: 增量对话管理：综述、讨论及其对人机交互的影响 

**Authors**: Casey Kennington, Pierre Lison, David Schlangen  

**Link**: [PDF](https://arxiv.org/pdf/2501.00953)  

**Abstract**: Efforts towards endowing robots with the ability to speak have benefited from recent advancements in NLP, in particular large language models. However, as powerful as current models have become, they still operate on sentence or multi-sentence level input, not on the word-by-word input that humans operate on, affecting the degree of responsiveness that they offer, which is critical in situations where humans interact with robots using speech. In this paper, we review the literature on interactive systems that operate incrementally (i.e., at the word level or below it). We motivate the need for incremental systems, survey incremental modeling of important aspects of dialogue like speech recognition and language generation. Primary focus is on the part of the system that makes decisions, known as the dialogue manager. We find that there is very little research on incremental dialogue management, offer some requirements for practical incremental dialogue management, and the implications of incremental dialogue for embodied, robotic platforms. 

**Abstract (ZH)**: 赋予机器人说话能力的努力得益于近年来自然语言处理（NLP）特别是大规模语言模型的进展。然而，尽管当前模型变得非常强大，它们仍然基于句子或句子组的输入，而不是基于人类所依赖的逐词输入，这影响了它们的响应速度，尤其是在人类使用语音与机器人进行交互的情况下，响应速度至关重要。本文回顾了逐步增量工作的交互系统文献（即，在句子或其以下级别）。我们论证了逐步增量系统的必要性，调查了诸如语音识别和语言生成等对话关键方面的重要方面的逐步增量模型。本文的主要焦点在于做出决策的部分，即对话管理器。我们发现关于逐步增量对话管理的研究非常少，提出了逐步增量对话管理的一些要求，并探讨了逐步增量对话对实体机器人平台的影响。 

---
# Unfolding the Headline: Iterative Self-Questioning for News Retrieval and Timeline Summarization 

**Title (ZH)**: 展开标题：迭代自我提问在新闻检索与时间轴总结中的应用 

**Authors**: Weiqi Wu, Shen Huang, Yong Jiang, Pengjun Xie, Fei Huang, Hai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2501.00888)  

**Abstract**: In the fast-changing realm of information, the capacity to construct coherent timelines from extensive event-related content has become increasingly significant and challenging. The complexity arises in aggregating related documents to build a meaningful event graph around a central topic. This paper proposes CHRONOS - Causal Headline Retrieval for Open-domain News Timeline SummarizatiOn via Iterative Self-Questioning, which offers a fresh perspective on the integration of Large Language Models (LLMs) to tackle the task of Timeline Summarization (TLS). By iteratively reflecting on how events are linked and posing new questions regarding a specific news topic to gather information online or from an offline knowledge base, LLMs produce and refresh chronological summaries based on documents retrieved in each round. Furthermore, we curate Open-TLS, a novel dataset of timelines on recent news topics authored by professional journalists to evaluate open-domain TLS where information overload makes it impossible to find comprehensive relevant documents from the web. Our experiments indicate that CHRONOS is not only adept at open-domain timeline summarization, but it also rivals the performance of existing state-of-the-art systems designed for closed-domain applications, where a related news corpus is provided for summarization. 

**Abstract (ZH)**: 在信息快速变化的领域中，从大量事件相关信息中构建连贯的时间线的能力变得越来越重要且具有挑战性。复杂性在于将相关文档聚集起来，围绕一个中心主题构建有意义的事件图。本文提出了CHRONOS——面向开放领域的新闻时间线总结的因果标题检索方法，通过迭代自我提问以提供一种将大语言模型（LLMs）整合到时间线总结任务（TLS）中的新视角。通过迭代反思事件之间的联系并针对特定新闻主题提出新的问题，LLMs 可以基于每一轮检索到的文档生成和更新时间线总结。此外，我们还编纂了Open-TLS数据集，这是一个由专业记者编写的、包含当前新闻主题时间线的新型数据集，用于评估开放领域的时间线总结任务，其中信息过载使得从网络上找到全面的相关文档变得不可能。我们的实验表明，CHRONOS不仅适用于开放领域的专题总结，其性能也与为封闭领域应用设计的现有最先进系统相媲美，其中提供了一个相关新闻语料库用于总结。 

---
# Representation in large language models 

**Title (ZH)**: 大型语言模型中的表示 

**Authors**: Cameron C. Yetman  

**Link**: [PDF](https://arxiv.org/pdf/2501.00885)  

**Abstract**: The extraordinary success of recent Large Language Models (LLMs) on a diverse array of tasks has led to an explosion of scientific and philosophical theorizing aimed at explaining how they do what they do. Unfortunately, disagreement over fundamental theoretical issues has led to stalemate, with entrenched camps of LLM optimists and pessimists often committed to very different views of how these systems work. Overcoming stalemate requires agreement on fundamental questions, and the goal of this paper is to address one such question, namely: is LLM behavior driven partly by representation-based information processing of the sort implicated in biological cognition, or is it driven entirely by processes of memorization and stochastic table look-up? This is a question about what kind of algorithm LLMs implement, and the answer carries serious implications for higher level questions about whether these systems have beliefs, intentions, concepts, knowledge, and understanding. I argue that LLM behavior is partially driven by representation-based information processing, and then I describe and defend a series of practical techniques for investigating these representations and developing explanations on their basis. The resulting account provides a groundwork for future theorizing about language models and their successors. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）在广泛任务上的非凡成功导致了科学和哲学理论的爆炸式发展，旨在解释它们是如何完成这些任务的。不幸的是，对基本理论问题的不同看法已经导致了僵局，形成了一种LLM乐观派和悲观派的固有阵营，双方往往持有截然不同的系统工作方式观点。克服僵局需要在基本问题上达成共识，本文的目标是解决这样一个问题：即LLM的行为是部分由与生物认知相关的表示性信息处理驱动的，还是完全由记忆和随机查找表的过程驱动的？这个问题关乎LLMs所实现的算法类型，而对这一问题的回答对更高层次的问题，如这些系统是否具有信念、意图、概念、知识和理解持有重要意义。我认为，LLM的行为部分是由表示性信息处理驱动的，然后我描述并辩护了一系列实际技术，用于研究这些表示并基于此发展解释。所得的解释为未来关于语言模型及其后继者的理论提供了基础。 

---
# TrustRAG: Enhancing Robustness and Trustworthiness in RAG 

**Title (ZH)**: TrustRAG：提高RAG模型鲁棒性和可信度 

**Authors**: Huichi Zhou, Kin-Hei Lee, Zhonghao Zhan, Yue Chen, Zhenhao Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.00879)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems enhance large language models (LLMs) by integrating external knowledge sources, enabling more accurate and contextually relevant responses tailored to user queries. However, these systems remain vulnerable to corpus poisoning attacks that can significantly degrade LLM performance through the injection of malicious content. To address these challenges, we propose TrustRAG, a robust framework that systematically filters compromised and irrelevant content before it reaches the language model. Our approach implements a two-stage defense mechanism: first, it employs K-means clustering to identify potential attack patterns in retrieved documents based on their semantic embeddings, effectively isolating suspicious content. Second, it leverages cosine similarity and ROUGE metrics to detect malicious documents while resolving discrepancies between the model's internal knowledge and external information through a self-assessment process. TrustRAG functions as a plug-and-play, training-free module that integrates seamlessly with any language model, whether open or closed-source, maintaining high contextual relevance while strengthening defenses against attacks. Through extensive experimental validation, we demonstrate that TrustRAG delivers substantial improvements in retrieval accuracy, efficiency, and attack resistance compared to existing approaches across multiple model architectures and datasets. We have made TrustRAG available as open-source software at \url{this https URL}. 

**Abstract (ZH)**: 检索增强生成（RAG）系统通过整合外部知识源来强化大型语言模型（LLMs），从而使模型能够提供更准确和上下文相关的响应，更好地满足用户查询的需求。然而，这些系统仍然容易受到语料库投毒攻击的影响，这种攻击可以通过注入恶意内容显著降低LLM的性能。为了解决这些问题，我们提出了TrustRAG，这是一个稳健的框架，能够系统性地过滤掉被篡改和无关的内容，从而防止其被传递给语言模型。该方法采用两阶段的防御机制：首先，使用K-means聚类根据文档的语义嵌入识别潜在的攻击模式，从而有效隔离可疑内容；其次，利用余弦相似度和ROUGE指标来检测恶意文档，并通过模型内知识与外部信息之间的自我评估过程解决两者之间的不一致。TrustRAG作为插件式、无需训练的模块，能够无缝集成到任何语言模型中，无论其是开源还是闭源版本，同时保持高度的相关性并增强对抗攻击的能力。通过广泛的实验验证，我们证明了TrustRAG在多个模型架构和数据集中，在检索准确性、效率和攻击抵抗力方面都带来了显著改进，相比于现有方法具有明显优势。我们已将TrustRAG以开源软件的形式提供，并可以在 \url{this https URL} 下载。 

---
# LUSIFER: Language Universal Space Integration for Enhanced Multilingual Embeddings with Large Language Models 

**Title (ZH)**: LUSIFER：用于增强多语言嵌入的大语言模型通用空间集成 

**Authors**: Hieu Man, Nghia Trung Ngo, Viet Dac Lai, Ryan A. Rossi, Franck Dernoncourt, Thien Huu Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2501.00874)  

**Abstract**: Recent advancements in large language models (LLMs) based embedding models have established new state-of-the-art benchmarks for text embedding tasks, particularly in dense vector-based retrieval. However, these models predominantly focus on English, leaving multilingual embedding capabilities largely unexplored. To address this limitation, we present LUSIFER, a novel zero-shot approach that adapts LLM-based embedding models for multilingual tasks without requiring multilingual supervision. LUSIFER's architecture combines a multilingual encoder, serving as a language-universal learner, with an LLM-based embedding model optimized for embedding-specific tasks. These components are seamlessly integrated through a minimal set of trainable parameters that act as a connector, effectively transferring the multilingual encoder's language understanding capabilities to the specialized embedding model. Additionally, to comprehensively evaluate multilingual embedding performance, we introduce a new benchmark encompassing 5 primary embedding tasks, 123 diverse datasets, and coverage across 14 languages. Extensive experimental results demonstrate that LUSIFER significantly enhances the multilingual performance across various embedding tasks, particularly for medium and low-resource languages, without requiring explicit multilingual training data. 

**Abstract (ZH)**: 基于大型语言模型（LLM）的嵌入模型近年来在文本嵌入任务中建立了新的最先进基准，尤其是在密集向量检索方面。然而，这些模型主要关注英语，而多语言嵌入能力却被广泛忽视。为解决这一局限性，我们提出了LUSIFER，一种新颖的零样本方法，能够在无需多语言监督的情况下，将基于LLM的嵌入模型适配到多语言任务中。LUSIFER架构结合了多语言编码器，作为语言通用的学习者，以及一个专门为嵌入特定任务优化的基于LLM的嵌入模型。这些组件通过少量可训练参数无缝集成，这些参数充当连接器，有效地将多语言编码器的语言理解能力转移到专门的嵌入模型中。此外，为了全面评估多语言嵌入性能，我们引入了一个新的基准，该基准涵盖了5项主要嵌入任务、123个多样化的数据集，并且覆盖了14种语言。广泛的实验结果表明，LUSIFER在各种嵌入任务中显著提高了多语言性能，尤其对于中等和低资源语言，而无需使用明确的多语言训练数据。 

---
# Large Language Models Are Read/Write Policy-Makers for Simultaneous Generation 

**Title (ZH)**: 大规模语言模型是同时生成的读写政策制定者 

**Authors**: Shoutao Guo, Shaolei Zhang, Zhengrui Ma, Yang Feng  

**Link**: [PDF](https://arxiv.org/pdf/2501.00868)  

**Abstract**: Simultaneous generation models write generation results while reading streaming inputs, necessitating a policy-maker to determine the appropriate output timing. Existing simultaneous generation methods generally adopt the traditional encoder-decoder architecture and learn the generation and policy-making capabilities through complex dynamic programming techniques. Although LLMs excel at text generation, they face challenges in taking on the role of policy-makers through traditional training methods, limiting their exploration in simultaneous generation. To overcome these limitations, we propose a novel LLM-driven Simultaneous Generation (LSG) framework, which allows the off-the-shelf LLM to decide the generation timing and produce output concurrently. Specifically, LSG selects the generation policy that minimizes latency as the baseline policy. Referring to the baseline policy, LSG enables the LLM to devise an improved generation policy that better balances latency and generation quality, and writes generation results accordingly. Experiments on simultaneous translation and streaming automatic speech recognition tasks show that our method can achieve state-of-the-art performance utilizing the open-source LLMs and demonstrate practicality in real-world scenarios. 

**Abstract (ZH)**: 同时生成模型在读取流式输入的同时生成结果，需要决策者确定适当的输出时机。现有的同时生成方法通常采用传统的编码器-解码器架构，并通过复杂的动态规划技术来学习生成和决策制定的能力。尽管大型语言模型（LLMs）在文本生成方面表现出色，但在传统的训练方法下承担决策者的角色面临挑战，限制了它们在同时生成方面的探索。为了克服这些限制，我们提出了一种新颖的LLM驱动的即刻生成（LSG）框架，该框架允许即用型LLM决定生成时机并同时生成结果。具体来说，LSG 选择减少延迟的策略作为基准策略。参照基准策略，LSG 使LLM能够制定一个能更好地平衡延迟和生成质量的改进策略，并据此生成结果。在同时翻译和流式自动语音识别任务上的实验表明，我们的方法可以利用开源的LLMs达到最先进的性能，并在实际场景中具有实用价值。 

---
# Negative to Positive Co-learning with Aggressive Modality Dropout 

**Title (ZH)**: 负向到正向的模态共学习方法结合激进模态丢弃 

**Authors**: Nicholas Magal, Minh Tran, Riku Arakawa, Suzanne Nie  

**Link**: [PDF](https://arxiv.org/pdf/2501.00865)  

**Abstract**: This paper aims to document an effective way to improve multimodal co-learning by using aggressive modality dropout. We find that by using aggressive modality dropout we are able to reverse negative co-learning (NCL) to positive co-learning (PCL). Aggressive modality dropout can be used to "prep" a multimodal model for unimodal deployment, and dramatically increases model performance during negative co-learning, where during some experiments we saw a 20% gain in accuracy. We also benchmark our modality dropout technique against PCL to show that our modality drop out technique improves co-learning during PCL, although it does not have as much as an substantial effect as it does during NCL. Github: this https URL 

**Abstract (ZH)**: 本文旨在通过使用激进模态 Dropout 方法记录一种提高多模态协同学习有效性的方式。我们发现，通过使用激进模态 Dropout，可以将负协同学习（NCL）转变为正协同学习（PCL）。激进模态 Dropout 可以用来“预热”多模态模型以备单模态部署，并在负协同学习期间显著提高模型性能，在某些实验中我们看到了高达 20% 准确率的提升。此外，我们将模态 Dropout 技术与 PCL 进行基准测试，以展示我们的模态 Dropout 技术在 PCL 中能够改善协同学习效果，尽管其效果不如在 NCL 中显著。GitHub 地址：[这里](this https URL) 

---
# DiffETM: Diffusion Process Enhanced Embedded Topic Model 

**Title (ZH)**: DiffETM：增强扩散过程嵌入主题模型 

**Authors**: Wei Shao, Mingyang Liu, Linqi Song  

**Link**: [PDF](https://arxiv.org/pdf/2501.00862)  

**Abstract**: The embedded topic model (ETM) is a widely used approach that assumes the sampled document-topic distribution conforms to the logistic normal distribution for easier optimization. However, this assumption oversimplifies the real document-topic distribution, limiting the model's performance. In response, we propose a novel method that introduces the diffusion process into the sampling process of document-topic distribution to overcome this limitation and maintain an easy optimization process. We validate our method through extensive experiments on two mainstream datasets, proving its effectiveness in improving topic modeling performance. 

**Abstract (ZH)**: 嵌入主题模型（ETM）是一种广泛使用的统计方法，它假设采样的文档-主题分布符合逻辑正态分布，以便于优化过程。然而，这一假设过于简化了实际的文档-主题分布，限制了模型的性能。为应对这一局限，我们提出了一种新颖的方法，通过将扩散过程引入文档-主题分布的采样过程中来克服这一限制，同时保持优化过程的便捷性。我们通过在两个主流数据集上进行广泛实验验证了该方法的有效性，证明了它在提高主题建模性能方面的有效性。 

---
# LLM+AL: Bridging Large Language Models and Action Languages for Complex Reasoning about Actions 

**Title (ZH)**: LLM+AL：连接大型语言模型与动作语言以进行复杂动作推理 

**Authors**: Adam Ishay, Joohyung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2501.00830)  

**Abstract**: Large Language Models (LLMs) have made significant strides in various intelligent tasks but still struggle with complex action reasoning tasks that require systematic search. To address this limitation, we propose a method that bridges the natural language understanding capabilities of LLMs with the symbolic reasoning strengths of action languages. Our approach, termed "LLM+AL," leverages the LLM's strengths in semantic parsing and commonsense knowledge generation alongside the action language's proficiency in automated reasoning based on encoded knowledge. We compare LLM+AL against state-of-the-art LLMs, including ChatGPT-4, Claude 3 Opus, Gemini Ultra 1.0, and o1-preview, using benchmarks for complex reasoning about actions. Our findings indicate that, although all methods exhibit errors, LLM+AL, with relatively minimal human corrections, consistently leads to correct answers, whereas standalone LLMs fail to improve even with human feedback. LLM+AL also contributes to automated generation of action languages. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种智能任务中取得了显著进展，但在处理需要系统搜索的复杂动作推理任务方面仍然存在问题。为解决这一局限性，我们提出了一种方法，将LLMs的自然语言理解能力与动作语言的符号推理能力相结合。我们的方法名为“LLM+AL”，利用LLMs在语义解析和常识知识生成方面的优势，以及动作语言在基于编码知识的自动化推理方面的专长。我们使用针对复杂动作推理的基准测试，将LLM+AL与最新的LLMs（包括ChatGPT-4、Claude 3 Opus、Gemini Ultra 1.0和o1-preview）进行对比。研究结果表明，尽管所有方法都存在错误，但LLM+AL在相对较少的人工修正的情况下，能够持续给出正确答案，而独立的LLMs即使在得到人类反馈的情况下也未能改进。此外，LLM+AL还促进了动作语言的自动化生成。 

---
# Embedding Style Beyond Topics: Analyzing Dispersion Effects Across Different Language Models 

**Title (ZH)**: 超越主题的嵌入风格：分析不同语言模型中的分散效应 

**Authors**: Benjamin Icard, Evangelia Zve, Lila Sainero, Alice Breton, Jean-Gabriel Ganascia  

**Link**: [PDF](https://arxiv.org/pdf/2501.00828)  

**Abstract**: This paper analyzes how writing style affects the dispersion of embedding vectors across multiple, state-of-the-art language models. While early transformer models primarily aligned with topic modeling, this study examines the role of writing style in shaping embedding spaces. Using a literary corpus that alternates between topics and styles, we compare the sensitivity of language models across French and English. By analyzing the particular impact of style on embedding dispersion, we aim to better understand how language models process stylistic information, contributing to their overall interpretability. 

**Abstract (ZH)**: 本文分析了写作风格如何影响嵌入向量在多个最先进的语言模型中的分布。早期的变压器模型主要与主题建模相关，而本研究则探讨了写作风格在塑造嵌入空间中的作用。通过使用交替包含不同主题和风格的文学语料库，我们比较了法语和英语语言模型对风格的敏感性。通过分析风格对嵌入分布的特定影响，本文旨在更好地理解语言模型如何处理风格信息，从而提高其整体可解释性。 

---
# Reasoning-Oriented and Analogy-Based Methods for Locating and Editing in Zero-Shot Event-Relational Reasoning 

**Title (ZH)**: 面向推理和类比导向的方法在零样本事件关系推理中的定位与编辑 

**Authors**: Jingyao Tang, Lishuang Li, Liteng Mi, Haiming Wu, Hongbin Lu  

**Link**: [PDF](https://arxiv.org/pdf/2501.00803)  

**Abstract**: Zero-shot event-relational reasoning is an important task in natural language processing, and existing methods jointly learn a variety of event-relational prefixes and inference-form prefixes to achieve such tasks. However, training prefixes consumes large computational resources and lacks interpretability. Additionally, learning various relational and inferential knowledge inefficiently exploits the connections between tasks. Therefore, we first propose a method for Reasoning-Oriented Locating and Editing (ROLE), which locates and edits the key modules of the language model for reasoning about event relations, enhancing interpretability and also resource-efficiently optimizing the reasoning ability. Subsequently, we propose a method for Analogy-Based Locating and Editing (ABLE), which efficiently exploits the similarities and differences between tasks to optimize the zero-shot reasoning capability. Experimental results show that ROLE improves interpretability and reasoning performance with reduced computational cost. ABLE achieves SOTA results in zero-shot reasoning. 

**Abstract (ZH)**: 零样本事件关系推理是自然语言处理中的一个重要な任务，现有的方法联合学习各种事件关系前缀和推理形式前缀以实现该任务。然而，训练这些前缀消耗了大量的计算资源并且缺乏可解释性。此外，学习各种关系和推理知识未能有效地利用任务之间的连接。因此，我们首先提出了一种Reasoning-Oriented Locating and Editing（ROLE）的方法，该方法通过定位和编辑语言模型中的关键模块来增强关于事件关系的推理能力，从而提高可解释性，并且以资源高效的方式优化推理能力。随后，我们提出了Analogy-Based Locating and Editing（ABLE）的方法，该方法有效地利用任务间的相似性和差异性来优化零样本推理能力。实验结果显示，ROLE在减少计算成本的同时提高了可解释性和推理性能。ABLE在零样本推理任务中达到了最佳表现。 

---
# Navigating Nuance: In Quest for Political Truth 

**Title (ZH)**: 探索政治真相中的细腻差异：追求政治真实之路 

**Authors**: Soumyadeep Sar, Dwaipayan Roy  

**Link**: [PDF](https://arxiv.org/pdf/2501.00782)  

**Abstract**: This study investigates the several nuanced rationales for countering the rise of political bias. We evaluate the performance of the Llama-3 (70B) language model on the Media Bias Identification Benchmark (MBIB), based on a novel prompting technique that incorporates subtle reasons for identifying political leaning. Our findings underscore the challenges of detecting political bias and highlight the potential of transfer learning methods to enhance future models. Through our framework, we achieve a comparable performance with the supervised and fully fine-tuned ConvBERT model, which is the state-of-the-art model, performing best among other baseline models for the political bias task on MBIB. By demonstrating the effectiveness of our approach, we contribute to the development of more robust tools for mitigating the spread of misinformation and polarization. Our codes and dataset are made publicly available in github. 

**Abstract (ZH)**: 本研究探讨了对抗政治偏见上升的多种细腻动机。我们基于一种新颖的提示技术，在Media Bias Identification Benchmark (MBIB) 上评估了Llama-3 (70B) 语言模型的表现，该技术综合了识别政治倾向的微妙原因。我们的研究结果突出了检测政治偏见的挑战，并强调了转移学习方法在未来模型中提升性能的潜力。通过我们的框架，我们在MBIB 的政治偏见任务中实现了与当前最先进的模型ConvBERT（经过监督和完全微调）相当的表现，后者在其他基线模型中表现最佳。通过展示我们方法的有效性，我们为开发更 robust 的工具以减少虚假信息和两极分化做出了贡献。我们的代码和数据集已在 GitHub 上公开发布。 

---
# Decoding the Flow: CauseMotion for Emotional Causality Analysis in Long-form Conversations 

**Title (ZH)**: 解码流变：CauseMotion在长篇对话中情感因果关系分析中的应用 

**Authors**: Yuxuan Zhang, Yulong Li, Zichen Yu, Feilong Tang, Zhixiang Lu, Chong Li, Kang Dang, Jionglong Su  

**Link**: [PDF](https://arxiv.org/pdf/2501.00778)  

**Abstract**: Long-sequence causal reasoning seeks to uncover causal relationships within extended time series data but is hindered by complex dependencies and the challenges of validating causal links. To address the limitations of large-scale language models (e.g., GPT-4) in capturing intricate emotional causality within extended dialogues, we propose CauseMotion, a long-sequence emotional causal reasoning framework grounded in Retrieval-Augmented Generation (RAG) and multimodal fusion. Unlike conventional methods relying only on textual information, CauseMotion enriches semantic representations by incorporating audio-derived features-vocal emotion, emotional intensity, and speech rate-into textual modalities. By integrating RAG with a sliding window mechanism, it effectively retrieves and leverages contextually relevant dialogue segments, thus enabling the inference of complex emotional causal chains spanning multiple conversational turns. To evaluate its effectiveness, we constructed the first benchmark dataset dedicated to long-sequence emotional causal reasoning, featuring dialogues with over 70 turns. Experimental results demonstrate that the proposed RAG-based multimodal integrated approach, the efficacy of substantially enhances both the depth of emotional understanding and the causal inference capabilities of large-scale language models. A GLM-4 integrated with CauseMotion achieves an 8.7% improvement in causal accuracy over the original model and surpasses GPT-4o by 1.2%. Additionally, on the publicly available DiaASQ dataset, CauseMotion-GLM-4 achieves state-of-the-art results in accuracy, F1 score, and causal reasoning accuracy. 

**Abstract (ZH)**: 长序列因果推理旨在揭示扩展时间序列数据中的因果关系，但由于复杂的依赖关系和验证因果链接的挑战而受到限制。为了克服大型语言模型（如GPT-4）在捕捉扩展对话中复杂情感因果关系方面的局限性，我们提出了CauseMotion，这是一种基于检索增强生成（RAG）和多模态融合的长序列情感因果推理框架。与仅依赖文本信息的传统方法不同，CauseMotion通过将音频提取的特征（即声情、情感强度和语速）融入文本模态中，丰富了语义表示。通过将RAG与滑动窗口机制结合，它有效地检索和利用上下文相关的话语片段，从而能够推断跨越多个对话回合的复杂情感因果链。为了评估其有效性，我们构建了首个专注于长序列情感因果推理的基准数据集，其中包含超过70个回合的对话。实验结果表明，基于RAG的多模态集成方法显著增强了大型语言模型的情感理解深度和因果推理能力。集成CauseMotion的GLM-4在因果准确性上比原模型提升了8.7%，并且超越了GPT-4o 1.2%。此外，在公开可用的DiaASQ数据集上，CauseMotion-GLM-4在准确率、F1分数和因果推理准确性方面达到了最先进的性能。 

---
# FitCF: A Framework for Automatic Feature Importance-guided Counterfactual Example Generation 

**Title (ZH)**: FitCF：一种自动特征重要性指导的反事实示例生成框架 

**Authors**: Qianli Wang, Nils Feldhus, Simon Ostermann, Luis Felipe Villa-Arenas, Sebastian Möller, Vera Schmitt  

**Link**: [PDF](https://arxiv.org/pdf/2501.00777)  

**Abstract**: Counterfactual examples are widely used in natural language processing (NLP) as valuable data to improve models, and in explainable artificial intelligence (XAI) to understand model behavior. The automated generation of counterfactual examples remains a challenging task even for large language models (LLMs), despite their impressive performance on many tasks. In this paper, we first introduce ZeroCF, a faithful approach for leveraging important words derived from feature attribution methods to generate counterfactual examples in a zero-shot setting. Second, we present a new framework, FitCF, which further verifies aforementioned counterfactuals by label flip verification and then inserts them as demonstrations for few-shot prompting, outperforming two state-of-the-art baselines. Through ablation studies, we identify the importance of each of FitCF's core components in improving the quality of counterfactuals, as assessed through flip rate, perplexity, and similarity measures. Furthermore, we show the effectiveness of LIME and Integrated Gradients as backbone attribution methods for FitCF and find that the number of demonstrations has the largest effect on performance. Finally, we reveal a strong correlation between the faithfulness of feature attribution scores and the quality of generated counterfactuals. 

**Abstract (ZH)**: 对抗事实样本在自然语言处理（NLP）和可解释人工智能（XAI）中被广泛用作改善模型的重要数据，并被用来理解模型的行为。尽管大型语言模型（LLMs）在许多任务上表现出色，但它们自动生成对抗事实样本的任务仍然是一项具有挑战性的任务。在本文中，我们首先介绍了ZeroCF，这是一种在零样本设置下利用特征归因方法提取的重要词汇来生成对抗事实样本的可靠方法。其次，我们提出了一种新的框架FitCF，该框架通过标签翻转验证进一步验证上述的对抗事实样本，并将它们作为少样本提示的示例注入，从而优于两种最先进的基线方法。通过消融研究，我们确定了FitCF各个核心组件在通过翻转率、困惑度和相似度指标评估的对抗事实样本质量改进中的重要性。此外，我们展示了LIME和集成梯度在FitCF中的基础归因方法的有效性，并发现示例的数量对表现的影响最大。最后，我们揭示了特征归因得分的忠实性与生成对抗事实样本质量之间存在强烈的关联。 

---
# Enhancing Transformers for Generalizable First-Order Logical Entailment 

**Title (ZH)**: 增强变换器以提高泛化一阶逻辑蕴含能力 

**Authors**: Tianshi Zheng, Jiazheng Wang, Zihao Wang, Jiaxin Bai, Hang Yin, Zheye Deng, Yangqiu Song, Jianxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.00759)  

**Abstract**: Transformers, as a fundamental deep learning architecture, have demonstrated remarkable capabilities in reasoning. This paper investigates the generalizable first-order logical reasoning ability of transformers with their parameterized knowledge and explores ways to improve it. The first-order reasoning capability of transformers is assessed through their ability to perform first-order logical entailment, which is quantitatively measured by their performance in answering knowledge graph queries. We establish connections between (1) two types of distribution shifts studied in out-of-distribution generalization and (2) the unseen knowledge and query settings discussed in the task of knowledge graph query answering, enabling a characterization of fine-grained generalizability. Results on our comprehensive dataset show that transformers outperform previous methods specifically designed for this task and provide detailed empirical evidence on the impact of input query syntax, token embedding, and transformer architectures on the reasoning capability of transformers. Interestingly, our findings reveal a mismatch between positional encoding and other design choices in transformer architectures employed in prior practices. This discovery motivates us to propose a more sophisticated, logic-aware architecture, TEGA, to enhance the capability for generalizable first-order logical entailment in transformers. 

**Abstract (ZH)**: 作为基础的深度学习架构，变换器展示了在推理方面的能力。本文研究了变换器通过参数化知识所体现的可泛化的初等逻辑推理能力，并探讨了提升这种能力的方法。通过评估变换器执行初等逻辑蕴含的能力，我们定量测量了其在回答知识图谱查询方面的能力，以此评估变换器的初等逻辑推理能力。我们建立了以下两个方面之间的联系：（1）分布外泛化中研究的两种分布转移类型，与（2）知识图谱查询回答任务中讨论的未见过的知识和查询设置，从而为细致的泛化能力提供了描述。我们的综合数据集上的实验结果表明，变换器在执行此任务方面优于之前专门为该任务设计的方法，并提供了关于输入查询语法、词元嵌入和变换器架构对变换器推理能力影响的详细实证证据。有趣的是，我们的研究发现以往实践中使用的变换器架构中的位置编码与其他设计选择之间存在不匹配。这一发现促使我们提出了逻辑意识更强的架构TEGA，以增强变换器在可泛化的初等逻辑蕴含方面的推理能力。 

---
# DIVE: Diversified Iterative Self-Improvement 

**Title (ZH)**: DIVE：多样化迭代自我提升 

**Authors**: Yiwei Qin, Yixiu Liu, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.00747)  

**Abstract**: Recent advances in large language models (LLMs) have demonstrated the effectiveness of Iterative Self-Improvement (ISI) techniques. However, continuous training on self-generated data leads to reduced output diversity, a limitation particularly critical in reasoning tasks where diverse solution paths are essential. We present DIVE (Diversified Iterative Self-Improvement), a novel framework that addresses this challenge through two key components: Sample Pool Expansion for broader solution exploration, and Data Selection for balancing diversity and quality in preference pairs. Experiments on MATH and GSM8k datasets show that DIVE achieves a 10% to 45% relative increase in output diversity metrics while maintaining performance quality compared to vanilla ISI. Our ablation studies confirm both components' significance in achieving these improvements. Code is available at this https URL. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）的最新进展证明了迭代自我改进（Iterative Self-Improvement, ISI）技术的有效性。然而，持续使用自动生成的数据进行训练会导致输出多样性降低，这一限制在需要探索多种解题路径的推理任务中尤为关键。我们提出了一种新的框架DIVE（Diversified Iterative Self-Improvement），通过两个关键组件来解决这一挑战：样本池扩展以进行更广泛的解决方案探索，以及数据选择以在偏好配对中平衡多样性和质量。在MATH和GSM8k数据集上的实验表明，DIVE在保持性能质量的同时，使输出多样性的指标提高了10%到45%。我们的消融研究证实了这两个组件在实现这些改进中的重要性。代码可供参考，地址为：[这个链接](https://)。 

---
# Dynamics of Adversarial Attacks on Large Language Model-Based Search Engines 

**Title (ZH)**: 基于大型语言模型的搜索引擎中的恶意攻击 dynamics 

**Authors**: Xiyang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2501.00745)  

**Abstract**: The increasing integration of Large Language Model (LLM) based search engines has transformed the landscape of information retrieval. However, these systems are vulnerable to adversarial attacks, especially ranking manipulation attacks, where attackers craft webpage content to manipulate the LLM's ranking and promote specific content, gaining an unfair advantage over competitors. In this paper, we study the dynamics of ranking manipulation attacks. We frame this problem as an Infinitely Repeated Prisoners' Dilemma, where multiple players strategically decide whether to cooperate or attack. We analyze the conditions under which cooperation can be sustained, identifying key factors such as attack costs, discount rates, attack success rates, and trigger strategies that influence player behavior. We identify tipping points in the system dynamics, demonstrating that cooperation is more likely to be sustained when players are forward-looking. However, from a defense perspective, we find that simply reducing attack success probabilities can, paradoxically, incentivize attacks under certain conditions. Furthermore, defensive measures to cap the upper bound of attack success rates may prove futile in some scenarios. These insights highlight the complexity of securing LLM-based systems. Our work provides a theoretical foundation and practical insights for understanding and mitigating their vulnerabilities, while emphasizing the importance of adaptive security strategies and thoughtful ecosystem design. 

**Abstract (ZH)**: 大规模语言模型（LLM）为基础的搜索引擎的不断增加的整合已经改变了信息检索的格局。然而，这些系统对敌对攻击特别容易，特别是排名操控攻击，其中攻击者精心设计网页内容以操纵LLM的排名，并推广特定内容，从而在竞争对手中获得不公平的优势。在本文中，我们研究了排名操控攻击的动态。我们将这个问题描述为无限重复的囚徒困境，其中多名玩家战略性地决定合作还是攻击。我们分析了合作可以持续的条件，识别了影响玩家行为的关键因素，如攻击成本、贴现率、攻击成功率和触发策略。我们识别了系统动态中的临界点，表明当玩家具有前瞻性的视角时，合作更容易持续。然而，从防御角度来看，我们发现简单地降低攻击成功率在某些条件下反而会激励攻击。此外，限制攻击成功率上限的防御措施在某些情境下也许无效。这些见解突显了保护基于LLM系统的复杂性。我们的工作为理解并缓解其脆弱性提供了理论基础和实用洞察，同时强调了适应性安全策略和精心生态系统设计的重要性。 

---
# On Importance of Layer Pruning for Smaller BERT Models and Low Resource Languages 

**Title (ZH)**: 小型BERT模型和低资源语言中层剪枝的重要性探究 

**Authors**: Mayur Shirke, Amey Shembade, Madhushri Wagh, Pavan Thorat, Raviraj Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2501.00733)  

**Abstract**: This study explores the effectiveness of layer pruning for developing more efficient BERT models tailored to specific downstream tasks in low-resource languages. Our primary objective is to evaluate whether pruned BERT models can maintain high performance while reducing model size and complexity. We experiment with several BERT variants, including MahaBERT-v2 and Google-Muril, applying different pruning strategies and comparing their performance to smaller, scratch-trained models like MahaBERT-Small and MahaBERT-Smaller. We fine-tune these models on Marathi datasets, specifically Short Headlines Classification (SHC), Long Paragraph Classification (LPC) and Long Document Classification (LDC), to assess their classification accuracy. Our findings demonstrate that pruned models, despite having fewer layers, achieve comparable performance to their fully-layered counterparts while consistently outperforming scratch-trained models of similar size. Notably, pruning layers from the middle of the model proves to be the most effective strategy, offering performance competitive with pruning from the top and bottom. However, there is no clear winner, as different pruning strategies perform better in different model and dataset combinations. Additionally, monolingual BERT models outperform multilingual ones in these experiments. This approach, which reduces computational demands, provides a faster and more efficient alternative to training smaller models from scratch, making advanced NLP models more accessible for low-resource languages without compromising classification accuracy. 

**Abstract (ZH)**: 本研究探讨了剪枝技术在开发面向特定下游任务的低资源语言高效BERT模型中的有效性。我们的主要目标是在保持高性能的同时，通过剪枝来减少模型规模和复杂度。我们尝试了几种不同的BERT变体，包括MahaBERT-v2和Google-Muril，并应用了不同的剪枝策略，将其性能与MahaBERT-Small和MahaBERT-Smaller等从头训练的小模型进行对比。我们针对马拉地语数据集——短标题分类（SHC）、长段落分类（LPC）和长文档分类（LDC）进行微调，以评估其分类准确度。研究发现，尽管剪枝后的模型拥有较少的层，但它仍能达到与完整层模型相当的性能，并且在大多数情况下优于相似规模的从头训练模型。值得注意的是，从模型中间剪枝是最有效的策略，其性能与从顶部和底部剪枝相当。然而，没有一个明确的胜者，因为不同的剪枝策略在不同模型和数据集组合中表现更好。此外，在这些实验中，单语BERT模型优于多语BERT模型。这种方法通过减少计算需求，提供了一种更快、更高效的替代方案，能够训练出更小的模型，使先进的NLP模型对于低资源语言更加普及，而不牺牲分类准确度。 

---
# eRevise+RF: A Writing Evaluation System for Assessing Student Essay Revisions and Providing Formative Feedback 

**Title (ZH)**: eRevise+RF：一种评估学生作文修订并提供形成性反馈的写作评价系统 

**Authors**: Zhexiong Liu, Diane Litman, Elaine Wang, Tianwen Li, Mason Gobat, Lindsay Clare Matsumura, Richard Correnti  

**Link**: [PDF](https://arxiv.org/pdf/2501.00715)  

**Abstract**: The ability to revise essays in response to feedback is important for students' writing success. An automated writing evaluation (AWE) system that supports students in revising their essays is thus essential. We present eRevise+RF, an enhanced AWE system for assessing student essay revisions (e.g., changes made to an essay to improve its quality in response to essay feedback) and providing revision feedback. We deployed the system with 6 teachers and 406 students across 3 schools in Pennsylvania and Louisiana. The results confirmed its effectiveness in (1) assessing student essays in terms of evidence usage, (2) extracting evidence and reasoning revisions across essays, and (3) determining revision success in responding to feedback. The evaluation also suggested eRevise+RF is a helpful system for young students to improve their argumentative writing skills through revision and formative feedback. 

**Abstract (ZH)**: 根据反馈修订作文的能力对于学生的写作成功非常重要。因此，一个支持学生修订作文的自动写作评估（AWE）系统是必不可少的。我们提出了eRevise+RF，这是一种增强的AWE系统，用于评估学生的作文修订（例如，根据作文反馈进行修改以提高其质量）并提供反馈。我们在宾夕法尼亚州和路易斯安那州的3所学校部署了该系统，共6名教师和406名学生参与。结果表明，该系统在以下方面具有有效性：（1）评估学生作文中证据的使用情况；（2）提取作文间的证据和推理修订内容；（3）确定根据反馈进行修订的成功情况。评估还表明，eRevise+RF是一个有助于年轻学生通过修订和形成性反馈提高论证写作技能的有用系统。 

---
# CODEOFCONDUCT at Multilingual Counterspeech Generation: A Context-Aware Model for Robust Counterspeech Generation in Low-Resource Languages 

**Title (ZH)**: 多语言反驳生成的行为规范：一种面向低资源语言的上下文感知模型以实现稳健的反驳生成 

**Authors**: Michael Bennie, Bushi Xiao, Chryseis Xinyi Liu, Demi Zhang, Jian Meng, Alayo Tripp  

**Link**: [PDF](https://arxiv.org/pdf/2501.00713)  

**Abstract**: This paper introduces a context-aware model for robust counterspeech generation, which achieved significant success in the MCG-COLING-2025 shared task. Our approach particularly excelled in low-resource language settings. By leveraging a simulated annealing algorithm fine-tuned on multilingual datasets, the model generates factually accurate responses to hate speech.
We demonstrate state-of-the-art performance across four languages (Basque, English, Italian, and Spanish), with our system ranking first for Basque, second for Italian, and third for both English and Spanish. Notably, our model swept all three top positions for Basque, highlighting its effectiveness in low-resource scenarios.
Evaluation of the shared task employs both traditional metrics (BLEU, ROUGE, BERTScore, Novelty) and JudgeLM based on LLM. We present a detailed analysis of our results, including an empirical evaluation of the model performance and comprehensive score distributions across evaluation metrics.
This work contributes to the growing body of research on multilingual counterspeech generation, offering insights into developing robust models that can adapt to diverse linguistic and cultural contexts in the fight against online hate speech. 

**Abstract (ZH)**: 本文介绍了一种面向上下文的鲁棒反仇恨言论生成模型，该模型在2025年MCG-COLING共享任务中取得了显著成功。我们的方法特别适用于资源有限的语言环境。通过在多语种数据集上微调模拟退火算法，该模型能够生成事实准确的反仇恨言论。

我们在四种语言（巴斯克语、英语、意大利语和西班牙语）上展示了前沿的表现，我们的系统在巴斯克语中排名第一，意大利语中排名第二，而英语和西班牙语中则分别排名第三。值得注意的是，我们的模型在巴斯克语所有三个顶级位置的比赛中均排名首位，突显了其在资源有限环境中的有效性。

共享任务的评估采用了传统指标（BLEU、ROUGE、BERTScore、新颖性）和基于大规模语言模型（LLM）的评价指标（JudgeLM）。我们详细分析了我们的结果，包括对模型性能的实证评估以及各评估指标下的综合评分分布情况。

本文为多语言反仇恨言论生成领域的研究贡献了一部分成果，提供了关于开发能够适应多种语言和文化背景的鲁棒模型的见解，以对抗在线仇恨言论。 

---
# Rethinking Addressing in Language Models via Contexualized Equivariant Positional Encoding 

**Title (ZH)**: 通过上下文不变位置编码重新审视语言模型中的地址表示 

**Authors**: Jiajun Zhu, Peihao Wang, Ruisi Cai, Jason D. Lee, Pan Li, Zhangyang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.00712)  

**Abstract**: Transformers rely on both content-based and position-based addressing mechanisms to make predictions, but existing positional encoding techniques often diminish the effectiveness of position-based addressing. Many current methods enforce rigid patterns in attention maps, limiting the ability to model long-range dependencies and adapt to diverse tasks. Additionally, most positional encodings are learned as general biases, lacking the specialization required for different instances within a dataset. To address this, we propose con$\textbf{T}$extualized equivari$\textbf{A}$nt $\textbf{P}$osition $\textbf{E}$mbedding ($\textbf{TAPE}$), a novel framework that enhances positional embeddings by incorporating sequence content across layers. TAPE introduces dynamic, context-aware positional encodings, overcoming the constraints of traditional fixed patterns. By enforcing permutation and orthogonal equivariance, TAPE ensures the stability of positional encodings during updates, improving robustness and adaptability. Our method can be easily integrated into pre-trained transformers, offering parameter-efficient fine-tuning with minimal overhead. Extensive experiments shows that TAPE achieves superior performance in language modeling, arithmetic reasoning, and long-context retrieval tasks compared to existing positional embedding techniques. 

**Abstract (ZH)**: 变压器依赖于内容导向和位置导向的双重寻址机制来进行预测，但现有的位置编码技术往往削弱了位置导向寻址的有效性。当前许多方法在注意力图上施加了严格模式，限制了对长距离依赖关系建模的能力，并限制了针对不同任务的适应性。此外，大多数位置编码是作为一般偏置学习的，缺乏针对数据集内不同实例的专业化。为了解决这个问题，我们提出了**TAPE**（内容导向的变交换位置嵌入），这是一种新的框架，通过在各层中整合序列内容来增强位置嵌入。TAPE 引入了动态的、上下文感知的位置编码，克服了传统固定模式的约束。通过施加置换和正交的变交换性，TAPE 确保了位置编码在更新过程中的稳定性，从而提高了鲁棒性和适应性。该方法可以轻松集成到预训练的变压器中，提供参数高效的微调，同时保持较低的开销。广泛的实验表明，与现有的位置嵌入技术相比，TAPE 在语言建模、算术推理和长上下文检索任务中表现出更优的性能。 

---
# PANDA -- Paired Anti-hate Narratives Dataset from Asia: Using an LLM-as-a-Judge to Create the First Chinese Counterspeech Dataset 

**Title (ZH)**: PANDA——源自亚洲的配对反 Hate 叙述数据集：使用大语言模型作为裁判创建首个中文反仇恨言论数据集 

**Authors**: Michael Bennie, Demi Zhang, Bushi Xiao, Jing Cao, Chryseis Xinyi Liu, Jian Meng, Alayo Tripp  

**Link**: [PDF](https://arxiv.org/pdf/2501.00697)  

**Abstract**: Despite the global prevalence of Modern Standard Chinese language, counterspeech (CS) resources for Chinese remain virtually nonexistent. To address this gap in East Asian counterspeech research we introduce the a corpus of Modern Standard Mandarin counterspeech that focuses on combating hate speech in Mainland China. This paper proposes a novel approach of generating CS by using an LLM-as-a-Judge, simulated annealing, LLMs zero-shot CN generation and a round-robin algorithm. This is followed by manual verification for quality and contextual relevance. This paper details the methodology for creating effective counterspeech in Chinese and other non-Eurocentric languages, including unique cultural patterns of which groups are maligned and linguistic patterns in what kinds of discourse markers are programmatically marked as hate speech (HS). Analysis of the generated corpora, we provide strong evidence for the lack of open-source, properly labeled Chinese hate speech data and the limitations of using an LLM-as-Judge to score possible answers in Chinese. Moreover, the present corpus serves as the first East Asian language based CS corpus and provides an essential resource for future research on counterspeech generation and evaluation. 

**Abstract (ZH)**: 尽管现代标准汉语在全球范围内普遍存在，但汉语的对抗性言论（Counterspeech, CS）资源几乎不存在。为弥补东亚地区对抗性言论研究的空白，我们引入了一项针对中国大陆地区恶意言论的现代标准普通话CS语料库。本文提出了一种创新的方法，通过使用“语言模型作为法官”（LLM-as-a-Judge）、模拟退火算法、零样本汉语生成以及轮换算法来生成CS。在此之后，进行人工验证以确保质量和上下文相关性。本文详细说明了如何在汉语及其他非欧罗巴中心语言中创建有效的CS，包括不同的文化模式，哪些群体受到诋毁，以及哪种类型的语用标记在编程过程中被标记为恶意言论（Harassment Statements, HS）。通过分析生成的语料库，本文提供了有力证据，表明公开来源、正确标注的汉语恶意言论数据缺乏，并且仅使用“语言模型作为法官”来评分汉语潜在答案存在局限性。此外，本文提供的语料库是首个基于东亚语言的CS语料库，为今后的CS生成和评估研究提供了宝贵资源。 

---
# Labels Generated by Large Language Model Helps Measuring People's Empathy in Vitro 

**Title (ZH)**: 大型语言模型生成的标签有助于体外测量人们的情绪共感能力 

**Authors**: Md Rakibul Hasan, Yue Yao, Md Zakir Hossain, Aneesh Krishna, Imre Rudas, Shafin Rahman, Tom Gedeon  

**Link**: [PDF](https://arxiv.org/pdf/2501.00691)  

**Abstract**: Large language models (LLMs) have revolutionised numerous fields, with LLM-as-a-service (LLMSaaS) having a strong generalisation ability that offers accessible solutions directly without the need for costly training. In contrast to the widely studied prompt engineering for task solving directly (in vivo), this paper explores its potential in in-vitro applications. These involve using LLM to generate labels to help the supervised training of mainstream models by (1) noisy label correction and (2) training data augmentation with LLM-generated labels. In this paper, we evaluate this approach in the emerging field of empathy computing -- automating the prediction of psychological questionnaire outcomes from inputs like text sequences. Specifically, crowdsourced datasets in this domain often suffer from noisy labels that misrepresent underlying empathy. By leveraging LLM-generated labels to train pre-trained language models (PLMs) like RoBERTa, we achieve statistically significant accuracy improvements over baselines, achieving a state-of-the-art Pearson correlation coefficient of 0.648 on NewsEmp benchmarks. In addition, we bring insightful discussions, including current challenges in empathy computing, data biases in training data and evaluation metric selection. Code and LLM-generated data are available at this https URL (available once the paper is accepted). 

**Abstract (ZH)**: 大型语言模型（LLMs）已经革新了众多领域，LLM即服务（LLMSaaS）因其强大的泛化能力，能够提供无需昂贵训练成本的直接可访问解决方案。与广泛研究的任务直接解决方法（in vivo）的提示工程不同，本文探讨了其在体外（in-vitro）应用中的潜在价值。这些应用包括使用LLM生成标签以辅助主流模型的监督训练，具体表现在（1）嘈杂标签修正和（2）通过LLM生成的标签进行训练数据增强。在本文中，我们评估了该方法在新兴领域的情感计算中的应用——从如文本序列等输入自动预测心理问卷结果。具体而言，领域内的众包数据集常常存在难以代表真实同理心的嘈杂标签。通过利用LLM生成的标签训练像RoBERTa等预训练语言模型（PLMs），我们实现了显著的准确性改进，在NewsEmp基准测试中达到了0.648的最高皮尔逊相关系数。此外，我们还进行了深刻的讨论，包括情感计算领域的现有挑战、训练数据中的数据偏差以及评估指标的选择。相关代码和LLM生成的数据将在论文被接受后公布，请访问以下网址：[这个网址]。 

---
# 2 OLMo 2 Furious 

**Title (ZH)**: 很抱歉，您提供的“2 OLMo 2 Furious”似乎不是一个完整的论文标题或内容摘要，它看起来更像是一个简短的代码或缩写。为了能够准确地翻译并符合学术规范，我需要完整的句子或更详细的内容。如果是“OLMo 2 Furious”，可以推测OLMo可能是某个技术或方法的缩写，但具体含义不明。如果是“2 OLMo 2 Furious”，则含义更加不明确。

请您提供更多的背景信息或完整句子，我将很乐意帮助您进行准确的翻译和规范化处理。 

**Authors**: Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia, Yuling Gu, Shengyi Huang, Matt Jordan, Nathan Lambert, Dustin Schwenk, Oyvind Tafjord, Taira Anderson, David Atkinson, Faeze Brahman, Christopher Clark, Pradeep Dasigi, Nouha Dziri, Michal Guerquin, Hamish Ivison, Pang Wei Koh, Jiacheng Liu, Saumya Malik, William Merrill, Lester James V. Miranda, Jacob Morrison, Tyler Murray, Crystal Nam, Valentina Pyatkin, Aman Rangapur, Michael Schmitz, Sam Skjonsberg, David Wadden, Christopher Wilhelm, Michael Wilson, Luke Zettlemoyer, Ali Farhadi, Noah A. Smith, Hannaneh Hajishirzi  

**Link**: [PDF](https://arxiv.org/pdf/2501.00656)  

**Abstract**: We present OLMo 2, the next generation of our fully open language models. OLMo 2 includes dense autoregressive models with improved architecture and training recipe, pretraining data mixtures, and instruction tuning recipes. Our modified model architecture and training recipe achieve both better training stability and improved per-token efficiency. Our updated pretraining data mixture introduces a new, specialized data mix called Dolmino Mix 1124, which significantly improves model capabilities across many downstream task benchmarks when introduced via late-stage curriculum training (i.e. specialized data during the annealing phase of pretraining). Finally, we incorporate best practices from Tülu 3 to develop OLMo 2-Instruct, focusing on permissive data and extending our final-stage reinforcement learning with verifiable rewards (RLVR). Our OLMo 2 base models sit at the Pareto frontier of performance to compute, often matching or outperforming open-weight only models like Llama 3.1 and Qwen 2.5 while using fewer FLOPs and with fully transparent training data, code, and recipe. Our fully open OLMo 2-Instruct models are competitive with or surpassing open-weight only models of comparable size, including Qwen 2.5, Llama 3.1 and Gemma 2. We release all OLMo 2 artifacts openly -- models at 7B and 13B scales, both pretrained and post-trained, including their full training data, training code and recipes, training logs and thousands of intermediate checkpoints. The final instruction model is available on the Ai2 Playground as a free research demo. 

**Abstract (ZH)**: 我们介绍了OLMo 2，这是我们完全开放语言模型的下一代。OLMo 2 包含改进的密集自回归模型，采用改进的架构和训练方案，混合的预训练数据集，以及指令微调方案。我们修改的模型架构和训练方案在提高训练稳定性和提升每个令牌效率方面均表现更优。我们更新的预训练数据集混合引入了新的专门化数据混合Dolmino Mix 1124，当通过后期训练课程引入时（即，在预训练退火阶段期间引入专门化数据），显著提高了模型在许多下游任务基准上的能力。最后，我们借鉴Tülu 3的最佳实践，开发了OLMo 2-Instruct，重点关注宽松的数据，其最终阶段强化学习还扩展了可验证奖励（RLVR）。我们的OLMo 2基础模型位于性能与计算的帕累托前沿，通常能与或优于只使用开放权重的模型（如Llama 3.1和Qwen 2.5）相当或超越它们，同时使用较少的FLOPs，且训练数据、代码及配方完全透明。我们的完全开放的OLMo 2-Instruct模型在可比规模的只使用开放权重模型中具有竞争力，甚至超越了Qwen 2.5、Llama 3.1和Gemma 2。我们开放地发布了所有OLMo 2的全部艺术品，包括7B和13B规模的基础模型和后续模型，它们包括完整的训练数据、训练代码及配方、训练日志及数千个中间检查点。最终的指令模型在Ai2 Playground上作为免费的研究演示供公众使用。 

---
# Efficient Standardization of Clinical Notes using Large Language Models 

**Title (ZH)**: 使用大型语言模型高效标准化临床笔记 

**Authors**: Daniel B. Hier, Michael D. Carrithers, Thanh Son Do, Tayo Obafemi-Ajayi  

**Link**: [PDF](https://arxiv.org/pdf/2501.00644)  

**Abstract**: Clinician notes are a rich source of patient information but often contain inconsistencies due to varied writing styles, colloquialisms, abbreviations, medical jargon, grammatical errors, and non-standard formatting. These inconsistencies hinder the extraction of meaningful data from electronic health records (EHRs), posing challenges for quality improvement, population health, precision medicine, decision support, and research.
We present a large language model approach to standardizing a corpus of 1,618 clinical notes. Standardization corrected an average of $4.9 +/- 1.8$ grammatical errors, $3.3 +/- 5.2$ spelling errors, converted $3.1 +/- 3.0$ non-standard terms to standard terminology, and expanded $15.8 +/- 9.1$ abbreviations and acronyms per note. Additionally, notes were re-organized into canonical sections with standardized headings. This process prepared notes for key concept extraction, mapping to medical ontologies, and conversion to interoperable data formats such as FHIR.
Expert review of randomly sampled notes found no significant data loss after standardization. This proof-of-concept study demonstrates that standardization of clinical notes can improve their readability, consistency, and usability, while also facilitating their conversion into interoperable data formats. 

**Abstract (ZH)**: 医生笔记是患者信息的丰富来源，但由于其写作风格各异、口语化、缩写、医学术语、语法错误和非标准化格式的存在，往往会出现不一致性。这些不一致性阻碍了从电子健康记录（EHR）中提取有意义的数据，对质量改进、公共卫生、精准医学、决策支持和研究构成了挑战。

我们提出了一种大规模语言模型方法，用于标准化1,618份临床笔记。标准化过程平均纠正了4.9±1.8个语法错误、3.3±5.2个拼写错误、将3.1±3.0个非标准术语转换为标准术语，并扩展了每份笔记中的15.8±9.1个缩写词和缩写。此外，将笔记重新组织为具有标准化标题的标准段落。这一过程为关键概念提取、映射到医学本体以及转换为如FHIR之类的兼容数据格式做好了准备。

专家随机抽取的笔记审查结果显示，在标准化之后没有显著的数据丢失。这一概念性研究展示了标准化临床笔记可以提高其可读性、一致性和实用性，同时也促进了其转换为兼容数据格式。 

---
# Toward Corpus Size Requirements for Training and Evaluating Depression Risk Models Using Spoken Language 

**Title (ZH)**: 面向语音语言训练和评估抑郁风险模型的语料库规模要求研究 

**Authors**: Tomek Rutowski, Amir Harati, Elizabeth Shriberg, Yang Lu, Piotr Chlebek, Ricardo Oliveira  

**Link**: [PDF](https://arxiv.org/pdf/2501.00617)  

**Abstract**: Mental health risk prediction is a growing field in the speech community, but many studies are based on small corpora. This study illustrates how variations in test and train set sizes impact performance in a controlled study. Using a corpus of over 65K labeled data points, results from a fully crossed design of different train/test size combinations are provided. Two model types are included: one based on language and the other on speech acoustics. Both use methods current in this domain. An age-mismatched test set was also included. Results show that (1) test sizes below 1K samples gave noisy results, even for larger training set sizes; (2) training set sizes of at least 2K were needed for stable results; (3) NLP and acoustic models behaved similarly with train/test size variations, and (4) the mismatched test set showed the same patterns as the matched test set. Additional factors are discussed, including label priors, model strength and pre-training, unique speakers, and data lengths. While no single study can specify exact size requirements, results demonstrate the need for appropriately sized train and test sets for future studies of mental health risk prediction from speech and language. 

**Abstract (ZH)**: 心理健康风险预测是语音社区中的一个快速发展领域，但许多研究基于小型语料库。本研究展示了在受控条件下不同测试集和训练集大小对性能的影响。使用超过65000个标记数据点的语料库，提供了不同训练集/测试集大小组合的全面交叉设计结果。包括两种模型类型：一种基于语言，另一种基于语音声学。这两种模型都采用了本领域当前常用的方法。此外，还包括一个年龄不匹配的测试集。结果表明：（1）测试集小于1000个样本时，即使是较大的训练集，也会得到噪声结果；（2）至少需要2000个样本的训练集才能获得稳定的结果；（3）在训练集和测试集大小变化时，自然语言处理（NLP）模型和声学模型表现出相似的行为；（4）年龄不匹配的测试集与匹配的测试集表现出相同模式。文中还讨论了其他因素，包括标签先验、模型强度和预训练、独特说话人和数据长度。虽然没有单一的研究能够确定具体的数据集大小要求，但结果表明，在未来关于从语音和语言预测心理健康风险的研究中，需要适当大小的训练集和测试集。 

---
# Optimizing Speech-Input Length for Speaker-Independent Depression Classification 

**Title (ZH)**: 优化语音输入长度以实现独立说话人抑郁分类 

**Authors**: Tomasz Rutowski, Amir Harati, Yang Lu, Elizabeth Shriberg  

**Link**: [PDF](https://arxiv.org/pdf/2501.00608)  

**Abstract**: Machine learning models for speech-based depression classification offer promise for health care applications. Despite growing work on depression classification, little is understood about how the length of speech-input impacts model performance. We analyze results for speaker-independent depression classification using a corpus of over 1400 hours of speech from a human-machine health screening application. We examine performance as a function of response input length for two NLP systems that differ in overall performance.
Results for both systems show that performance depends on natural length, elapsed length, and ordering of the response within a session. Systems share a minimum length threshold, but differ in a response saturation threshold, with the latter higher for the better system. At saturation it is better to pose a new question to the speaker, than to continue the current response. These and additional reported results suggest how applications can be better designed to both elicit and process optimal input lengths for depression classification. 

**Abstract (ZH)**: 基于语音的抑郁症分类机器学习模型在医疗保健应用中展现出巨大潜力。尽管在抑郁症分类方面已经开展了一定的研究工作，但人们对语音输入长度如何影响模型性能知之甚少。我们利用超过1400小时的人机健康筛查应用中的语音数据，对独立说话者抑郁症分类进行分析。我们基于两个在总体性能方面有所差异的自然语言处理（NLP）系统，研究了响应输入长度与性能之间的关系。

两个系统的测试结果表明，性能取决于自然长度、实际长度以及会话中响应的顺序。这两种系统都存在一个最小长度阈值，但响应饱和阈值有所不同，后者在性能更好的系统中更高。在饱和状态下，继续当前响应不如向说话者提出新的问题更好。这些以及其他报告的结果表明，如何更好地设计应用以获取并处理最佳输入长度，对于抑郁症分类至关重要。 

---
# "Dialogue" vs "Dialog" in NLP and AI research: Statistics from a Confused Discourse 

**Title (ZH)**: “对话” vs “会话”在NLP和AI研究中的差异：一种混淆 discourse 的统计分析 

**Authors**: David Gros  

**Link**: [PDF](https://arxiv.org/pdf/2501.00598)  

**Abstract**: Within computing research, there are two spellings for an increasingly important term - dialogue and dialog. We analyze thousands of research papers to understand this "dialog(ue) debacle". Among publications in top venues that use "dialog(ue)" in the title or abstract, 72% use "dialogue", 24% use "dialog", and 5% use both in the same title and abstract. This split distribution is more common in Computing than any other academic discipline. We investigate trends over ~20 years of NLP/AI research, not finding clear evidence of a shift over time. Author nationality is weakly correlated with spelling choice, but far from explains the mixed use. Many prolific authors publish papers with both spellings. We use several methods (such as syntactic parses and LM embeddings) to study how dialog(ue) context influences spelling, finding limited influence. Combining these results together, we discuss different theories that might explain the dialog(ue) divergence. 

**Abstract (ZH)**: 在计算研究领域中，越来越重要的术语“对话”（dialogue）和“对话”（dialog）出现了两种拼法。我们分析了数千篇研究论文，以探究这一“对话之争（dialogue debacle）”。在顶级学术场所发表的文章中，使用“dialogue”的比例占72%，使用“dialog”的比例占24%，同时在标题或摘要中使用两种拼法的比例为5%。这种拼法的分布差异在计算领域比其他任何学科都更为常见。我们研究了过去约20年的自然语言处理/人工智能研究趋势，但没有发现随着时间推移的明确拼法变化趋势。作者国籍与拼写选择之间的关联性较弱，但远不足以解释这种混合使用现象。许多高产作者在其论文中同时使用这两种拼法。我们使用了多种方法（如句法分析和语言模型嵌入）来研究对话（dialogue）背景如何影响拼写，发现其影响有限。综合以上结果，我们探讨了可能解释对话（dialogue）分歧的不同理论。 

---
# Setting Standards in Turkish NLP: TR-MMLU for Large Language Model Evaluation 

**Title (ZH)**: 土耳其自然语言处理标准设定：TR-MMLU 大型语言模型评估iset系统 

**Authors**: M. Ali Bayram, Ali Arda Fincan, Ahmet Semih G"um"uş, Banu Diri, Savaş Yıldırım, "Oner Aytaş  

**Link**: [PDF](https://arxiv.org/pdf/2501.00593)  

**Abstract**: Language models have made remarkable advancements in understanding and generating human language, achieving notable success across a wide array of applications. However, evaluating these models remains a significant challenge, particularly for resource-limited languages such as Turkish. To address this gap, we introduce the Turkish MMLU (TR-MMLU) benchmark, a comprehensive evaluation framework designed to assess the linguistic and conceptual capabilities of large language models (LLMs) in Turkish. TR-MMLU is constructed from a carefully curated dataset comprising 6200 multiple-choice questions across 62 sections, selected from a pool of 280000 questions spanning 67 disciplines and over 800 topics within the Turkish education system. This benchmark provides a transparent, reproducible, and culturally relevant tool for evaluating model performance. It serves as a standard framework for Turkish NLP research, enabling detailed analyses of LLMs' capabilities in processing Turkish text and fostering the development of more robust and accurate language models. In this study, we evaluate state-of-the-art LLMs on TR-MMLU, providing insights into their strengths and limitations for Turkish-specific tasks. Our findings reveal critical challenges, such as the impact of tokenization and fine-tuning strategies, and highlight areas for improvement in model design. By setting a new standard for evaluating Turkish language models, TR-MMLU aims to inspire future innovations and support the advancement of Turkish NLP research. 

**Abstract (ZH)**: 语言模型在理解和生成人类语言方面取得了显著进展，并在众多应用中取得了显著成功。然而，评估这些模型仍然是一个重大挑战，特别是对于资源有限的语言，如土耳其语。为应对这一挑战，我们介绍了土耳其多模态匹配理解（TR-MMLU）基准，这是一种全面的评估框架，旨在评估大型语言模型（LLMs）在土耳其语中的语言和概念能力。TR-MMLU 基于一个精心筛选的数据集构建，包含来自280,000个问题池中的6200个多选题，这些问题覆盖了67个学科和超过800个教育系统内的主题。该基准提供了一种透明、可再现并且具有文化相关性的工具，用于评估模型性能。它为土耳其自然语言处理（NLP）研究提供了一个标准化框架，使研究人员能够详细分析LLMs在处理土耳其文本方面的能力和促进开发更稳健和准确的语言模型。在本研究中，我们对TR-MMLU上的最新一代LLMs进行了评估，提供了它们在特定土耳其任务方面的强项和局限性的见解。我们的研究结果揭示了一些关键挑战，例如标记化和微调策略的影响，并指出了模型设计中需要改进的领域。通过为评估土耳其语言模型设定新的标准，TR-MMLU旨在激发未来创新并支持土耳其NLP研究的进步。 

---
# Causal Graph Guided Steering of LLM Values via Prompts and Sparse Autoencoders 

**Title (ZH)**: 因果图引导的大语言模型值通过提示和稀疏自编码器操控的研究 

**Authors**: Yipeng Kang, Junqi Wang, Yexin Li, Fangwei Zhong, Xue Feng, Mengmeng Wang, Wenming Tu, Quansen Wang, Hengli Li, Zilong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2501.00581)  

**Abstract**: As large language models (LLMs) become increasingly integrated into critical applications, aligning their behavior with human values presents significant challenges. Current methods, such as Reinforcement Learning from Human Feedback (RLHF), often focus on a limited set of values and can be resource-intensive. Furthermore, the correlation between values has been largely overlooked and remains underutilized. Our framework addresses this limitation by mining a causal graph that elucidates the implicit relationships among various values within the LLMs. Leveraging the causal graph, we implement two lightweight mechanisms for value steering: prompt template steering and Sparse Autoencoder feature steering, and analyze the effects of altering one value dimension on others. Extensive experiments conducted on Gemma-2B-IT and Llama3-8B-IT demonstrate the effectiveness and controllability of our steering methods. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在关键应用程序中越来越广泛的应用，使它们的行为与人类价值观相一致面临着显著的挑战。现有的方法，如从人类反馈强化学习（RLHF），通常关注一组有限的价值观，并且往往资源密集型。此外，价值观之间的关联很大程度上被忽略了，尚未得到充分利用。我们的框架通过挖掘揭示LLMs中各种价值观之间隐含关系的因果图来解决这一限制。利用因果图，我们实现了两种轻量级的价值导向机制：提示模板导向和稀疏自编码器特征导向，并分析了改变一个价值维度对其他维度的影响。在Gemma-2B-IT和Llama3-8B-IT上进行的大量实验表明，我们的导向方法的有效性和可控性。 

---
# KnowRA: Knowledge Retrieval Augmented Method for Document-level Relation Extraction with Comprehensive Reasoning Abilities 

**Title (ZH)**: KnowRA：综合推理能力增强的文档级关系提取知识检索方法 

**Authors**: Chengcheng Mai, Yuxiang Wang, Ziyu Gong, Hanxiang Wang, Yihua Huang  

**Link**: [PDF](https://arxiv.org/pdf/2501.00571)  

**Abstract**: Document-level relation extraction (Doc-RE) aims to extract relations between entities across multiple sentences. Therefore, Doc-RE requires more comprehensive reasoning abilities like humans, involving complex cross-sentence interactions between entities, contexts, and external general knowledge, compared to the sentence-level RE. However, most existing Doc-RE methods focus on optimizing single reasoning ability, but lack the ability to utilize external knowledge for comprehensive reasoning on long documents. To solve these problems, a knowledge retrieval augmented method, named KnowRA, was proposed with comprehensive reasoning to autonomously determine whether to accept external knowledge to assist DocRE. Firstly, we constructed a document graph for semantic encoding and integrated the co-reference resolution model into KnowRA to augment the co-reference reasoning ability. Then, we further expanded the document graph into a document knowledge graph by retrieving the external knowledge base and introduced the axis attention mechanism into KnowRA to improve its common-sense and logical reasoning abilities, respectively. Finally, a knowledge filtering method was presented in the common-sense and co-reference reasoning module to filter out irrelevant knowledge. Extensive experiments conducted on two datasets verified the effectiveness of our method compared to the state-of-the-art baselines. Our code is available at this https URL. 

**Abstract (ZH)**: 文档级别关系提取（Doc-RE）旨在跨多个句子提取实体之间的关系。因此，与句子级别的关系提取相比，Doc-RE 需要更强的推理能力，涉及复杂的句子间交互、上下文以及外部一般知识。然而，现有的大多数 Doc-RE 方法专注于优化单一的推理能力，缺乏利用外部知识进行全面推理的能力，特别是在长文档中。为了解决这些问题，我们提出了一种增强知识检索的方法，命名为KnowRA，该方法通过全面的推理能力自主决定是否接受外部知识以协助Doc-RE。首先，我们构建了一个文档图进行语义编码，并将共指解析模型集成到KnowRA中，以增强其共指推理能力。然后，我们进一步通过检索外部知识库扩展了文档图，将其转换为文档知识图，并引入轴注意力机制来分别提升其常识和逻辑推理能力。最后，我们在常识和共指推理模块中提出了一个知识过滤方法，以过滤掉无关的知识。在两个数据集上进行的广泛实验验证了与最先进的基线方法相比，我们方法的有效性。我们的代码可通过以下链接获取：this https URL。 

---
# An Overview and Discussion on Using Large Language Models for Implementation Generation of Solutions to Open-Ended Problems 

**Title (ZH)**: 关于使用大型语言模型生成开放性问题解决方案的实现概述与讨论 

**Authors**: Hashmath Shaik, Alex Doboli  

**Link**: [PDF](https://arxiv.org/pdf/2501.00562)  

**Abstract**: Large Language Models offer new opportunities to devise automated implementation generation methods that can tackle problem solving activities beyond traditional methods, which require algorithmic specifications and can use only static domain knowledge, like performance metrics and libraries of basic building blocks. Large Language Models could support creating new methods to support problem solving activities for open-ended problems, like problem framing, exploring possible solving approaches, feature elaboration and combination, more advanced implementation assessment, and handling unexpected situations. This report summarized the current work on Large Language Models, including model prompting, Reinforcement Learning, and Retrieval-Augmented Generation. Future research requirements were also discussed. 

**Abstract (ZH)**: 大型语言模型为设计自动化实现生成方法提供了新的机遇，这些方法能够处理传统的基于算法规范的方法无法解决的问题。传统方法仅能使用静态领域知识，如性能指标和基本构建块库，而大型语言模型能够支持解决开放性问题的新方法，包括问题界定、探索可能的解决策略、特征详细描述与组合、更高级的实现评估，以及应对意外情况。本报告总结了当前大型语言模型的研究工作，包括模型提示、强化学习和检索增强生成等方面的内容，并讨论了未来研究的需求。 

---
# Re-evaluating Automatic LLM System Ranking for Alignment with Human Preference 

**Title (ZH)**: 重新评估自动大型语言模型系统排名与人类偏好的一致性 

**Authors**: Mingqi Gao, Yixin Liu, Xinyu Hu, Xiaojun Wan, Jonathan Bragg, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2501.00560)  

**Abstract**: Evaluating and ranking the capabilities of different LLMs is crucial for understanding their performance and alignment with human preferences. Due to the high cost and time-consuming nature of human evaluations, an automatic LLM bencher (i.e., an automatic evaluation framework that aims to rank LLMs based on their alignment with human preferences) is indispensable. An automatic LLM bencher consists of four components: the input set (e.g., a user instruction), the evaluation model (e.g., an LLM), the evaluation type (e.g., pairwise comparison), and the aggregation method (e.g., the ELO rating system). However, previous work has not thoroughly explored how to select these components or how their different combinations influence the results. In this work, through controlled experiments, we provide a series of recommendations on how to choose each component to better automate the evaluation of LLMs. Furthermore, we discovered that when evaluating LLMs with similar performance, the performance of the automatic LLM bencher declines sharply, underscoring the limitations of current benchers and calling for future work. Lastly, we found that the evaluation models' performance at the instance level (e.g., the accuracy of selecting the best output) does not always align with their effectiveness when used as a component of a bencher, highlighting the importance of dedicated system-level evaluation of benchers. 

**Abstract (ZH)**: 评估和排名不同语言模型（LLM）的能力对于理解其性能及其与人类偏好的一致程度至关重要。由于人工评估成本高昂且耗时，因此一个自动化的LLM评估框架（即旨在基于与人类偏好的一致性对LLM进行排名的自动评估框架）是不可或缺的。一个自动化的LLM评估框架包括四个组成部分：输入集（例如，用户指令）、评估模型（例如，一个LLM）、评估类型（例如，配对比较）以及聚合方法（例如，ELO排名系统）。然而，之前的研究所未充分探讨如何选择这些组成部分，以及它们的不同组合如何影响结果。在这项工作中，通过控制实验，我们提供了一系列关于如何选择每个组成部分以更好地自动化LLM评估的建议。此外，我们发现，在评估具有相似性能的LLM时，自动LLM评估框架的性能会急剧下降，这揭示了当前评估框架的局限性，并呼吁未来的研究。最后，我们发现，评估模型在实例级别上的性能（例如，选择最佳输出的准确性）并不总是与其作为评估框架组成部分时的有效性相一致，这强调了对评估框架进行专门的系统级评估的重要性。 

---
# AraSTEM: A Native Arabic Multiple Choice Question Benchmark for Evaluating LLMs Knowledge In STEM Subjects 

**Title (ZH)**: AraSTEM：一个用于评估大型语言模型在STEM学科知识上的阿拉伯语多项选择题基准测试 

**Authors**: Ahmad Mustapha, Hadi Al-Khansa, Hadi Al-Mubasher, Aya Mourad, Ranam Hamoud, Hasan El-Husseini, Marwah Al-Sakkaf, Mariette Awad  

**Link**: [PDF](https://arxiv.org/pdf/2501.00559)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities, not only in generating human-like text, but also in acquiring knowledge. This highlights the need to go beyond the typical Natural Language Processing downstream benchmarks and asses the various aspects of LLMs including knowledge and reasoning. Numerous benchmarks have been developed to evaluate LLMs knowledge, but they predominantly focus on the English language. Given that many LLMs are multilingual, relying solely on benchmarking English knowledge is insufficient. To address this issue, we introduce AraSTEM, a new Arabic multiple-choice question dataset aimed at evaluating LLMs knowledge in STEM subjects. The dataset spans a range of topics at different levels which requires models to demonstrate a deep understanding of scientific Arabic in order to achieve high accuracy. Our findings show that publicly available models of varying sizes struggle with this dataset, and underscores the need for more localized language models. The dataset is freely accessible on Hugging Face. 

**Abstract (ZH)**: 大型语言模型（LLMs）展示了卓越的能力，不仅能够生成类人的文本，还能够获取知识。这突显了超越传统自然语言处理下游基准评估LLMs多个方面的需求，包括知识和推理。已经开发出了许多基准来评估LLMs的知识，但大多数都侧重于英语。鉴于许多LLMs是多语言的，仅依赖于英语知识的基准是不够的。为解决这一问题，我们引入了AraSTEM，这是一个新的阿拉伯语多项选择题数据集，旨在评估LLMs在STEM科目中的知识水平。该数据集涵盖了不同级别和范围的主题，要求模型展示对科学阿拉伯语的深刻理解，以便实现高准确性。我们的研究表明，公开可用的各种规模的模型在这个数据集中遇到困难，突显了需要更多本地化语言模型的需求。该数据集在Hugging Face上免费提供。 

---
# Superposition in Transformers: A Novel Way of Building Mixture of Experts 

**Title (ZH)**: 变压器中的叠加机制：一种新型的专家混合构建方法 

**Authors**: Ayoub Ben Chaliah, Hela Dellagi  

**Link**: [PDF](https://arxiv.org/pdf/2501.00530)  

**Abstract**: Catastrophic forgetting remains a major challenge when adapting large language models (LLMs) to new tasks or domains. Conventional fine-tuning often overwrites existing knowledge, causing performance degradation on original tasks. We introduce Superposition in Transformers, a novel architecture that leverages autoencoders to superimpose the hidden representations of a base model and a fine-tuned model within a shared parameter space. By using B-spline-based blending coefficients and autoencoders that adaptively reconstruct hidden states based on the input data distribution, our method effectively mitigates catastrophic forgetting and enables a new paradigm of "in-model" superposition. This approach preserves original model capabilities while allowing compact domain-specific expertise to be added, and it supports dynamic switching between model states during inference. 

**Abstract (ZH)**: 当将大型语言模型（LLMs）适应到新任务或领域时，灾难性遗忘仍然是一个主要挑战。传统的微调方法往往会覆盖现有知识，导致在原始任务上的性能下降。我们提出了一种新颖的超叠加（Superposition in Transformers）架构，该架构利用自编码器在共享参数空间内叠加基模型和微调模型的隐藏表示。通过使用基于B样条的融合系数，并根据输入数据分布自适应地重构隐藏状态，我们的方法有效地减轻了灾难性遗忘问题，并实现了“在模型内”的超叠加新范式。该方法能够保留原始模型的能力，同时允许添加紧凑的领域特定专业知识，并支持推理过程中模型状态的动态切换。 

---
# Sinhala Transliteration: A Comparative Analysis Between Rule-based and Seq2Seq Approaches 

**Title (ZH)**: 基于规则和Seq2Seq方法的僧伽罗语转写比较分析 

**Authors**: Yomal De Mel, Kasun Wickramasinghe, Nisansa de Silva, Surangika Ranathunga  

**Link**: [PDF](https://arxiv.org/pdf/2501.00529)  

**Abstract**: Due to reasons of convenience and lack of tech literacy, transliteration (i.e., Romanizing native scripts instead of using localization tools) is eminently prevalent in the context of low-resource languages such as Sinhala, which have their own writing script. In this study, our focus is on Romanized Sinhala transliteration. We propose two methods to address this problem: Our baseline is a rule-based method, which is then compared against our second method where we approach the transliteration problem as a sequence-to-sequence task akin to the established Neural Machine Translation (NMT) task. For the latter, we propose a Transformer-based Encode-Decoder solution. We witnessed that the Transformer-based method could grab many ad-hoc patterns within the Romanized scripts compared to the rule-based method. The code base associated with this paper is available on GitHub - this https URL 

**Abstract (ZH)**: 由于便利性和缺乏技术素养的原因，在资源有限的语言如僧伽罗语（Sinhala）中，直接罗马化原生书写系统而非使用本地化工具的情况极为常见。在本文的研究中，我们主要关注罗马化的僧伽罗语转写问题。为此，我们提出两种方法来解决这一问题：我们的基线方法是一个基于规则的方法，然后我们将其与第二种方法进行了比较，即我们将转写问题视为类似于已有的神经机器翻译（Neural Machine Translation, NMT）任务的序列到序列（Sequence-to-Sequence, Seq2Seq）任务。对于后者，我们提出了一种基于Transformer的编码-解码（Encoder-Decoder）解决方案。我们发现基于Transformer的方法能够比基于规则的方法更好地捕捉罗马化脚本中的许多即兴模式。与本文相关的代码库已发布在GitHub上，访问链接为：this https URL 

---
# TinyHelen's First Curriculum: Training and Evaluating Tiny Language Models in a Simpler Language Environment 

**Title (ZH)**: TinyHelen 的首个课程：在更简单的语言环境中训练和评估小微企业型语言模型 

**Authors**: Ke Yang, Volodymyr Kindratenko, ChengXiang Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2501.00522)  

**Abstract**: Training language models (LMs) and their application agents is increasingly costly due to large datasets and models, making test failures difficult to bear. Simplified language environments serve as primordial training and testing grounds, retaining essential commonsense and communication skills but in a more digestible form, potentially enhancing the learning efficiency of LMs, and thus reducing the required model size and data volume for effective training and evaluation. In these simplified language environments, workable strategies for small models, datasets, and agents may be adaptable to larger models, datasets, and agents in complex language environments.
To create such environments, we focus on two aspects: i) minimizing language dataset noise and complexity, and ii) preserving the essential text distribution characteristics. Unlike previous methods, we propose a pipeline to refine text data by eliminating noise, minimizing vocabulary, and maintaining genre-specific patterns (e.g., for books, conversation, code, etc.). Implementing this pipeline with large LMs, we have created a leaner suite of LM training and evaluation datasets: 71M Leaner-Pretrain, 7M Leaner-Instruct, Leaner-Glue for assessing linguistic proficiency, and Leaner-Eval for testing instruction-following ability.
Our experiments show that leaner pre-training boosts LM learning efficiency. Tiny LMs trained on these datasets outperform those trained on original datasets in instruction-following across different language granularity levels. Moreover, the Leaner-Pretrain dataset's alignment with conventional large LM training sets enables resource-optimized analysis of how learning objectives, model architectures, and training techniques impact performance on language modeling and downstream tasks. Our code and datasets are available at this https URL. 

**Abstract (ZH)**: 训练语言模型（LMs）及其应用代理的成本因大型数据集和模型而不断增加，这使得测试失败更加难以承受。简化语言环境作为初级训练和测试场所，保留了基本的常识和沟通技巧，但以更易于理解的形式呈现，这可能提高LM的学习效率，从而减少有效训练和评估所需的模型大小和数据量。在这些简化语言环境中，适用于小型模型、数据集和代理的工作策略可能适用于复杂语言环境中较大的模型、数据集和代理。

为创建这样的环境，我们重点关注两个方面：i）最小化语言数据集的噪音和复杂性，以及ii）保留关键的文本分布特征。不同于以往的方法，我们提出了一种管线方法，通过消除噪音、最小化词汇量并保持特定体裁（如书籍、对话、代码等）的模式来精炼文本数据。通过使用大型LM实现这一管线方法，我们创建了一套更精简的LM训练和评估数据集：71M Leaner-Pretrain，7M Leaner-Instruct，Leaner-Glue 以评估语言能力，以及Leaner-Eval 以测试指令执行能力。

我们的实验表明，较精简的预训练能够提高LM的学习效率。使用这些数据集训练的小型LM在不同语言粒度级别上的指令执行表现优于使用原始数据集训练的LM。此外，Leaner-Pretrain数据集与传统大型LM训练集的对齐使我们能够优化资源，分析学习目标、模型架构和训练技术对语言建模和下游任务性能的影响。我们的代码和数据集可从以下链接访问：this https URL。 

---
# Fotheidil: an Automatic Transcription System for the Irish Language 

**Title (ZH)**: Fotheidil：盖尔语自动转录系统 

**Authors**: Liam Lonergan, Ibon Saratxaga, John Sloan, Oscar Maharog, Mengjie Qian, Neasa Ní Chiaráin, Christer Gobl, Ailbhe Ní Chasaide  

**Link**: [PDF](https://arxiv.org/pdf/2501.00509)  

**Abstract**: This paper sets out the first web-based transcription system for the Irish language - Fotheidil, a system that utilises speech-related AI technologies as part of the ABAIR initiative. The system includes both off-the-shelf pre-trained voice activity detection and speaker diarisation models and models trained specifically for Irish automatic speech recognition and capitalisation and punctuation restoration. Semi-supervised learning is explored to improve the acoustic model of a modular TDNN-HMM ASR system, yielding substantial improvements for out-of-domain test sets and dialects that are underrepresented in the supervised training set. A novel approach to capitalisation and punctuation restoration involving sequence-to-sequence models is compared with the conventional approach using a classification model. Experimental results show here also substantial improvements in performance. The system will be made freely available for public use, and represents an important resource to researchers and others who transcribe Irish language materials. Human-corrected transcriptions will be collected and included in the training dataset as the system is used, which should lead to incremental improvements to the ASR model in a cyclical, community-driven fashion. 

**Abstract (ZH)**: 本文介绍了第一项基于网络的手语转录系统——Fotheidil，该系统作为ABAIR计划的一部分，利用了与语音相关的AI技术。该系统包括现成的预训练语音活动检测和说话人辨认模型，以及专门针对爱尔兰自动语音识别和大小写及标点符号恢复的模型。通过探索半监督学习来改进模块化TDNN-HMM自动语音识别系统的声学模型，从而在测试集和未被监督训练数据集欠代表的方言上取得了显著的进步。本文还提出了一种新的大小写及标点符号恢复方法，利用序列到序列模型进行实验，并与使用分类模型的惯例方法进行了比较，实验结果表明性能也取得了显著提升。该系统将免费提供给公众使用，代表了研究人员及其他手语转录爱尔兰语言材料的人员的重要资源。随着系统的使用，将收集到的人工校正转录并纳入训练数据集，从而以循序渐进的方式、社区驱动的方式不断改进ASR模型。 

---
# Enhancing LLM Reasoning with Multi-Path Collaborative Reactive and Reflection agents 

**Title (ZH)**: 使用多路径协作反应与反思代理增强生成式预训练语言模型的推理能力 

**Authors**: Chengbo He, Bochao Zou, Xin Li, Jiansheng Chen, Junliang Xing, Huimin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2501.00430)  

**Abstract**: Agents have demonstrated their potential in scientific reasoning tasks through large language models. However, they often face challenges such as insufficient accuracy and degeneration of thought when handling complex reasoning tasks, which impede their performance. To overcome these issues, we propose the Reactive and Reflection agents with Multi-Path Reasoning (RR-MP) Framework, aimed at enhancing the reasoning capabilities of LLMs. Our approach improves scientific reasoning accuracy by employing a multi-path reasoning mechanism where each path consists of a reactive agent and a reflection agent that collaborate to prevent degeneration of thought inherent in single-agent reliance. Additionally, the RR-MP framework does not require additional training; it utilizes multiple dialogue instances for each reasoning path and a separate summarizer to consolidate insights from all paths. This design integrates diverse perspectives and strengthens reasoning across each path. We conducted zero-shot and few-shot evaluations on tasks involving moral scenarios, college-level physics, and mathematics. Experimental results demonstrate that our method outperforms baseline approaches, highlighting the effectiveness and advantages of the RR-MP framework in managing complex scientific reasoning tasks. 

**Abstract (ZH)**: 大型语言模型已经在科学推理任务中展示了其潜力，但当处理复杂推理任务时，代理常常面临准确性不足和思维退化等挑战，这限制了它们的表现。为了解决这些问题，我们提出了一种反应性和反思性多路径推理框架（RR-MP框架），旨在增强大模型的推理能力。我们的方法通过采用多路径推理机制来提高科学推理的准确性，其中每条路径由一个反应性代理和一个反思性代理组成，二者协作以防止单代理依赖性所固有的思维退化。此外，RR-MP框架不需要额外的训练，它利用每条推理路径中的多个对话实例以及一个单独的总结器来整合所有路径所获得的洞见。这种设计整合了多角度的观点，并增强了每条路径的推理能力。我们在涉及道德场景、大学物理和数学的任务上进行了零样本和少样本评估。实验结果表明，我们的方法优于基线方法，突显了RR-MP框架在管理复杂科学推理任务方面的有效性和优势。 

---
# Whisper Turns Stronger: Augmenting Wav2Vec 2.0 for Superior ASR in Low-Resource Languages 

**Title (ZH)**: Whisper 越来越强大：增强 Wav2Vec 2.0 以在低资源语言中的 ASR 表现更优 

**Authors**: Or Haim Anidjar, Revital Marbel, Roi Yozevitch  

**Link**: [PDF](https://arxiv.org/pdf/2501.00425)  

**Abstract**: Approaching Speech-to-Text and Automatic Speech Recognition problems in low-resource languages is notoriously challenging due to the scarcity of validated datasets and the diversity of dialects. Arabic, Russian, and Portuguese exemplify these difficulties, being low-resource languages due to the many dialects of these languages across different continents worldwide. Moreover, the variety of accents and pronunciations of such languages complicate ASR models' success. With the increasing popularity of Deep Learning and Transformers, acoustic models like the renowned Wav2Vec2 have achieved superior performance in the Speech Recognition field compared to state-of-the-art approaches. However, despite Wav2Vec2's improved efficiency over traditional methods, its performance significantly declines for under-represented languages, even though it requires significantly less labeled data. This paper introduces an end-to-end framework that enhances ASR systems fine-tuned on Wav2Vec2 through data augmentation techniques. To validate our framework's effectiveness, we conducted a detailed experimental evaluation using three datasets from Mozilla's Common Voice project in Arabic, Russian, and Portuguese. Additionally, the framework presented in this paper demonstrates robustness to different diacritics. Ultimately, our approach outperforms two previous baseline models, which are the pre-trained Wav2Vec2 and the well-known Whisper ASR model, resulting in an average relative improvement of 33.9\% in Word Error Rate and a 53.2\% relative improvement in Character Error Rate. 

**Abstract (ZH)**: 在资源稀缺语言中的语音转文本和自动语音识别问题因其验证数据的稀缺性和方言的多样性而格外具有挑战性。阿拉伯语、俄语和葡萄牙语就是这些困难的代表，这些语言因在同一洲际的不同大陆上存在许多方言而成为资源稀缺语言。此外，这些语言的音调和发音多样性也使自动语音识别（ASR）模型的成功变得复杂。随着深度学习和变换器的流行，像Wav2Vec2这样声学模型在语音识别领域的表现优于最先进的方法。然而，尽管Wav2Vec2相比传统方法在效率上有显著提高，但其在少数代表性语言上的表现显著下降，尽管它只需要更少的标注数据。本文介绍了一种端到端框架，通过数据增强技术增强了基于Wav2Vec2微调的ASR系统。为了验证我们框架的有效性，我们在Mozilla Common Voice项目中的阿拉伯语、俄语和葡萄牙语三个数据集上进行了详细的实验评估。此外，本文中提出的框架对不同的书写符号具有较强的鲁棒性。最终，我们的方法在单词错误率（Word Error Rate, WER）上相比前两个基线模型平均提高了33.9%，字符错误率（Character Error Rate, CER）上提高了53.2%。 

---
# Trajectories of Change: Approaches for Tracking Knowledge Evolution 

**Title (ZH)**: 变化轨迹：追踪知识演变的方法 

**Authors**: Raphael Schlattmann, Malte Vogl  

**Link**: [PDF](https://arxiv.org/pdf/2501.00391)  

**Abstract**: We explore local vs. global evolution of knowledge systems through the framework of socio-epistemic networks (SEN), applying two complementary methods to a corpus of scientific texts. The framework comprises three interconnected layers-social, semiotic (material), and semantic-proposing a multilayered approach to understanding structural developments of knowledge. To analyse diachronic changes on the semantic layer, we first use information-theoretic measures based on relative entropy to detect semantic shifts, assess their significance, and identify key driving features. Second, variations in document embedding densities reveal changes in semantic neighbourhoods, tracking how concentration of similar documents increase, remain stable, or disperse. This enables us to trace document trajectories based on content (topics) or metadata (authorship, institution). Case studies of Joseph Silk and Hans-Jürgen Treder illustrate how individual scholar's work aligns with broader disciplinary shifts in general relativity and gravitation research, demonstrating the applications, limitations, and further potential of this approach. 

**Abstract (ZH)**: 我们通过社会认知网络（SEN）框架探索知识系统的局部与全局演化，应用两种互补的方法对一组科学文本进行分析。该框架包括三个相互关联的层次：社会层、符号（物质）层和语义层，提出了一种多层次的方法来理解知识结构的演变。为了分析语义层上的历时变化，我们首先使用基于相对熵的信息理论度量来检测语义变化、评估其重要性并识别关键驱动特征。其次，文档嵌入密度的变化揭示了语义临近区的变化，追踪了相似文档集中度的增加、稳定或分散情况。这使我们能够基于内容（主题）或元数据（作者身份、机构）追踪文档的演变轨迹。作为案例研究，约瑟夫·席尔 (Joseph Silk) 和汉斯-ю尔根·特德尔 (Hans-Jürgen Treder) 的工作展示了个人学者的研究如何与广义相对论和引力研究领域的更广泛转变相吻合，证明了该方法的应用价值、局限性和进一步的发展潜力。 

---
# RAG-Instruct: Boosting LLMs with Diverse Retrieval-Augmented Instructions 

**Title (ZH)**: RAG-Instruct：通过多样化检索增强指令提升大型语言模型 

**Authors**: Wanlong Liu, Junying Chen, Ke Ji, Li Zhou, Wenyu Chen, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.00353)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a key paradigm for enhancing large language models (LLMs) by incorporating external knowledge. However, current RAG methods face two limitations: (1) they only cover limited RAG scenarios. (2) They suffer from limited task diversity due to the lack of a general RAG dataset. To address these limitations, we propose RAG-Instruct, a general method for synthesizing diverse and high-quality RAG instruction data based on any source corpus. Our approach leverages (1) five RAG paradigms, which encompass diverse query-document relationships, and (2) instruction simulation, which enhances instruction diversity and quality by utilizing the strengths of existing instruction datasets. Using this method, we construct a 40K instruction dataset from Wikipedia, comprehensively covering diverse RAG scenarios and tasks. Experiments demonstrate that RAG-Instruct effectively enhances LLMs' RAG capabilities, achieving strong zero-shot performance and significantly outperforming various RAG baselines across a diverse set of tasks. RAG-Instruct is publicly available at this https URL. 

**Abstract (ZH)**: 检索增强生成（RAG）已成为通过引入外部知识增强大规模语言模型（LLMs）的关键范式。然而，当前的RAG方法面临着两个局限性：（1）它们仅涵盖有限的RAG场景；（2）由于缺乏通用的RAG数据集，它们在任务多样性方面也受到限制。为解决这些局限性，我们提出了RAG-Instruct，这是一种基于任何来源语料库生成多样且高质量RAG指令数据的通用方法。我们的方法利用了以下两点：（1）五种RAG范式，涵盖了多样化的查询-文档关系；（2）指令模拟，通过利用现有指令数据集的优势来增强指令多样化和质量。利用这种方法，我们从Wikipedia中构建了一个包含4万个指令的语料库，全面涵盖了多样化的RAG场景和任务。实验表明，RAG-Instruct有效地增强了LLMs的RAG能力，实现了出色的零样本性能，并且在一系列多样化任务中显著优于各种RAG基准。RAG-Instruct已在以下链接公开发布：[此网址](this https URL)。 

---
# Chunk-Distilled Language Modeling 

**Title (ZH)**: 块提炼语言模型 

**Authors**: Yanhong Li, Karen Livescu, Jiawei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2501.00343)  

**Abstract**: We introduce Chunk-Distilled Language Modeling (CD-LM), an approach to text generation that addresses two challenges in current large language models (LLMs): the inefficiency of token-level generation, and the difficulty of adapting to new data and knowledge. Our method combines deep network-based LLMs with a straightforward retrieval module, which allows the generation of multi-token text chunks at a single decoding step. Our retrieval framework enables flexible construction of model- or domain-specific datastores, either leveraging the internal knowledge of existing models, or incorporating expert insights from human-annotated corpora. This adaptability allows for enhanced control over the language model's distribution without necessitating additional training. We present the CD-LM formulation along with performance metrics demonstrating its ability to improve language model performance and efficiency across a diverse set of downstream tasks. Code and data will be made publicly available. 

**Abstract (ZH)**: 我们介绍了片段提炼语言模型（Chunk-Distilled Language Model, CD-LM），这是一种解决当前大型语言模型（Large Language Models, LLMs）中两个挑战的方法：按令牌级生成的低效性，以及适应新数据和知识的困难。我们的方法结合了基于深层神经网络的LLMs与一个简单的检索模块，使得在一次解码步骤中即可生成多令牌文本片段。我们的检索框架允许灵活构建模型特定或领域特定的数据存储，既可以利用现有模型的内部知识，也可以结合人类注释语料库中的专家见解。这种适应性允许在不需额外训练的情况下增强对语言模型分布的控制。我们介绍了CD-LM的建模方法，并提供了性能指标，证明了其能够在各种下游任务中提高语言模型的性能和效率。相关代码和数据将公开发布。 

---
# Rethinking Layer Removal: Preserving Critical Components with Task-Aware Singular Value Decomposition 

**Title (ZH)**: 重新思考层删除：基于任务意识的奇异值分解保留关键组件 

**Authors**: Kainan Liu, Yong Zhang, Ning Cheng, Zhitao Li, Shaojun Wang, Jing Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2501.00339)  

**Abstract**: Layer removal has emerged as a promising approach for compressing large language models (LLMs) by leveraging redundancy within layers to reduce model size and accelerate inference. However, this technique often compromises internal consistency, leading to performance degradation and instability, with varying impacts across different model architectures. In this work, we propose Taco-SVD, a task-aware framework that retains task-critical singular value directions, preserving internal consistency while enabling efficient compression. Unlike direct layer removal, Taco-SVD preserves task-critical transformations to mitigate performance degradation. By leveraging gradient-based attribution methods, Taco-SVD aligns singular values with downstream task objectives. Extensive evaluations demonstrate that Taco-SVD outperforms existing methods in perplexity and task performance across different architectures while ensuring minimal computational overhead. 

**Abstract (ZH)**: 层删除作为一种利用层内冗余来减少模型大小并加速推理的潜在方法，已经成为了压缩大型语言模型（LLMs）的一种有前景的途径。然而，这种技术往往会损害内部一致性，导致性能下降和不稳定，而且不同模型架构的影响各不相同。在本工作中，我们提出了Taco-SVD，这是一种任务感知的框架，旨在保留任务关键的奇异值方向，从而保持内部一致性同时实现高效的压缩。与直接删除层不同，Taco-SVD 保留了任务关键的变换，以减轻性能下降的影响。通过利用基于梯度的归因方法，Taco-SVD 使奇异值与下游任务目标对齐。广泛的实验结果表明，在不同架构下，Taco-SVD 在困惑度和任务性能方面优于现有方法，同时确保了最小的计算开销。 

---
# Loss-Aware Curriculum Learning for Chinese Grammatical Error Correction 

**Title (ZH)**: 面向损失的学习层次训练对中国语法错误纠正的应用 

**Authors**: Ding Zhang, Yangning Li, Lichen Bai, Hao Zhang, Yinghui Li, Haiye Lin, Hai-Tao Zheng, Xin Su, Zifei Shan  

**Link**: [PDF](https://arxiv.org/pdf/2501.00334)  

**Abstract**: Chinese grammatical error correction (CGEC) aims to detect and correct errors in the input Chinese sentences. Recently, Pre-trained Language Models (PLMS) have been employed to improve the performance. However, current approaches ignore that correction difficulty varies across different instances and treat these samples equally, enhancing the challenge of model learning. To address this problem, we propose a multi-granularity Curriculum Learning (CL) framework. Specifically, we first calculate the correction difficulty of these samples and feed them into the model from easy to hard batch by batch. Then Instance-Level CL is employed to help the model optimize in the appropriate direction automatically by regulating the loss function. Extensive experimental results and comprehensive analyses of various datasets prove the effectiveness of our method. 

**Abstract (ZH)**: 中文翻译如下，符合学术规范：

中文语法错误修正（CGEC）旨在检测并修正输入的中文句子中的错误。近年来，预训练语言模型（PLMs）已被用于提高性能。然而，当前的方法忽视了修正难度在不同实例之间存在差异，并且对待这些样本一视同仁，增加了模型学习的难度。为解决这一问题，我们提出了一种多粒度的课程学习（CL）框架。具体而言，我们首先计算这些样本的修正难度，并按照从易到难的顺序逐批馈送给模型。然后采用实例级别（Instance-Level）的课程学习，通过调节损失函数帮助模型自动优化到合适的方向。广泛的数据集上的实验结果和综合分析证明了我们方法的有效性。 

---
# MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation 

**Title (ZH)**: MAIN-RAG：多agent过滤检索增强生成 

**Authors**: Chia-Yuan Chang, Zhimeng Jiang, Vineeth Rakesh, Menghai Pan, Chin-Chia Michael Yeh, Guanchu Wang, Mingzhi Hu, Zhichao Xu, Yan Zheng, Mahashweta Das, Na Zou  

**Link**: [PDF](https://arxiv.org/pdf/2501.00332)  

**Abstract**: Large Language Models (LLMs) are becoming essential tools for various natural language processing tasks but often suffer from generating outdated or incorrect information. Retrieval-Augmented Generation (RAG) addresses this issue by incorporating external, real-time information retrieval to ground LLM responses. However, the existing RAG systems frequently struggle with the quality of retrieval documents, as irrelevant or noisy documents degrade performance, increase computational overhead, and undermine response reliability. To tackle this problem, we propose Multi-Agent Filtering Retrieval-Augmented Generation (MAIN-RAG), a training-free RAG framework that leverages multiple LLM agents to collaboratively filter and score retrieved documents. Specifically, MAIN-RAG introduces an adaptive filtering mechanism that dynamically adjusts the relevance filtering threshold based on score distributions, effectively minimizing noise while maintaining high recall of relevant documents. The proposed approach leverages inter-agent consensus to ensure robust document selection without requiring additional training data or fine-tuning. Experimental results across four QA benchmarks demonstrate that MAIN-RAG consistently outperforms traditional RAG approaches, achieving a 2-11% improvement in answer accuracy while reducing the number of irrelevant retrieved documents. Quantitative analysis further reveals that our approach achieves superior response consistency and answer accuracy over baseline methods, offering a competitive and practical alternative to training-based solutions. 

**Abstract (ZH)**: 大型语言模型（LLMs）已成为各种自然语言处理任务的关键工具，但往往会产生过时或错误的信息。检索增强生成（RAG）通过结合外部的实时信息检索来增强LLM的回答，解决了这一问题。然而，现有的RAG系统在检索文档的质量上经常遇到困难，因为无关或噪声文档会降低性能、增加计算成本，并削弱回答的可靠性。为了解决这一问题，我们提出了一种无需训练的多代理过滤检索增强生成（MAIN-RAG）框架，利用多个LLM代理协作过滤和评分检索到的文档。具体而言，MAIN-RAG引入了一种自适应的过滤机制，根据评分分布动态调整相关性过滤阈值，有效减少噪声同时保持高相关文档的召回率。该方法利用代理之间的共识来确保稳健的文档选择，无需额外的训练数据或微调。在四个问答基准测试上的实验结果表明，MAIN-RAG始终优于传统的RAG方法，实现了2-11%的答案准确性的提升，并且减少了无关的检索文档数量。定量分析进一步表明，我们的方法在响应一致性及答案准确率方面优于基线方法，提供了一种具有竞争力且实用的训练基线解决方案的替代方案。 

---
# Exploring the Implicit Semantic Ability of Multimodal Large Language Models: A Pilot Study on Entity Set Expansion 

**Title (ZH)**: 探索多模态大语言模型的隐含语义能力：关于实体集扩展的初步研究 

**Authors**: Hebin Wang, Yangning Li, Yinghui Li, Hai-Tao Zheng, Wenhao Jiang, Hong-Gee Kim  

**Link**: [PDF](https://arxiv.org/pdf/2501.00330)  

**Abstract**: The rapid development of multimodal large language models (MLLMs) has brought significant improvements to a wide range of tasks in real-world applications. However, LLMs still exhibit certain limitations in extracting implicit semantic information. In this paper, we apply MLLMs to the Multi-modal Entity Set Expansion (MESE) task, which aims to expand a handful of seed entities with new entities belonging to the same semantic class, and multi-modal information is provided with each entity. We explore the capabilities of MLLMs to understand implicit semantic information at the entity-level granularity through the MESE task, introducing a listwise ranking method LUSAR that maps local scores to global rankings. Our LUSAR demonstrates significant improvements in MLLM's performance on the MESE task, marking the first use of generative MLLM for ESE tasks and extending the applicability of listwise ranking. 

**Abstract (ZH)**: 大规模多模态语言模型（MLLMs）的迅速发展在实际应用中的诸多任务中带来了显著改进。然而，LLMs 在提取隐含语义信息方面仍然存在一定的局限性。本文将MLLMs 应用于多模态实体集扩展（MESE）任务，该任务旨在在每个实体提供多模态信息的前提下，扩展少量的种子实体到同一语义类别的新实体。我们通过MESE任务探索MLLMs 在实体级别理解隐含语义信息的能力，并引入了一种列表级排名方法LUSAR，将局部评分映射到全局排名。我们的LUSAR 在MESE任务中显著提升了MLLMs 的性能，标志着首次使用生成型MLLMs 进行实体集扩展任务，并扩展了列表级排名的应用范围。 

---
# MapEval: A Map-Based Evaluation of Geo-Spatial Reasoning in Foundation Models 

**Title (ZH)**: MapEval：基于地图的地理空间推理评估方法 

**Authors**: Mahir Labib Dihan, Md Tanvir Hassan, Md Tanvir Parvez, Md Hasebul Hasan, Md Almash Alam, Muhammad Aamir Cheema, Mohammed Eunus Ali, Md Rizwan Parvez  

**Link**: [PDF](https://arxiv.org/pdf/2501.00316)  

**Abstract**: Recent advancements in foundation models have enhanced AI systems' capabilities in autonomous tool usage and reasoning. However, their ability in location or map-based reasoning - which improves daily life by optimizing navigation, facilitating resource discovery, and streamlining logistics - has not been systematically studied. To bridge this gap, we introduce MapEval, a benchmark designed to assess diverse and complex map-based user queries with geo-spatial reasoning. MapEval features three task types (textual, API-based, and visual) that require collecting world information via map tools, processing heterogeneous geo-spatial contexts (e.g., named entities, travel distances, user reviews or ratings, images), and compositional reasoning, which all state-of-the-art foundation models find challenging. Comprising 700 unique multiple-choice questions about locations across 180 cities and 54 countries, MapEval evaluates foundation models' ability to handle spatial relationships, map infographics, travel planning, and navigation challenges. Using MapEval, we conducted a comprehensive evaluation of 28 prominent foundation models. While no single model excelled across all tasks, Claude-3.5-Sonnet, GPT-4o, and Gemini-1.5-Pro achieved competitive performance overall. However, substantial performance gaps emerged, particularly in MapEval, where agents with Claude-3.5-Sonnet outperformed GPT-4o and Gemini-1.5-Pro by 16% and 21%, respectively, and the gaps became even more amplified when compared to open-source LLMs. Our detailed analyses provide insights into the strengths and weaknesses of current models, though all models still fall short of human performance by more than 20% on average, struggling with complex map images and rigorous geo-spatial reasoning. This gap highlights MapEval's critical role in advancing general-purpose foundation models with stronger geo-spatial understanding. 

**Abstract (ZH)**: 近年来，基础模型的最新进展提升了AI系统在自主工具使用和推理方面的能力。然而，这些模型在基于地理位置或地图的推理方面的能力——这种推理能够通过优化导航、促进资源发现和简化物流来改善日常生活——尚未进行系统的研究。为了填补这一空白，我们引入了MapEval这一基准测试，旨在评估多样且复杂的基于地理位置的用户查询。MapEval包含三种任务类型（文本、基于API和视觉），要求通过地图工具收集世界信息、处理异构地理位置上下文（例如命名实体、旅行距离、用户评论或评分、图像）和组合推理，这些都是最先进的基础模型面临的一大挑战。

MapEval共有180个城市和54个国家的700个独特的选择题目，评估基础模型处理空间关系、地图信息图、旅行规划和导航挑战的能力。使用MapEval，我们对28个知名的基础模型进行了全面评估。尽管没有单一模型在所有任务中表现优异，但Claude-3.5-Sonnet、GPT-4o 和 Gemini-1.5-Pro 在总体上取得了竞争力的表现。然而，在MapEval中，使用Claude-3.5-Sonnet的代理比GPT-4o和Gemini-1.5-Pro分别高出16%和21%。当与开源的大规模语言模型（LLM）相比时，这种差距更是进一步扩大。我们的详细分析提供了对当前模型强项和弱项的见解，尽管所有模型在平均表现上仍低于人类表现20%以上，特别是在复杂地图图像和严格的地理位置推理方面存在困难。这一差距强调了MapEval在推动具有更强地理位置理解的基础模型方面的重要作用。 

---
# LLM-Rubric: A Multidimensional, Calibrated Approach to Automated Evaluation of Natural Language Texts 

**Title (ZH)**: LLM-评分标准：一种多维度、校准化的自然语言文本自动化评估方法 

**Authors**: Helia Hashemi, Jason Eisner, Corby Rosset, Benjamin Van Durme, Chris Kedzie  

**Link**: [PDF](https://arxiv.org/pdf/2501.00274)  

**Abstract**: This paper introduces a framework for the automated evaluation of natural language texts. A manually constructed rubric describes how to assess multiple dimensions of interest. To evaluate a text, a large language model (LLM) is prompted with each rubric question and produces a distribution over potential responses. The LLM predictions often fail to agree well with human judges -- indeed, the humans do not fully agree with one another. However, the multiple LLM distributions can be $\textit{combined}$ to $\textit{predict}$ each human judge's annotations on all questions, including a summary question that assesses overall quality or relevance. LLM-Rubric accomplishes this by training a small feed-forward neural network that includes both judge-specific and judge-independent parameters. When evaluating dialogue systems in a human-AI information-seeking task, we find that LLM-Rubric with 9 questions (assessing dimensions such as naturalness, conciseness, and citation quality) predicts human judges' assessment of overall user satisfaction, on a scale of 1--4, with RMS error $< 0.5$, a $2\times$ improvement over the uncalibrated baseline. 

**Abstract (ZH)**: 本文介绍了一个自然语言文本自动化评估的框架。一个手工构建的评价标准描述了如何评估多个感兴趣维度的方法。为了评估一篇文本，大型语言模型（LLM）被每次被提示每个评价标准的问题，并生成潜在回应的概率分布。LLM的预测往往不能很好地与人类评委的评判匹配——实际上，人类评委之间也并不完全一致。然而，多个LLM的分布可以通过某种方式**结合**，用于**预测**每位人类评委在所有问题上的标注，包括一个评估整体质量和相关性的总结问题。通过训练一个包含评委特定参数和通用参数的小型前馈神经网络，LLM-Rubric实现了这一目标。在一项评估对话系统的实验中，当任务涉及人类与AI的信息寻求对话时，使用包含9个问题（评估自然性、简洁性和引文质量等维度）的LLM-Rubric，可以预测人类评委对整体用户满意度的评价（采用1到4的评分标准），均方根误差小于0.5，相较于未校准的基线模型，性能提高了两倍。 

---
# Echoes in AI: Quantifying Lack of Plot Diversity in LLM Outputs 

**Title (ZH)**: 《AI回声：量化LLM输出中情节多样性不足的情况》 

**Authors**: Weijia Xu, Nebojsa Jojic, Sudha Rao, Chris Brockett, Bill Dolan  

**Link**: [PDF](https://arxiv.org/pdf/2501.00273)  

**Abstract**: With rapid advances in large language models (LLMs), there has been an increasing application of LLMs in creative content ideation and generation. A critical question emerges: can current LLMs provide ideas that are diverse enough to truly bolster the collective creativity? We examine two state-of-the-art LLMs, GPT-4 and LLaMA-3, on story generation and discover that LLM-generated stories often consist of plot elements that are echoed across a number of generations. To quantify this phenomenon, we introduce the Sui Generis score, which estimates how unlikely a plot element is to appear in alternative storylines generated by the same LLM. Evaluating on 100 short stories, we find that LLM-generated stories often contain combinations of idiosyncratic plot elements echoed frequently across generations, while the original human-written stories are rarely recreated or even echoed in pieces. Moreover, our human evaluation shows that the ranking of Sui Generis scores among story segments correlates moderately with human judgment of surprise level, even though score computation is completely automatic without relying on human judgment. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）的快速发展，LLMs 在创意思维和内容生成方面的应用日益增多。一个关键的问题出现了：当前的LLMs 能否提供足够多样的想法，以真正增强集体创造力？我们考察了两个最先进的LLMs（GPT-4 和 LLaMA-3）在故事生成中的表现，并发现LLM生成的故事通常包含在多个生成过程中反复出现的情节元素。为了量化这一现象，我们引入了“独创性”（Sui Generis）得分，该得分估计特定情节元素在由相同LLM生成的不同故事情节中出现的概率有多低。在100个短故事的评估中，我们发现LLM生成的故事经常包含频繁出现在多个生成过程中的独特情节元素的组合，而原本人撰写的故事情节很少被再现或仅被部分重现。此外，我们的主观评价表明，“独创性”得分在故事情节片段间的排名与人类对意外程度的判断存在中等程度的相关性，尽管得分计算完全是自动化的，不依赖于人类判断。 

---
# A review of faithfulness metrics for hallucination assessment in Large Language Models 

**Title (ZH)**: 大型语言模型中幻觉评估忠实度度量综述 

**Authors**: Ben Malin, Tatiana Kalganova, Nikoloas Boulgouris  

**Link**: [PDF](https://arxiv.org/pdf/2501.00269)  

**Abstract**: This review examines the means with which faithfulness has been evaluated across open-ended summarization, question-answering and machine translation tasks. We find that the use of LLMs as a faithfulness evaluator is commonly the metric that is most highly correlated with human judgement. The means with which other studies have mitigated hallucinations is discussed, with both retrieval augmented generation (RAG) and prompting framework approaches having been linked with superior faithfulness, whilst other recommendations for mitigation are provided. Research into faithfulness is integral to the continued widespread use of LLMs, as unfaithful responses can pose major risks to many areas whereby LLMs would otherwise be suitable. Furthermore, evaluating open-ended generation provides a more comprehensive measure of LLM performance than commonly used multiple-choice benchmarking, which can help in advancing the trust that can be placed within LLMs. 

**Abstract (ZH)**: 本文回顾了在开放式摘要、问答和机器翻译任务中对忠实性进行评估的方法。我们发现，使用大规模语言模型（LLM）作为忠实性评估工具通常是与人类判断最高度相关的指标。文中还讨论了其他研究如何减轻幻觉的方法，包括检索增强生成（RAG）和提示框架方法已被证明能提高忠实性，同时提供了其他减轻幻觉的建议。对忠实性的研究对于LLM的持续广泛应用至关重要，因为不忠实的回答可能会在许多原本适合应用LLM的领域带来重大风险。此外，评估开放式生成提供了比常用的多项选择基准测试更为全面的LLM性能衡量标准，有助于增强对LLM的信任度。 

---
# EQUATOR: A Deterministic Framework for Evaluating LLM Reasoning with Open-Ended Questions. # v1.0.0-beta 

**Title (ZH)**: EQUATOR：一种评估大模型在解答开放性问题时推理能力的确定性框架。# v1.0.0-beta 

**Authors**: Raymond Bernard, Shaina Raza, Subhabrata Das, Rahul Murugan  

**Link**: [PDF](https://arxiv.org/pdf/2501.00257)  

**Abstract**: Despite the remarkable coherence of Large Language Models (LLMs), existing evaluation methods often suffer from fluency bias and rely heavily on multiple-choice formats, making it difficult to assess factual accuracy and complex reasoning effectively. LLMs thus frequently generate factually inaccurate responses, especially in complex reasoning tasks, highlighting two prominent challenges: (1) the inadequacy of existing methods to evaluate reasoning and factual accuracy effectively, and (2) the reliance on human evaluators for nuanced judgment, as illustrated by Williams and Huckle (2024)[1], who found manual grading indispensable despite automated grading advancements.
To address evaluation gaps in open-ended reasoning tasks, we introduce the EQUATOR Evaluator (Evaluation of Question Answering Thoroughness in Open-ended Reasoning). This framework combines deterministic scoring with a focus on factual accuracy and robust reasoning assessment. Using a vector database, EQUATOR pairs open-ended questions with human-evaluated answers, enabling more precise and scalable evaluations. In practice, EQUATOR significantly reduces reliance on human evaluators for scoring and improves scalability compared to Williams and Huckle's (2004)[1] methods.
Our results demonstrate that this framework significantly outperforms traditional multiple-choice evaluations while maintaining high accuracy standards. Additionally, we introduce an automated evaluation process leveraging smaller, locally hosted LLMs. We used LLaMA 3.2B, running on the Ollama binaries to streamline our assessments. This work establishes a new paradigm for evaluating LLM performance, emphasizing factual accuracy and reasoning ability, and provides a robust methodological foundation for future research. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）具有显著的连贯性，现有的评估方法常常存在流畅性偏见，并且高度依赖多项选择格式，这使得评估事实正确性和复杂推理能力变得困难。因此，LLMs 经常生成事实不准确的回答，尤其是在复杂推理任务中，这突显了两个主要挑战：（1）现有方法评估推理和事实正确性能力的不足，以及（2）对依靠人类评估者的细致判断的依赖性，正如 Williams 和 Huckle（2024）[1] 所指出的，即便在自动化评分技术取得进展的情况下，人工评分依然是必不可少的。

为了填补开放性推理任务评估中的空白，我们引入了 EQUATOR 评估器（开放性推理中问题回答全面性的评估）。该框架结合了确定性评分，并着重于事实正确性和稳健推理的评估。通过使用向量数据库，EQUATOR 将开放式问题与人类评估的回答相关联，从而实现更精确和可扩展的评估。实际上，EQUATOR 显著减少了评分对人类评估者的依赖，并在可扩展性方面优于 Williams 和 Huckle（2004）[1] 的方法。

我们的结果显示，该框架显著优于传统的多项选择评估方法，同时保持了高准确度标准。此外，我们还引入了一种利用本地部署的小型语言模型的自动化评估过程。我们使用了 LLaMA 3.2B，运行在 Ollama 二进制文件上，以简化我们的评估过程。这项工作为评估 LLM 性能确立了新的范式，强调了事实正确性和推理能力，并为未来的研究提供了坚实的方法论基础。 

---
# Have We Designed Generalizable Structural Knowledge Promptings? Systematic Evaluation and Rethinking 

**Title (ZH)**: 我们设计的结构化知识提示具有普适性吗？系统评估与重新思考 

**Authors**: Yichi Zhang, Zhuo Chen, Lingbing Guo, Yajing Xu, Shaokai Chen, Mengshu Sun, Binbin Hu, Zhiqiang Zhang, Lei Liang, Wen Zhang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.00244)  

**Abstract**: Large language models (LLMs) have demonstrated exceptional performance in text generation within current NLP research. However, the lack of factual accuracy is still a dark cloud hanging over the LLM skyscraper. Structural knowledge prompting (SKP) is a prominent paradigm to integrate external knowledge into LLMs by incorporating structural representations, achieving state-of-the-art results in many knowledge-intensive tasks. However, existing methods often focus on specific problems, lacking a comprehensive exploration of the generalization and capability boundaries of SKP. This paper aims to evaluate and rethink the generalization capability of the SKP paradigm from four perspectives including Granularity, Transferability, Scalability, and Universality. To provide a thorough evaluation, we introduce a novel multi-granular, multi-level benchmark called SUBARU, consisting of 9 different tasks with varying levels of granularity and difficulty. 

**Abstract (ZH)**: 大型语言模型（LLMs）在当前自然语言处理（NLP）研究中展示了卓越的文本生成性能。然而，事实准确性的缺乏仍然是悬在LLMs头顶的乌云。结构化知识提示（SKP）是一种将外部知识整合到LLMs中的显著范式，通过融合结构化表示，已在许多知识密集型任务上取得了最先进的成果。然而，现有方法往往专注于特定问题，缺乏对SKP的一般化能力和边界进行全面探索。本文旨在从粒度、可迁移性、可扩展性和通用性四个维度评估和重新思考SKP范式的推广能力。为了进行全面评估，我们引入了一个新颖的多粒度、多层次基准SUBARU，该基准由9种不同粒度和难度的任务组成。 

---
# Exploring Variability in Fine-Tuned Models for Text Classification with DistilBERT 

**Title (ZH)**: 探索细调模型中使用DistilBERT进行文本分类的变异性 

**Authors**: Giuliano Lorenzoni, Ivens Portugal, Paulo Alencar, Donald Cowan  

**Link**: [PDF](https://arxiv.org/pdf/2501.00241)  

**Abstract**: This study evaluates fine-tuning strategies for text classification using the DistilBERT model, specifically the distilbert-base-uncased-finetuned-sst-2-english variant. Through structured experiments, we examine the influence of hyperparameters such as learning rate, batch size, and epochs on accuracy, F1-score, and loss. Polynomial regression analyses capture foundational and incremental impacts of these hyperparameters, focusing on fine-tuning adjustments relative to a baseline model.
Results reveal variability in metrics due to hyperparameter configurations, showing trade-offs among performance metrics. For example, a higher learning rate reduces loss in relative analysis (p=0.027) but challenges accuracy improvements. Meanwhile, batch size significantly impacts accuracy and F1-score in absolute regression (p=0.028 and p=0.005) but has limited influence on loss optimization (p=0.170). The interaction between epochs and batch size maximizes F1-score (p=0.001), underscoring the importance of hyperparameter interplay.
These findings highlight the need for fine-tuning strategies addressing non-linear hyperparameter interactions to balance performance across metrics. Such variability and metric trade-offs are relevant for tasks beyond text classification, including NLP and computer vision. This analysis informs fine-tuning strategies for large language models and promotes adaptive designs for broader model applicability. 

**Abstract (ZH)**: 本研究评估了使用DistilBERT模型进行文本分类时的微调策略，特别关注“distilbert-base-uncased-finetuned-sst-2-english”变体。通过结构化的实验，我们探讨了学习率、批次大小和 epoch 等超参数对准确率、F1 分数和损失的影响。多项式回归分析捕捉到了这些超参数的基础性和增量性影响，重点在于与基准模型相比的微调调整。

研究结果表明，由于超参数配置的不同，性能指标存在差异，显示出不同性能指标之间的权衡。例如，在相对分析中，较高的学习率减少了损失（p=0.027），但对准确率的提高构成了挑战。另一方面，批次大小显著影响绝对回归中的准确率和 F1 分数（p=0.028 和 p=0.005），但在损失优化方面影响有限（p=0.170）。批次大小与 epoch 的交互作用最大化了 F1 分数（p=0.001），突显了超参数交互作用的重要性。

这些发现强调了需要制定微调策略来处理非线性超参数交互作用，以在不同性能指标之间取得平衡。这种变异性和指标权衡不仅适用于文本分类任务，还适用于自然语言处理（NLP）和计算机视觉等任务。本分析为大型语言模型的微调策略提供指导，促进更广泛模型应用的适应性设计。 

---
# Zero-Shot Strategies for Length-Controllable Summarization 

**Title (ZH)**: 长度可控的零样本总结策略 

**Authors**: Fabian Retkowski, Alexander Waibel  

**Link**: [PDF](https://arxiv.org/pdf/2501.00233)  

**Abstract**: Large language models (LLMs) struggle with precise length control, particularly in zero-shot settings. We conduct a comprehensive study evaluating LLMs' length control capabilities across multiple measures and propose practical methods to improve controllability. Our experiments with LLaMA 3 reveal stark differences in length adherence across measures and highlight inherent biases of the model. To address these challenges, we introduce a set of methods: length approximation, target adjustment, sample filtering, and automated revisions. By combining these methods, we demonstrate substantial improvements in length compliance while maintaining or enhancing summary quality, providing highly effective zero-shot strategies for precise length control without the need for model fine-tuning or architectural changes. With our work, we not only advance our understanding of LLM behavior in controlled text generation but also pave the way for more reliable and adaptable summarization systems in real-world applications. 

**Abstract (ZH)**: 大型语言模型（LLMs）在精确长度控制方面存在挑战，特别是在零样本设置中。我们进行了一项全面的研究，评估LLMs在多个指标上的长度控制能力，并提出了一些实际方法来提高可控性。我们使用LaMA 3进行的实验揭示了在不同指标下长度遵守情况的巨大差异，并突显了该模型的固有偏差。为了应对这些挑战，我们介绍了一套方法：长度近似、目标调整、样本过滤和自动化修正。通过结合这些方法，我们展示了在保持或提升摘要质量的同时显著提高长度遵守度，为无需模型微调或架构更改即可实现精确长度控制提供了有效的零样本策略。通过我们的研究工作，我们不仅深化了对LLM在受控文本生成中行为的理解，还为在实际应用中提供更可靠和适应性强的摘要系统铺平了道路。 

---
# Extracting effective solutions hidden in large language models via generated comprehensive specialists: case studies in developing electronic devices 

**Title (ZH)**: 通过生成综合专家提取大型语言模型中隐藏的有效解决方案：电子设备开发案例研究 

**Authors**: Hikari Tomita, Nobuhiro Nakamura, Shoichi Ishida, Toshio Kamiya, Kei Terayama  

**Link**: [PDF](https://arxiv.org/pdf/2501.00224)  

**Abstract**: Recently, many studies have increasingly explored the use of large language models (LLMs) to generate research ideas and scientific hypotheses. However, real-world research and development often require solving complex, interdisciplinary challenges where solutions may not be readily found through existing knowledge related to the problem. Therefore, it is desirable to leverage the vast, comprehensive knowledge of LLMs to generate effective, breakthrough solutions by integrating various perspectives from other disciplines. Here, we propose SELLM (Solution Enumeration via comprehensive List and LLM), a framework leveraging LLMs and structured guidance using MECE (Mutually Exclusive, Collectively Exhaustive) principles, such as International Patent Classification (IPC) and the periodic table of elements. SELLM systematically constructs comprehensive expert agents from the list to generate cross-disciplinary and effective solutions. To evaluate SELLM's practicality, we applied it to two challenges: improving light extraction in organic light-emitting diode (OLED) lighting and developing electrodes for next-generation memory materials. The results demonstrate that SELLM significantly facilitates the generation of effective solutions compared to cases without specific customization or effort, showcasing the potential of SELLM to enable LLMs to generate effective solutions even for challenging problems. 

**Abstract (ZH)**: 近年来，许多研究已经开始探索使用大规模语言模型（LLMs）生成研究想法和科学假说。然而，实际的研究和发展往往需要解决复杂且跨学科的挑战，这些问题可能无法通过现有的相关知识直接找到解决方案。因此，利用LLMs的庞大且综合的知识库，通过多学科角度进行综合，生成有效的突破性解决方案是十分必要的。在此，我们提出了SELLM（基于综合列表和LLMs的解决方案枚举框架），该框架利用LLMs和MECE（互斥且详尽）原则（例如国际专利分类IPC和元素周期表）的结构化指导。SELLM系统地从列表中构建全面的专家代理，生成跨学科且有效的解决方案。为了评估SELLM的实际应用价值，我们将其应用于两个挑战：改善有机发光二极管（OLED）照明的光提取效率和开发下一代记忆材料的电极。结果显示，SELLM显著提高了生成有效解决方案的能力，相较于无特定定制或努力的情况，证明了SELLM在解决复杂问题时使LLMs生成有效解决方案的潜力。 

---
# An Empirical Evaluation of Large Language Models on Consumer Health Questions 

**Title (ZH)**: 大型语言模型在解答消费者健康问题方面的实证评估 

**Authors**: Moaiz Abrar, Yusuf Sermet, Ibrahim Demir  

**Link**: [PDF](https://arxiv.org/pdf/2501.00208)  

**Abstract**: This study evaluates the performance of several Large Language Models (LLMs) on MedRedQA, a dataset of consumer-based medical questions and answers by verified experts extracted from the AskDocs subreddit. While LLMs have shown proficiency in clinical question answering (QA) benchmarks, their effectiveness on real-world, consumer-based, medical questions remains less understood. MedRedQA presents unique challenges, such as informal language and the need for precise responses suited to non-specialist queries. To assess model performance, responses were generated using five LLMs: GPT-4o mini, Llama 3.1: 70B, Mistral-123B, Mistral-7B, and Gemini-Flash. A cross-evaluation method was used, where each model evaluated its responses as well as those of others to minimize bias. The results indicated that GPT-4o mini achieved the highest alignment with expert responses according to four out of the five models' judges, while Mistral-7B scored lowest according to three out of five models' judges. This study highlights the potential and limitations of current LLMs for consumer health medical question answering, indicating avenues for further development. 

**Abstract (ZH)**: 本研究评估了几种大规模语言模型（LLMs）在MedRedQA数据集上的性能，该数据集包含来自AskDocs subreddit的专业验证消费者医疗问题和答案。尽管LLMs在临床问答（QA）基准测试中显示出专业性，但它们在实际的、以消费者为基础的医疗问题上的效果尚不完全清楚。MedRedQA提出了独特的挑战，例如非正式的语言以及需要适合非专业人士查询的精确回答。为了评估模型性能，使用了五个LLM生成了响应：GPT-4o mini、Llama 3.1: 70B、Mistral-123B、Mistral-7B和Gemini-Flash。采用了一种交叉评估方法，每种模型不仅评估自己的响应，还评估其他模型的响应，以最小化偏差。结果显示，根据五种模型中四种模型的评判，GPT-4o mini的表现与专家回答的契合度最高，而Mistral-7B的得分最低，根据五种模型中三种模型的评判。本研究指出了当前LLMs在消费者健康医疗问答方面的潜力和局限性，并指出了进一步发展的途径。 

---
# GPT-4 on Clinic Depression Assessment: An LLM-Based Pilot Study 

**Title (ZH)**: GPT-4在临床抑郁评估中的初步研究：基于大规模语言模型的探索 

**Authors**: Giuliano Lorenzoni, Pedro Elkind Velmovitsky, Paulo Alencar, Donald Cowan  

**Link**: [PDF](https://arxiv.org/pdf/2501.00199)  

**Abstract**: Depression has impacted millions of people worldwide and has become one of the most prevalent mental disorders. Early mental disorder detection can lead to cost savings for public health agencies and avoid the onset of other major comorbidities. Additionally, the shortage of specialized personnel is a critical issue because clinical depression diagnosis is highly dependent on expert professionals and is time consuming.
In this study, we explore the use of GPT-4 for clinical depression assessment based on transcript analysis. We examine the model's ability to classify patient interviews into binary categories: depressed and not depressed. A comparative analysis is conducted considering prompt complexity (e.g., using both simple and complex prompts) as well as varied temperature settings to assess the impact of prompt complexity and randomness on the model's performance.
Results indicate that GPT-4 exhibits considerable variability in accuracy and F1-Score across configurations, with optimal performance observed at lower temperature values (0.0-0.2) for complex prompts. However, beyond a certain threshold (temperature >= 0.3), the relationship between randomness and performance becomes unpredictable, diminishing the gains from prompt complexity.
These findings suggest that, while GPT-4 shows promise for clinical assessment, the configuration of the prompts and model parameters requires careful calibration to ensure consistent results. This preliminary study contributes to understanding the dynamics between prompt engineering and large language models, offering insights for future development of AI-powered tools in clinical settings. 

**Abstract (ZH)**: 抑郁障碍已影响到全世界 millions 的人群，并已成为最常见的精神疾病之一。早期精神障碍检测可以为公共卫生机构节省成本，并避免其他严重共病的发生。此外，专业人员短缺是一个关键问题，因为临床抑郁诊断需要依赖专家并耗时较长。

在本研究中，我们探讨了使用 GPT-4 对抑郁障碍进行评估的可行性，基于访谈记录的分析。我们研究了该模型根据患者访谈将其分类为抑郁和非抑郁两类的能力。我们对不同提示复杂性（例如使用简单和复杂提示）以及不同的温度设置进行了比较分析，以评估提示复杂性和随机性对模型性能的影响。

结果显示，GPT-4 在不同配置下的准确性和F1-Score表现出显著波动，对于复杂提示，最佳性能通常出现在较低的温度值（0.0-0.2）。然而，超出某个阈值（温度≥0.3）后，随机性与性能之间的关系变得不可预测，从而减少了提示复杂性的收益。

这些发现表明，虽然 GPT-4 在临床评估方面具有潜力，但需要仔细调整提示配置和模型参数以确保一致的结果。这项初步研究为理解提示工程与大型语言模型之间的动态关系提供了见解，为未来开发临床环境中的AI辅助工具提供了参考。 

---
# The Text Classification Pipeline: Starting Shallow going Deeper 

**Title (ZH)**: 文本分类流水线：从浅层开始逐步深入 

**Authors**: Marco Siino, Ilenia Tinnirello, Marco La Cascia  

**Link**: [PDF](https://arxiv.org/pdf/2501.00174)  

**Abstract**: Text Classification (TC) stands as a cornerstone within the realm of Natural Language Processing (NLP), particularly when viewed through the lens of computer science and engineering. The past decade has seen deep learning revolutionize TC, propelling advancements in text retrieval, categorization, information extraction, and summarization. The scholarly literature is rich with datasets, models, and evaluation criteria, with English being the predominant language of focus, despite studies involving Arabic, Chinese, Hindi, and others. The efficacy of TC models relies heavily on their ability to capture intricate textual relationships and nonlinear correlations, necessitating a comprehensive examination of the entire TC pipeline.
This monograph provides an in-depth exploration of the TC pipeline, with a particular emphasis on evaluating the impact of each component on the overall performance of TC models. The pipeline includes state-of-the-art datasets, text preprocessing techniques, text representation methods, classification models, evaluation metrics, current results and future trends. Each chapter meticulously examines these stages, presenting technical innovations and significant recent findings. The work critically assesses various classification strategies, offering comparative analyses, examples, case studies, and experimental evaluations. These contributions extend beyond a typical survey, providing a detailed and insightful exploration of TC. 

**Abstract (ZH)**: 文本分类（TC）作为自然语言处理（NLP）领域的一个基石，在计算机科学和工程的视角下尤其重要。过去十年间，深度学习技术的革新极大地推动了TC的发展，改善了文本检索、分类、信息提取和摘要等任务。学术文献中充满了各种数据集、模型和评估标准，尽管大多数研究集中在英文上，但也有涉及阿拉伯语、中文、印地语等其他语言的研究。

TC模型的有效性很大程度上依赖于其捕捉复杂文本关系和非线性关联的能力，因此全面审视整个TC管道是至关重要的。
本专著深入探讨了TC管道的工作流程，重点关注每个组成部分对TC模型整体性能的影响。该管道涵盖了最新数据集、文本预处理技术、文本表示方法、分类模型、评估指标、当前结果及未来趋势。每一章详细分析了这些阶段，介绍了技术创新和近期的重要发现。该专著批判性地评估了各种分类策略，提供了对比分析、实例、案例研究和实验评估。这些贡献超越了一般的综述性文章，为TC提供了详尽而深刻的探索。 

---
# Measuring Large Language Models Capacity to Annotate Journalistic Sourcing 

**Title (ZH)**: 测量大型语言模型在标注新闻来源方面的能力 

**Authors**: Subramaniam Vincent, Phoebe Wang, Zhan Shi, Sahas Koka, Yi Fang  

**Link**: [PDF](https://arxiv.org/pdf/2501.00164)  

**Abstract**: Since the launch of ChatGPT in late 2022, the capacities of Large Language Models and their evaluation have been in constant discussion and evaluation both in academic research and in the industry. Scenarios and benchmarks have been developed in several areas such as law, medicine and math (Bommasani et al., 2023) and there is continuous evaluation of model variants. One area that has not received sufficient scenario development attention is journalism, and in particular journalistic sourcing and ethics. Journalism is a crucial truth-determination function in democracy (Vincent, 2023), and sourcing is a crucial pillar to all original journalistic output. Evaluating the capacities of LLMs to annotate stories for the different signals of sourcing and how reporters justify them is a crucial scenario that warrants a benchmark approach. It offers potential to build automated systems to contrast more transparent and ethically rigorous forms of journalism with everyday fare. In this paper we lay out a scenario to evaluate LLM performance on identifying and annotating sourcing in news stories on a five-category schema inspired from journalism studies (Gans, 2004). We offer the use case, our dataset and metrics and as the first step towards systematic benchmarking. Our accuracy findings indicate LLM-based approaches have more catching to do in identifying all the sourced statements in a story, and equally, in matching the type of sources. An even harder task is spotting source justifications. 

**Abstract (ZH)**: 自2022年底ChatGPT发布以来，大型语言模型及其评估能力一直在学术界和工业界持续讨论和评估。已在法律、医学和数学等多个领域开发了情景和基准测试（Bommasani等，2023），并且不断对模型变体进行评估。然而，尚未在新闻业，尤其是新闻来源和伦理方面获得足够的情景发展关注。新闻业在民主制度中是决定事实的关键功能（Vincent，2023），而来源是所有原创新闻输出的关键支柱。评估大型语言模型在识别和注释新闻故事中不同来源信号的能力及其记者如何证明这些信号的能力是至关重要的一个情景，需要采用基准测试方法。这对于构建自动化系统，以更透明和伦理严谨的方式来对比日常新闻内容具有潜在价值。本文旨在提出一个情景，利用来自新闻学研究的五类架构（Gans，2004）评估大型语言模型在识别和注释新闻故事来源方面的性能。我们提供了应用场景、数据集和评估指标，并迈出了系统基准测试的第一步。我们的准确度结果显示，基于大型语言模型的方法在识别故事中所有来源声明方面还有待改进，同样在匹配不同类型的来源方面亦有不足。更困难的任务在于识别来源证明。 

---
# Temporal reasoning for timeline summarisation in social media 

**Title (ZH)**: 社交媒体时间线总结中的时间推理 

**Authors**: Jiayu Song, Mahmud Akhter, Dana Atzil Slonim, Maria Liakata  

**Link**: [PDF](https://arxiv.org/pdf/2501.00152)  

**Abstract**: This paper explores whether enhancing temporal reasoning capabilities in Large Language Models (LLMs) can improve the quality of timeline summarization, the task of summarising long texts containing sequences of events, particularly social media threads . We introduce \textit{NarrativeReason}, a novel dataset focused on temporal relationships among sequential events within narratives, distinguishing it from existing temporal reasoning datasets that primarily address pair-wise event relationships. Our approach then combines temporal reasoning with timeline summarization through a knowledge distillation framework, where we first fine-tune a teacher model on temporal reasoning tasks and then distill this knowledge into a student model while simultaneously training it for the task of timeline summarization. Experimental results demonstrate that our model achieves superior performance on mental health-related timeline summarization tasks, which involve long social media threads with repetitions of events and a mix of emotions, highlighting the importance of leveraging temporal reasoning to improve timeline summarisation. 

**Abstract (ZH)**: 本文探讨了在大型语言模型（LLMs）中增强时间推理能力是否能够提高时间线总结的质量，时间线总结是指总结包含一系列事件的长文本，特别是社交媒体帖子。我们介绍了\textit{NarrativeReason}，这是一个新颖的数据集，专注于叙述中序列事件间的时间关系，将其与主要关注成对事件关系的现有时间推理数据集区分开来。然后，我们的方法通过一个知识蒸馏框架将时间推理与时间线总结相结合，首先在时间推理任务上对教师模型进行微调，然后将这些知识蒸馏到学生模型中，并同时训练其完成时间线总结任务。实验结果表明，我们的模型在心理健康相关的时间线总结任务上表现更优，这些任务涉及长的社交媒体帖子，包含事件的重复以及复杂的情绪混合，突出了利用时间推理提高时间线总结的重要性和优势。 

---
# A Data-Centric Approach to Detecting and Mitigating Demographic Bias in Pediatric Mental Health Text: A Case Study in Anxiety Detection 

**Title (ZH)**: 一种以数据为中心的方法用于检测和缓解儿童心理健康文本中的人口统计偏差：以焦虑检测为例 

**Authors**: Julia Ive, Paulina Bondaronek, Vishal Yadav, Daniel Santel, Tracy Glauser, Tina Cheng, Jeffrey R. Strawn, Greeshma Agasthya, Jordan Tschida, Sanghyun Choo, Mayanka Chandrashekar, Anuj J. Kapadia, John Pestian  

**Link**: [PDF](https://arxiv.org/pdf/2501.00129)  

**Abstract**: Introduction: Healthcare AI models often inherit biases from their training data. While efforts have primarily targeted bias in structured data, mental health heavily depends on unstructured data. This study aims to detect and mitigate linguistic differences related to non-biological differences in the training data of AI models designed to assist in pediatric mental health screening. Our objectives are: (1) to assess the presence of bias by evaluating outcome parity across sex subgroups, (2) to identify bias sources through textual distribution analysis, and (3) to develop a de-biasing method for mental health text data. Methods: We examined classification parity across demographic groups and assessed how gendered language influences model predictions. A data-centric de-biasing method was applied, focusing on neutralizing biased terms while retaining salient clinical information. This methodology was tested on a model for automatic anxiety detection in pediatric patients. Results: Our findings revealed a systematic under-diagnosis of female adolescent patients, with a 4% lower accuracy and a 9% higher False Negative Rate (FNR) compared to male patients, likely due to disparities in information density and linguistic differences in patient notes. Notes for male patients were on average 500 words longer, and linguistic similarity metrics indicated distinct word distributions between genders. Implementing our de-biasing approach reduced diagnostic bias by up to 27%, demonstrating its effectiveness in enhancing equity across demographic groups. Discussion: We developed a data-centric de-biasing framework to address gender-based content disparities within clinical text. By neutralizing biased language and enhancing focus on clinically essential information, our approach demonstrates an effective strategy for mitigating bias in AI healthcare models trained on text. 

**Abstract (ZH)**: 引言：医疗AI模型往往会从训练数据中继承偏差。尽管努力主要集中在结构化数据上的偏差，但心理健康状况严重依赖于非结构化数据。本研究旨在检测和缓解AI模型在儿科心理健康筛查设计时所用训练数据中与生物差异无关的语言差异。我们的目标是：（1）通过评估性别亚组之间的结果公平性来评估偏差的存在；（2）通过文本分布分析识别偏差来源；（3）开发一种消除心理健康文本数据偏差的方法。方法：我们分析了不同人口统计学组分类的公平性，并评估了性别化语言如何影响模型预测。应用了一种以数据为中心的去偏差方法，专注于中和偏斜词汇的同时保留关键的临床信息。该方法在用于自动检测儿科患者焦虑的模型中进行了测试。结果：我们的研究表明，女性青少年患者存在系统性的诊断不足，其准确性比男性患者低4%，且误诊率（FNR）高9%，这可能是由于患者记录中的信息密度差异和语言差异导致的。男性患者的笔记平均比女性长500个单词，语言相似度指标显示出性别间的词分布差异。实施我们的去偏差方法将诊断偏差降低了最多27%，证明了其在不同人口统计学组间增强公平性的有效性。讨论：我们开发了一种以数据为中心的去偏差框架，以解决临床文本中的性别内容差异。通过中和偏斜语言并增强对临床关键信息的重视，我们的方法展示了在基于文本训练的医疗AI模型中缓解偏斜的有效策略。 

---
# CaseSumm: A Large-Scale Dataset for Long-Context Summarization from U.S. Supreme Court Opinions 

**Title (ZH)**: CaseSumm：来自美国最高法院判决的大型规模数据集，用于长上下文摘要生成 

**Authors**: Mourad Heddaya, Kyle MacMillan, Anup Malani, Hongyuan Mei, Chenhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2501.00097)  

**Abstract**: This paper introduces CaseSumm, a novel dataset for long-context summarization in the legal domain that addresses the need for longer and more complex datasets for summarization evaluation. We collect 25.6K U.S. Supreme Court (SCOTUS) opinions and their official summaries, known as "syllabuses." Our dataset is the largest open legal case summarization dataset, and is the first to include summaries of SCOTUS decisions dating back to 1815.
We also present a comprehensive evaluation of LLM-generated summaries using both automatic metrics and expert human evaluation, revealing discrepancies between these assessment methods. Our evaluation shows Mistral 7b, a smaller open-source model, outperforms larger models on most automatic metrics and successfully generates syllabus-like summaries. In contrast, human expert annotators indicate that Mistral summaries contain hallucinations. The annotators consistently rank GPT-4 summaries as clearer and exhibiting greater sensitivity and specificity. Further, we find that LLM-based evaluations are not more correlated with human evaluations than traditional automatic metrics. Furthermore, our analysis identifies specific hallucinations in generated summaries, including precedent citation errors and misrepresentations of case facts. These findings demonstrate the limitations of current automatic evaluation methods for legal summarization and highlight the critical role of human evaluation in assessing summary quality, particularly in complex, high-stakes domains.
CaseSumm is available at this https URL 

**Abstract (ZH)**: 本文介绍了CaseSumm，这是一个在法律领域用于长上下文摘要的新颖数据集，旨在满足摘要评估中对更长、更复杂数据集的需求。我们收集了25,600份美国最高法院（SCOTUS）的判决书及其官方摘要，即“概要”。我们的数据集是最大的开放法律案例摘要数据集，并且是首个包括自1815年以来SCOTUS判决摘要的数据集。

我们还对大语言模型（LLM）生成的摘要进行了全面评估，使用了自动评价指标和专家的人类评估，揭示了这两种评价方法之间的差异。评估结果显示，一个较小的开源模型Mistral 7b在大部分自动评价指标上表现更优，并能生成类似概要的摘要。相比之下，人类专家标注者指出Mistral摘要包含虚构内容。标注者一致认为GPT-4摘要更为清晰，并且更具有针对性和准确性。此外，我们发现基于大语言模型的评估与人类评估的相关性并不高于传统自动评价指标。进一步分析还发现生成摘要中的特定虚构内容，包括案例先例引用错误和对案件事实的曲解。这些发现表明当前自动评价方法在法律摘要评估中的局限性，并突显了在复杂、高风险领域中进行人类评估的重要性。

CaseSumm 数据集可在以下链接访问：https://CASESumm.link（请根据实际情况调整链接） 

---
# Position Information Emerges in Causal Transformers Without Positional Encodings via Similarity of Nearby Embeddings 

**Title (ZH)**: 无需位置编码的情况下，因果变压器中位置信息通过邻近嵌入的相似性 Emerges 无需位置编码的因果变压器中位置信息通过邻近嵌入的相似性 

**Authors**: Chunsheng Zuo, Pavel Guerzhoy, Michael Guerzhoy  

**Link**: [PDF](https://arxiv.org/pdf/2501.00073)  

**Abstract**: Transformers with causal attention can solve tasks that require positional information without using positional encodings. In this work, we propose and investigate a new hypothesis about how positional information can be stored without using explicit positional encoding. We observe that nearby embeddings are more similar to each other than faraway embeddings, allowing the transformer to potentially reconstruct the positions of tokens. We show that this pattern can occur in both the trained and the randomly initialized Transformer models with causal attention and no positional encodings over a common range of hyperparameters. 

**Abstract (ZH)**: 具有因果注意力的变换器能够在无需使用位置编码的情况下解决需要位置信息的任务。在本研究中，我们提出并探讨了一个新的假设，即位置信息如何能够在不使用显式位置编码的情况下被存储。我们观察到，相邻的嵌入彼此更为相似，而远离的嵌入则相差较大，从而使变换器有可能重新构造出标记的位置。我们展示了这一模式可以在具有因果注意力且未使用位置编码的训练和随机初始化的变换器模型中，在一系列常用的超参数范围内同时出现。 

---
# ICLR: In-Context Learning of Representations 

**Title (ZH)**: ICLR：基于上下文的学习表示 

**Authors**: Core Francisco Park, Andrew Lee, Ekdeep Singh Lubana, Yongyi Yang, Maya Okawa, Kento Nishi, Martin Wattenberg, Hidenori Tanaka  

**Link**: [PDF](https://arxiv.org/pdf/2501.00070)  

**Abstract**: Recent work has demonstrated that semantics specified by pretraining data influence how representations of different concepts are organized in a large language model (LLM). However, given the open-ended nature of LLMs, e.g., their ability to in-context learn, we can ask whether models alter these pretraining semantics to adopt alternative, context-specified ones. Specifically, if we provide in-context exemplars wherein a concept plays a different role than what the pretraining data suggests, do models reorganize their representations in accordance with these novel semantics? To answer this question, we take inspiration from the theory of conceptual role semantics and define a toy "graph tracing" task wherein the nodes of the graph are referenced via concepts seen during training (e.g., apple, bird, etc.) and the connectivity of the graph is defined via some predefined structure (e.g., a square grid). Given exemplars that indicate traces of random walks on the graph, we analyze intermediate representations of the model and find that as the amount of context is scaled, there is a sudden re-organization from pretrained semantic representations to in-context representations aligned with the graph structure. Further, we find that when reference concepts have correlations in their semantics (e.g., Monday, Tuesday, etc.), the context-specified graph structure is still present in the representations, but is unable to dominate the pretrained structure. To explain these results, we analogize our task to energy minimization for a predefined graph topology, providing evidence towards an implicit optimization process to infer context-specified semantics. Overall, our findings indicate scaling context-size can flexibly re-organize model representations, possibly unlocking novel capabilities. 

**Abstract (ZH)**: 近年来的研究表明，预训练数据指定的语义会影响大规模语言模型（LLM）中不同概念的表示方式。然而，鉴于LLM的开放性，例如它们的上下文学习能力，我们不禁要问：模型是否会根据上下文改变这些预训练语义，采用新的、上下文指定的语义？具体来说，如果我们提供一种上下文示例，其中概念的作用与预训练数据所暗示的有所不同，模型是否会根据这些新的语义重新组织其表示？为了回答这个问题，我们借鉴概念角色语义理论，定义了一个简单的“图跟踪”任务：图的节点通过训练过程中出现的概念进行引用（例如，苹果、鸟等），而图的连接性则通过某种预定义结构（例如，方形网格）来定义。鉴于示例表明了图上随机游走的路径，我们分析模型的中间表示，并发现随着上下文量的增加，模型的表示经历了从预训练语义到与图结构对齐的上下文指定语义的突然重新组织。进一步的研究表明，当引用概念在语义上存在关联（例如，周一、周二等）时，上下文指定的图结构仍然存在于模型表示中，但无法占主导地位。为了解释这些结果，我们将我们的任务类比为预定义图拓扑的能最化过程，提供了隐含的优化过程的证据，以推断上下文指定的语义。总体而言，我们的研究表明，扩大上下文规模可以灵活地重新组织模型的表示，可能解锁新的功能。 

---
# Adversarial Negotiation Dynamics in Generative Language Models 

**Title (ZH)**: 生成语言模型中的对抗性谈判动力学 

**Authors**: Arinbjörn Kolbeinsson, Benedikt Kolbeinsson  

**Link**: [PDF](https://arxiv.org/pdf/2501.00069)  

**Abstract**: Generative language models are increasingly used for contract drafting and enhancement, creating a scenario where competing parties deploy different language models against each other. This introduces not only a game-theory challenge but also significant concerns related to AI safety and security, as the language model employed by the opposing party can be unknown. These competitive interactions can be seen as adversarial testing grounds, where models are effectively red-teamed to expose vulnerabilities such as generating biased, harmful or legally problematic text. Despite the importance of these challenges, the competitive robustness and safety of these models in adversarial settings remain poorly understood. In this small study, we approach this problem by evaluating the performance and vulnerabilities of major open-source language models in head-to-head competitions, simulating real-world contract negotiations. We further explore how these adversarial interactions can reveal potential risks, informing the development of more secure and reliable models. Our findings contribute to the growing body of research on AI safety, offering insights into model selection and optimisation in competitive legal contexts and providing actionable strategies for mitigating risks. 

**Abstract (ZH)**: 生成式语言模型在合同起草和优化中的应用日益增多，这导致竞争各方可能使用不同的语言模型对抗，从而不仅带来博弈论上的挑战，还引发了重大的人工智能安全和安全问题。因为对手使用的语言模型可能是未知的。这些竞争性互动可以被视为具有挑战性的对抗环境，模型在此环境中被有效地用作“红队”以暴露潜在漏洞，例如生成带有偏见、有害或法律问题的文字。尽管这些挑战至关重要，但在对抗环境中的模型的 robustness 和安全性仍然缺乏充分理解。在这项初步研究中，我们通过将主要的开源语言模型在一对一竞赛中进行评估，模拟实际的合同谈判情况，来探索这一问题。我们进一步探讨了这些对抗性互动如何揭示潜在风险，为开发更安全、更可靠的语言模型提供了指导。我们的发现补充了关于人工智能安全的研究，为在竞争性的法律环境中进行模型选择和优化提供了见解，并提出了应对风险的实际策略。 

---
# On Adversarial Robustness of Language Models in Transfer Learning 

**Title (ZH)**: 语言模型在迁移学习中的对抗robust性研究 

**Authors**: Bohdan Turbal, Anastasiia Mazur, Jiaxu Zhao, Mykola Pechenizkiy  

**Link**: [PDF](https://arxiv.org/pdf/2501.00066)  

**Abstract**: We investigate the adversarial robustness of LLMs in transfer learning scenarios. Through comprehensive experiments on multiple datasets (MBIB Hate Speech, MBIB Political Bias, MBIB Gender Bias) and various model architectures (BERT, RoBERTa, GPT-2, Gemma, Phi), we reveal that transfer learning, while improving standard performance metrics, often leads to increased vulnerability to adversarial attacks. Our findings demonstrate that larger models exhibit greater resilience to this phenomenon, suggesting a complex interplay between model size, architecture, and adaptation methods. Our work highlights the crucial need for considering adversarial robustness in transfer learning scenarios and provides insights into maintaining model security without compromising performance. These findings have significant implications for the development and deployment of LLMs in real-world applications where both performance and robustness are paramount. 

**Abstract (ZH)**: 我们研究了在迁移学习场景中大型语言模型（LLMs）的对抗鲁棒性。通过在多个数据集（MBIB仇恨言论、MBIB政治偏见、MBIB性别偏见）和多种模型架构（BERT、RoBERTa、GPT-2、Gemma、Phi）上进行综合实验，我们发现，虽然迁移学习可以改善标准性能指标，但往往会增加模型对对抗攻击的脆弱性。我们的研究结果表明，较大的模型展现出更强的对该现象的抵抗力，这暗示了模型大小、架构和适应方法之间复杂的相互作用。我们的工作突显了在迁移学习场景中考虑对抗鲁棒性的重要性，并为在不牺牲性能的前提下维护模型安全提供了见解。这些发现对于在真实应用场景中开发和部署既具有出色性能又具有高鲁棒性的大型语言模型具有重要意义。 

---
# ELECTRA and GPT-4o: Cost-Effective Partners for Sentiment Analysis 

**Title (ZH)**: ELECTRA和GPT-4o：经济高效的 sentiment 分析合作伙伴 

**Authors**: James P. Beno  

**Link**: [PDF](https://arxiv.org/pdf/2501.00062)  

**Abstract**: Bidirectional transformers excel at sentiment analysis, and Large Language Models (LLM) are effective zero-shot learners. Might they perform better as a team? This paper explores collaborative approaches between ELECTRA and GPT-4o for three-way sentiment classification. We fine-tuned (FT) four models (ELECTRA Base/Large, GPT-4o/4o-mini) using a mix of reviews from Stanford Sentiment Treebank (SST) and DynaSent. We provided input from ELECTRA to GPT as: predicted label, probabilities, and retrieved examples. Sharing ELECTRA Base FT predictions with GPT-4o-mini significantly improved performance over either model alone (82.74 macro F1 vs. 79.29 ELECTRA Base FT, 79.52 GPT-4o-mini) and yielded the lowest cost/performance ratio (\$0.12/F1 point). However, when GPT models were fine-tuned, including predictions decreased performance. GPT-4o FT-M was the top performer (86.99), with GPT-4o-mini FT close behind (86.77) at much less cost (\$0.38 vs. \$1.59/F1 point). Our results show that augmenting prompts with predictions from fine-tuned encoders is an efficient way to boost performance, and a fine-tuned GPT-4o-mini is nearly as good as GPT-4o FT at 76% less cost. Both are affordable options for projects with limited resources. 

**Abstract (ZH)**: 双向变换器在情感分析方面表现出色，而大规模语言模型（LLM）是有效的零样本学习者。它们作为团队是否表现更好？本文探讨了ELECTRA与GPT-4o三种情感分类方法的合作策略。我们使用斯坦福情感树库（SST）和DynaSent的混合评论数据，精细调整了四个模型（ELECTRA 基础/大型，GPT-4o/4o-迷你）。我们将ELECTRA的输入提供给GPT，内容包括预测标签、概率以及检索到的示例。向GPT-4o-迷你提供ELECTRA的已精细调整预测显著提高了性能（宏F1得分为82.74，高于仅使用ELECTRA基础模型的81.79和仅使用GPT-4o-迷你模型的82.02），并且性能价格比最优（每得一分宏F1成本为0.12美元）。然而，当GPT模型自身进行了精细调整并包含预测时，性能反而下降了。GPT-4o的FT-M版本表现最佳，得分为86.99，而GPT-4o-迷你的FT版本接近其水平，得分为86.77，成本明显更低（每得一分宏F1成本为0.38美元对比1.59美元）。我们的结果表明，用已精细调整编码器的预测补充提示是一种有效的性能提升策略，并且与GPT-4o的精细调整版本相比，精细调整的GPT-4o-迷你版本的成本降低了76%，性能几乎相同。这两种模型是资源有限项目中经济实惠的选择。 

---
# Large Language Models for Mathematical Analysis 

**Title (ZH)**: 大规模语言模型在数学分析中的应用 

**Authors**: Ziye Chen, Hao Qi  

**Link**: [PDF](https://arxiv.org/pdf/2501.00059)  

**Abstract**: Mathematical problem-solving is a key field in artificial intelligence (AI) and a critical benchmark for evaluating the capabilities of large language models (LLMs). While extensive research has focused on mathematical problem-solving, most existing work and datasets concentrate on computational tasks, leaving gaps in areas like mathematical analysis, which demands rigorous proofs and formal reasoning. We developed the DEMI-MathAnalysis dataset, comprising proof-based problems from mathematical analysis topics such as Sequences and Limits, Infinite Series, and Convex Functions. We also designed a guiding framework to rigorously enhance LLMs' ability to solve these problems. Through fine-tuning LLMs on this dataset and employing our framework, we observed significant improvements in their capability to generate logical, complete, and elegant proofs. This work addresses critical gaps in mathematical reasoning and contributes to advancing trustworthy AI capable of handling formalized mathematical language. The code is publicly accessible at LLMs for Mathematical Analysis. 

**Abstract (ZH)**: 数学问题求解是人工智能（AI）的一个关键领域，同时也是评估大型语言模型（LLMs）能力的一个重要基准。尽管已有大量研究集中在数学问题求解上，但大多数现有工作和数据集主要关注计算任务，而在需要严谨证明和正式推理的数学分析领域则存在空白。我们开发了DEMIMathAnalysis数据集，其中包含来自序列与极限、无穷级数和凸函数等数学分析主题的基于证明的问题。我们还设计了一个指导框架，以严格提升LLMs解决这些问题的能力。通过在该数据集上微调LLMs并采用我们设计的框架，我们观察到它们生成逻辑、完整且优雅证明的能力显著提高。这项工作填补了数学推理的关键空白，并促进了能够处理正式化数学语言的可信赖AI的发展。相关代码可从LLMs for Mathematical Analysis公开访问。 

---
# Seq2Seq Model-Based Chatbot with LSTM and Attention Mechanism for Enhanced User Interaction 

**Title (ZH)**: 基于LSTM和注意力机制的Seq2Seq模型聊天机器人以增强用户交互 

**Authors**: Lamya Benaddi, Charaf Ouaddi, Adnane Souha, Abdeslam Jakimi, Mohamed Rahouti, Mohammed Aledhari, Diogo Oliveira, Brahim Ouchao  

**Link**: [PDF](https://arxiv.org/pdf/2501.00049)  

**Abstract**: A chatbot is an intelligent software application that automates conversations and engages users in natural language through messaging platforms. Leveraging artificial intelligence (AI), chatbots serve various functions, including customer service, information gathering, and casual conversation. Existing virtual assistant chatbots, such as ChatGPT and Gemini, demonstrate the potential of AI in Natural Language Processing (NLP). However, many current solutions rely on predefined APIs, which can result in vendor lock-in and high costs. To address these challenges, this work proposes a chatbot developed using a Sequence-to-Sequence (Seq2Seq) model with an encoder-decoder architecture that incorporates attention mechanisms and Long Short-Term Memory (LSTM) cells. By avoiding predefined APIs, this approach ensures flexibility and cost-effectiveness. The chatbot is trained, validated, and tested on a dataset specifically curated for the tourism sector in Draa-Tafilalet, Morocco. Key evaluation findings indicate that the proposed Seq2Seq model-based chatbot achieved high accuracies: approximately 99.58% in training, 98.03% in validation, and 94.12% in testing. These results demonstrate the chatbot's effectiveness in providing relevant and coherent responses within the tourism domain, highlighting the potential of specialized AI applications to enhance user experience and satisfaction in niche markets. 

**Abstract (ZH)**: 聊天机器人是一种智能化的软件应用程序，能够自动化对话并利用即时通讯平台与用户进行自然语言交互。通过利用人工智能（AI），聊天机器人可以承担多种功能，包括客户服务、信息收集和休闲对话。现有的虚拟助手聊天机器人，如ChatGPT和Gemini，展示了人工智能在自然语言处理（NLP）领域中的潜力。然而，许多现有的解决方案依赖于预定义的API，这可能导致供应商锁定和高成本。为了解决这些挑战，本研究提出了一种基于序列到序列（Seq2Seq）模型的聊天机器人，该模型采用了编码器-解码器架构，并结合了注意力机制和长短期记忆（LSTM）单元。通过避免使用预定义的API，这种方法确保了灵活性和成本效益。该聊天机器人是在专门为摩洛哥德拉塔菲勒特地区的旅游业定制的数据集上进行训练、验证和测试的。关键评估结果显示，基于Seq2Seq模型的聊天机器人的准确率较高：训练准确率为约99.58%，验证准确率为98.03%，测试准确率为94.12%。这些结果表明，该聊天机器人在旅游领域能够有效地提供相关且连贯的响应，突显了专门化的人工智能应用程序在增值服务和提升特定市场用户体验方面的潜力。 

---
# Cross-Linguistic Examination of Machine Translation Transfer Learning 

**Title (ZH)**: 跨语言领域的机器翻译迁移学习研究 

**Authors**: Saughmon Boujkian  

**Link**: [PDF](https://arxiv.org/pdf/2501.00045)  

**Abstract**: This study investigates the effectiveness of transfer learning in machine translation across diverse linguistic families by evaluating five distinct language pairs. Leveraging pre-trained models on high-resource languages, these models were fine-tuned on low-resource languages, examining variations in hyperparameters such as learning rate, batch size, number of epochs, and weight decay. The research encompasses language pairs from different linguistic backgrounds: Semitic (Modern Standard Arabic - Levantine Arabic), Bantu (Hausa - Zulu), Romance (Spanish - Catalan), Slavic (Slovakian - Macedonian), and language isolates (Eastern Armenian - Western Armenian). Results demonstrate that transfer learning is effective across different language families, although the impact of hyperparameters varies. A moderate batch size (e.g., 32) is generally more effective, while very high learning rates can disrupt model training. The study highlights the universality of transfer learning in multilingual contexts and suggests that consistent hyperparameter settings can simplify and enhance the efficiency of multilingual model training. 

**Abstract (ZH)**: 本研究通过评估五个不同的语言对，考察了跨多样语言家族的机器翻译中迁移学习的有效性。利用高资源语言的预训练模型，并对低资源语言进行微调，研究了学习率、批量大小、训练轮数和权重衰减等超参数的差异性影响。研究涵盖了来自不同语言背景的语言对：闪语系（现代标准阿拉伯语-黎凡特阿拉伯语）、班图语系（豪萨语-祖鲁语）、罗曼语系（西班牙语-加泰罗尼亚语）、斯拉夫语系（斯洛伐克语-马其顿语）和孤立语系（东方亚美尼亚语-西方亚美尼亚语）。研究结果表明，迁移学习在不同语言家族中都是有效的，尽管超参数的影响有所不同。通常来说，中等批量大小（例如，32）更为有效，而非常高的学习率可能会干扰模型的训练。本研究强调了在多语言背景下迁移学习的普遍性，并表明一致的超参数设置可以简化和提高多语言模型训练的效率。 

---
# Distilling Large Language Models for Efficient Clinical Information Extraction 

**Title (ZH)**: 高效临床信息提取的大语言模型精简方法 

**Authors**: Karthik S. Vedula, Annika Gupta, Akshay Swaminathan, Ivan Lopez, Suhana Bedi, Nigam H. Shah  

**Link**: [PDF](https://arxiv.org/pdf/2501.00031)  

**Abstract**: Large language models (LLMs) excel at clinical information extraction but their computational demands limit practical deployment. Knowledge distillation--the process of transferring knowledge from larger to smaller models--offers a potential solution. We evaluate the performance of distilled BERT models, which are approximately 1,000 times smaller than modern LLMs, for clinical named entity recognition (NER) tasks. We leveraged state-of-the-art LLMs (Gemini and OpenAI models) and medical ontologies (RxNorm and SNOMED) as teacher labelers for medication, disease, and symptom extraction. We applied our approach to over 3,300 clinical notes spanning five publicly available datasets, comparing distilled BERT models against both their teacher labelers and BERT models fine-tuned on human labels. External validation was conducted using clinical notes from the MedAlign dataset. For disease extraction, F1 scores were 0.82 (teacher model), 0.89 (BioBERT trained on human labels), and 0.84 (BioBERT-distilled). For medication, F1 scores were 0.84 (teacher model), 0.91 (BioBERT-human), and 0.87 (BioBERT-distilled). For symptoms: F1 score of 0.73 (teacher model) and 0.68 (BioBERT-distilled). Distilled BERT models had faster inference (12x, 4x, 8x faster than GPT-4o, o1-mini, and Gemini Flash respectively) and lower costs (85x, 101x, 2x cheaper than GPT-4o, o1-mini, and Gemini Flash respectively). On the external validation dataset, the distilled BERT model achieved F1 scores of 0.883 (medication), 0.726 (disease), and 0.699 (symptom). Distilled BERT models were up to 101x cheaper and 12x faster than state-of-the-art LLMs while achieving similar performance on NER tasks. Distillation offers a computationally efficient and scalable alternative to large LLMs for clinical information extraction. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在临床信息提取方面表现出色，但其计算需求限制了其实用部署。知识蒸馏——即将大模型的知识转移到小模型的过程——提供了一种潜在的解决方案。我们评估了蒸馏后的BERT模型在临床命名实体识别（NER）任务中的性能，这些模型大约比现代LLMs小1,000倍。我们利用最先进的LLMs（Gemini和OpenAI模型）和医学本体（RxNorm和SNOMED）作为教师标签器，对药物、疾病和症状进行提取。我们将这种方法应用于超过3,300份临床笔记，并跨越五个公开可用的数据集，将蒸馏后的BERT模型与教师标签器和基于人类标签微调的BERT模型进行了对比。外部验证是在MedAlign数据集的临床笔记中进行的。对于疾病提取，F1分数分别为0.82（教师模型）、0.89（BioBERT基于人类标签训练）和0.84（蒸馏后的BioBERT）。对于药物提取，F1分数分别为0.84（教师模型）、0.91（BioBERT基于人类标签训练）和0.87（蒸馏后的BioBERT）。对于症状提取，F1分数分别为0.73（教师模型）和0.68（蒸馏后的BioBERT）。蒸馏后的BERT模型具有更快的推理速度（分别快于GPT-4o、o1-mini和Gemini Flash的12倍、4倍和8倍），并具有更低的成本（分别比GPT-4o、o1-mini和Gemini Flash便宜85倍、101倍和2倍）。在外部分区验证数据集中，蒸馏后的BERT模型的F1分数分别为0.883（药物）、0.726（疾病）和0.699（症状）。蒸馏后的BERT模型在成本和推理速度上分别比最先进的LLMs便宜101倍和12倍，同时在命名实体识别任务上实现了相似的性能。知识蒸馏为临床信息提取提供了计算有效且可扩展的替代方案，与大规模LLMs相比。 

---
# Underutilization of Syntactic Processing by Chinese Learners of English in Comprehending English Sentences, Evidenced from Adapted Garden-Path Ambiguity Experiment 

**Title (ZH)**: 中文翻译如下，符合学术规范：

汉语学习者在理解英语句子时对句法处理的利用不足：基于修改后的花园路径歧义实验的证据 

**Authors**: Jiapeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2501.00030)  

**Abstract**: Many studies have revealed that sentence comprehension relies more on semantic processing than on syntactic processing. However, previous studies have predominantly emphasized the preference for semantic processing, focusing on the semantic perspective. In contrast, this current study highlights the under-utilization of syntactic processing, from a syntactic perspective. Based on the traditional garden-path experiment, which involves locally ambiguous but globally unambiguous sentences, this study's empirical experiment innovatively crafted an adapted version featuring semantically ambiguous but syntactically unambiguous sentences to meet its specific research objective. This experiment, involving 140 subjects, demonstrates through descriptive and inferential statistical analyses using SPSS, Graph Pad Prism, and Cursor that Chinese learners of English tend to under-utilize syntactic processing when comprehending English sentences. The study identifies two types of parsing under-utilization: partial and complete. Further exploration reveals that trial and error in syntactic processing contributes to both. Consequently, this study lays a foundation for the development of a novel parsing method designed to fully integrate syntactic processing into sentence comprehension, thereby enhancing the level of English sentence comprehension for Chinese learners of English. 

**Abstract (ZH)**: 许多研究表明，句子理解更多依赖于语义处理而不是句法处理。然而，以往的研究主要强调了语义处理的重要性，主要从语义的角度进行了探讨。相比之下，本研究则重点关注句法处理的低利用性，从句法的角度进行了探讨。基于传统的花园路径实验方法，该实验创新性地设计了一种新版本，采用了语义上模糊但句法上不模糊的句子，以满足其特定的研究目的。该实验涉及140名被试者，并通过SPSS、Graph Pad Prism和Cursor进行描述性和推断性统计分析，表明中国英语学习者在理解英语句子时，对句法处理的利用不足。本研究识别了两种类型的句法处理利用不足：部分利用不足和完全利用不足。进一步的探索发现，句法处理过程中的试错是这两种情况的共同原因。因此，本研究为开发一种新型的句法整合方法奠定了基础，旨在全面将句法处理融入句子理解中，从而提高中国英语学习者英语句子理解的水平。 

---
# A Breadth-First Catalog of Text Processing, Speech Processing and Multimodal Research in South Asian Languages 

**Title (ZH)**: 南亚语言文本处理、语音处理及多模态研究的广度优先目录 

**Authors**: Pranav Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2501.00029)  

**Abstract**: We review the recent literature (January 2022- October 2024) in South Asian languages on text-based language processing, multimodal models, and speech processing, and provide a spotlight analysis focused on 21 low-resource South Asian languages, namely Saraiki, Assamese, Balochi, Bhojpuri, Bodo, Burmese, Chhattisgarhi, Dhivehi, Gujarati, Kannada, Kashmiri, Konkani, Khasi, Malayalam, Meitei, Nepali, Odia, Pashto, Rajasthani, Sindhi, and Telugu. We identify trends, challenges, and future research directions, using a step-wise approach that incorporates relevance classification and clustering based on large language models (LLMs). Our goal is to provide a breadth-first overview of the recent developments in South Asian language technologies to NLP researchers interested in working with South Asian languages. 

**Abstract (ZH)**: 我们回顾了2022年1月至2024年10月期间南亚语言中关于基于文本的语言处理、多模态模型和语音处理的最新文献，并针对21种低资源南亚语言（即 Saraiki、Assamese、Balochi、Bhojpuri、Bodo、Burmese、Chhattisgarhi、Dhivehi、Gujarati、Kannada、Kashmiri、Konkani、Khasi、Malayalam、Meitei、Nepali、Odia、Pashto、Rajasthani、Sindhi 和 Telugu）进行了重点分析。我们通过结合大型语言模型（LLM）进行的相关性分类和聚类，识别出趋势、挑战和未来研究方向。我们的目标是为对南亚语言感兴趣并致力于南亚语言技术研究的自然语言处理（NLP）研究人员提供一个全面的近期发展概述。 

---
# Unifying Specialized Visual Encoders for Video Language Models 

**Title (ZH)**: 统一专门视觉编码器以构建视频语言模型 

**Authors**: Jihoon Chung, Tyler Zhu, Max Gonzalez Saez-Diez, Juan Carlos Niebles, Honglu Zhou, Olga Russakovsky  

**Link**: [PDF](https://arxiv.org/pdf/2501.01426)  

**Abstract**: The recent advent of Large Language Models (LLMs) has ushered sophisticated reasoning capabilities into the realm of video through Video Large Language Models (VideoLLMs). However, VideoLLMs currently rely on a single vision encoder for all of their visual processing, which limits the amount and type of visual information that can be conveyed to the LLM. Our method, MERV, Multi-Encoder Representation of Videos, instead leverages multiple frozen visual encoders to create a unified representation of a video, providing the VideoLLM with a comprehensive set of specialized visual knowledge. Spatio-temporally aligning the features from each encoder allows us to tackle a wider range of open-ended and multiple-choice video understanding questions and outperform prior state-of-the-art works. MERV is up to 3.7% better in accuracy than Video-LLaVA across the standard suite video understanding benchmarks, while also having a better Video-ChatGPT score. We also improve upon SeViLA, the previous best on zero-shot Perception Test accuracy, by 2.2%. MERV introduces minimal extra parameters and trains faster than equivalent single-encoder methods while parallelizing the visual processing. Finally, we provide qualitative evidence that MERV successfully captures domain knowledge from each of its encoders. Our results offer promising directions in utilizing multiple vision encoders for comprehensive video understanding. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）的兴起为视频领域带来了复杂的推理能力，通过Video Large Language Models（VideoLLMs）实现了这一目标。然而，目前VideoLLMs在视觉处理方面仍然依赖单一的视觉编码器，这限制了传递给LLM的视觉信息的数量和类型。我们提出的方法，即MERV（Multi-Encoder Representation of Videos），采用多个冻结的视觉编码器来创建视频的统一表示，为VideoLLM提供了一套全面的专业视觉知识。通过时空对齐每个编码器的特征，MERV能够应对更加开放和多元的视频理解问题，并在先前的先进工作基础上取得了更好的性能。与Video-LLaVA相比，MERV在标准视频理解基准上的准确率提高了3.7%，同时具有更好的Video-ChatGPT评分。我们在零样本感知测试中的准确率上也超越了此前最佳的SeViLA，提高了2.2%。MERV引入了极少的额外参数，训练速度也快于同等单编码器方法，同时实现了视觉处理的并行化。最后，我们提供了定性的证据，证明MERV成功地从每个编码器中捕获了领域知识。我们的研究结果为利用多个视觉编码器进行全面视频理解提供了令人鼓舞的方向。 

---
# Embedding-based Approaches to Hyperpartisan News Detection 

**Title (ZH)**: 基于嵌入的方法在虚假两极化新闻检测中的应用 

**Authors**: Karthik Mohan, Pengyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.01370)  

**Abstract**: In this paper, we describe our systems in which the objective is to determine whether a given news article could be considered as hyperpartisan. Hyperpartisan news is news that takes an extremely polarized political standpoint with an intention of creating political divide among the public. We attempted several approaches, including n-grams, sentiment analysis, as well as sentence and document representation using pre-tained ELMo. Our best system using pre-trained ELMo with Bidirectional LSTM achieved an accuracy of 83% through 10-fold cross-validation without much hyperparameter tuning. 

**Abstract (ZH)**: 在本文中，我们描述了一种系统，其目标是确定给定的新闻文章是否可以被视为过度党派化（hyperpartisan）。过度党派化新闻指的是那些持有极其极化政治立场，并有意制造公众政治分裂的新闻。我们尝试了多种方法，包括n元组分析、情感分析，以及利用预训练ELMo表示句子和文档的方法。通过10折交叉验证，我们使用预训练ELMo与双向LSTM的最佳系统在未经大量超参数调整的情况下取得了83%的准确率。 

---
# ViGiL3D: A Linguistically Diverse Dataset for 3D Visual Grounding 

**Title (ZH)**: ViGiL3D: 一种语言多样性的三维视觉定位数据集 

**Authors**: Austin T. Wang, ZeMing Gong, Angel X. Chang  

**Link**: [PDF](https://arxiv.org/pdf/2501.01366)  

**Abstract**: 3D visual grounding (3DVG) involves localizing entities in a 3D scene referred to by natural language text. Such models are useful for embodied AI and scene retrieval applications, which involve searching for objects or patterns using natural language descriptions. While recent works have focused on LLM-based scaling of 3DVG datasets, these datasets do not capture the full range of potential prompts which could be specified in the English language. To ensure that we are scaling up and testing against a useful and representative set of prompts, we propose a framework for linguistically analyzing 3DVG prompts and introduce Visual Grounding with Diverse Language in 3D (ViGiL3D), a diagnostic dataset for evaluating visual grounding methods against a diverse set of language patterns. We evaluate existing open-vocabulary 3DVG methods to demonstrate that these methods are not yet proficient in understanding and identifying the targets of more challenging, out-of-distribution prompts, toward real-world applications. 

**Abstract (ZH)**: 3D视觉定位（3DVG）涉及将由自然语言文本引用的实体在3D场景中进行定位。这种模型对于自主AI和场景检索应用非常有用，这些应用涉及使用自然语言描述来搜索物体或模式。虽然近期的研究主要集中在使用大型语言模型（LLM）扩大3DVG数据集的规模上，但这些数据集未能涵盖英语中可能指定的全部范围的提示。为了确保我们在扩大并针对一个实用且具有代表性的提示集进行测试，我们提出了一种对3DVG提示进行语言分析的框架，并推出了Visual Grounding with Diverse Language in 3D（ViGiL3D），这是一个用于评估视觉定位方法的诊断数据集，以应对多样化的语言模式。我们评估现有的开放式词汇3DVG方法，以证明这些方法尚无法熟练理解并识别更具有挑战性和分布外的提示，以适应实际应用。 

---
# AdaptVC: High Quality Voice Conversion with Adaptive Learning 

**Title (ZH)**: AdaptVC：具有自适应学习的高质量语音转换 

**Authors**: Jaehun Kim, Ji-Hoon Kim, Yeunju Choi, Tan Dat Nguyen, Seongkyu Mun, Joon Son Chung  

**Link**: [PDF](https://arxiv.org/pdf/2501.01347)  

**Abstract**: The goal of voice conversion is to transform the speech of a source speaker to sound like that of a reference speaker while preserving the original content. A key challenge is to extract disentangled linguistic content from the source and voice style from the reference. While existing approaches leverage various methods to isolate the two, a generalization still requires further attention, especially for robustness in zero-shot scenarios. In this paper, we achieve successful disentanglement of content and speaker features by tuning self-supervised speech features with adapters. The adapters are trained to dynamically encode nuanced features from rich self-supervised features, and the decoder fuses them to produce speech that accurately resembles the reference with minimal loss of content. Moreover, we leverage a conditional flow matching decoder with cross-attention speaker conditioning to further boost the synthesis quality and efficiency. Subjective and objective evaluations in a zero-shot scenario demonstrate that the proposed method outperforms existing models in speech quality and similarity to the reference speech. 

**Abstract (ZH)**: 语音转换的目标是在保留原始内容的同时，将源说话人的语音转换为听起来像参考说话人的语音。一个关键挑战是从源说话人中提取独立的语言内容，并从参考说话人中提取语音风格。尽管现有的方法在隔离这两个方面采用了各种方法，但在鲁棒性方面特别是在零样本场景中，这一任务仍需进一步关注。在本文中，我们通过调整自监督语音特征与适配器来成功地解耦内容特征和说话人特征。适配器被训练为动态编码丰富的自监督特征中的细微特征，并将这些特征与解码器融合，以生成与参考语音相似的语音，同时最大限度地减少内容损失。此外，我们使用条件流匹配解码器并结合跨注意机制的说话人条件，进一步提高了合成质量和效率。在零样本场景中的主观和客观评估表明，所提出的方法在语音质量和与参考语音的相似度方面优于现有模型。 

---
# Large Vision-Language Model Alignment and Misalignment: A Survey Through the Lens of Explainability 

**Title (ZH)**: 大规模多模态模型的一致性与分歧：基于可解释性视角的综述 

**Authors**: Dong Shu, Haiyan Zhao, Jingyu Hu, Weiru Liu, Lu Cheng, Mengnan Du  

**Link**: [PDF](https://arxiv.org/pdf/2501.01346)  

**Abstract**: Large Vision-Language Models (LVLMs) have demonstrated remarkable capabilities in processing both visual and textual information. However, the critical challenge of alignment between visual and linguistic representations is not fully understood. This survey presents a comprehensive examination of alignment and misalignment in LVLMs through an explainability lens. We first examine the fundamentals of alignment, exploring its representational and behavioral aspects, training methodologies, and theoretical foundations. We then analyze misalignment phenomena across three semantic levels: object, attribute, and relational misalignment. Our investigation reveals that misalignment emerges from challenges at multiple levels: the data level, the model level, and the inference level. We provide a comprehensive review of existing mitigation strategies, categorizing them into parameter-frozen and parameter-tuning approaches. Finally, we outline promising future research directions, emphasizing the need for standardized evaluation protocols and in-depth explainability studies. 

**Abstract (ZH)**: 大型多模态语言视觉模型（LVLMs）在处理视觉和文本信息方面展现了卓越的能力。然而，视觉和语言表示之间的对齐问题尚未完全理解。本文综述通过解释性视角全面探讨了LVLMs中的对齐与不对齐问题。我们首先探讨了对齐的基本原理，包括其表示和行为方面的特点、训练方法以及理论基础。随后，我们将分析从物体、属性到关系三个语义层次的不对齐现象。我们的研究揭示出，不对齐现象源自多个层面的挑战：数据层面、模型层面和推理层面。我们提供了一个对现有缓解策略的全面回顾，并将这些策略分为参数冻结和参数调优两类。最后，我们指出了未来研究有希望的方向，强调需要标准化的评估协议和深入的解释性研究。 

---
# The Prompt Alchemist: Automated LLM-Tailored Prompt Optimization for Test Case Generation 

**Title (ZH)**: 奇妙的提示炼金师：针对测试案例生成的自动调优提示优化方法 

**Authors**: Shuzheng Gao, Chaozheng Wang, Cuiyun Gao, Xiaoqian Jiao, Chun Yong Chong, Shan Gao, Michael Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2501.01329)  

**Abstract**: Test cases are essential for validating the reliability and quality of software applications. Recent studies have demonstrated the capability of Large Language Models (LLMs) to generate useful test cases for given source code. However, the existing work primarily relies on human-written plain prompts, which often leads to suboptimal results since the performance of LLMs can be highly influenced by the prompts. Moreover, these approaches use the same prompt for all LLMs, overlooking the fact that different LLMs might be best suited to different prompts. Given the wide variety of possible prompt formulations, automatically discovering the optimal prompt for each LLM presents a significant challenge. Although there are methods on automated prompt optimization in the natural language processing field, they are hard to produce effective prompts for the test case generation task. First, the methods iteratively optimize prompts by simply combining and mutating existing ones without proper guidance, resulting in prompts that lack diversity and tend to repeat the same errors in the generated test cases. Second, the prompts are generally lack of domain contextual knowledge, limiting LLMs' performance in the task. 

**Abstract (ZH)**: 测试用例对于验证软件应用的可靠性和质量至关重要。近期研究证明，大型语言模型（LLMs）能够生成适用于给定源代码的有效测试用例。然而，现有工作主要依赖于人工编写的简明提示，这往往导致结果不尽如人意，因为LLMs的表现高度依赖于提示的质量。此外，这些方法使用相同的提示来针对所有的LLMs，忽视了不同的LLMs可能最适合不同提示的事实。鉴于提示的各种可能形式非常多样，自动发现每个LLMs的最佳提示是一个重大挑战。尽管自然语言处理领域存在一些自动提示优化的方法，但它们难以为测试用例生成任务生成有效的提示。首先，这些方法通过简单的组合和变异现有提示进行迭代优化，缺乏适当的指导，导致生成的提示缺乏多样性，且倾向于在生成的测试用例中重复相同的错误。其次，这些提示通常缺乏领域上下文知识，限制了LLMs在该任务中的表现。 

---
# CultureVLM: Characterizing and Improving Cultural Understanding of Vision-Language Models for over 100 Countries 

**Title (ZH)**: CultureVLM： characterizing and improving the cultural understanding of vision-language models across more than 100 countries

或者更加正式的表达方式：

CultureVLM：刻画并提升跨逾100个国家的视觉-语言模型文化理解能力 

**Authors**: Shudong Liu, Yiqiao Jin, Cheng Li, Derek F. Wong, Qingsong Wen, Lichao Sun, Haipeng Chen, Xing Xie, Jindong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.01282)  

**Abstract**: Vision-language models (VLMs) have advanced human-AI interaction but struggle with cultural understanding, often misinterpreting symbols, gestures, and artifacts due to biases in predominantly Western-centric training data. In this paper, we construct CultureVerse, a large-scale multimodal benchmark covering 19, 682 cultural concepts, 188 countries/regions, 15 cultural concepts, and 3 question types, with the aim of characterizing and improving VLMs' multicultural understanding capabilities. Then, we propose CultureVLM, a series of VLMs fine-tuned on our dataset to achieve significant performance improvement in cultural understanding. Our evaluation of 16 models reveals significant disparities, with a stronger performance in Western concepts and weaker results in African and Asian contexts. Fine-tuning on our CultureVerse enhances cultural perception, demonstrating cross-cultural, cross-continent, and cross-dataset generalization without sacrificing performance on models' general VLM benchmarks. We further present insights on cultural generalization and forgetting. We hope that this work could lay the foundation for more equitable and culturally aware multimodal AI systems. 

**Abstract (ZH)**: 视觉-语言模型（VLMs）促进了人机交互的发展，但在文化理解方面存在挑战，常常由于训练数据主要以西方为中心而误解符号、手势和文物。在本文中，我们构建了CultureVerse，这是一个大规模的多模态基准，涵盖了19,682个文化概念、188个国家/地区、15个文化概念和3种问题类型，并旨在表征并提升VLMs的跨文化理解能力。随后，我们提出了CultureVLM，这是一个系列的VLMs，在我们的数据集上进行微调，实现了文化理解方面的显著性能改进。我们对16个模型的评估揭示了显著的差异，西方概念的性能较强，而非洲和亚洲的性能较弱。在我们的CultureVerse上进行微调可以增强文化感知能力，证明了跨文化、跨大陆和跨数据集的一般性，而不会牺牲模型在一般VLM基准上的性能。此外，我们还呈现了关于文化一般性和遗忘的洞见。我们希望这项工作能够为更加公平和文化意识更强的多模态AI系统奠定基础。 

---
# Face-Human-Bench: A Comprehensive Benchmark of Face and Human Understanding for Multi-modal Assistants 

**Title (ZH)**: 面部-人类-台架：多模态辅助系统中面部和人类理解的综合性基准测试 

**Authors**: Lixiong Qin, Shilong Ou, Miaoxuan Zhang, Jiangning Wei, Yuhang Zhang, Xiaoshuai Song, Yuchen Liu, Mei Wang, Weiran Xu  

**Link**: [PDF](https://arxiv.org/pdf/2501.01243)  

**Abstract**: Faces and humans are crucial elements in social interaction and are widely included in everyday photos and videos. Therefore, a deep understanding of faces and humans will enable multi-modal assistants to achieve improved response quality and broadened application scope. Currently, the multi-modal assistant community lacks a comprehensive and scientific evaluation of face and human understanding abilities. In this paper, we first propose a hierarchical ability taxonomy that includes three levels of abilities. Then, based on this taxonomy, we collect images and annotations from publicly available datasets in the face and human community and build a semi-automatic data pipeline to produce problems for the new benchmark. Finally, the obtained Face-Human-Bench comprises a development set with 900 problems and a test set with 1800 problems, supporting both English and Chinese. We conduct evaluations over 25 mainstream multi-modal large language models (MLLMs) with our Face-Human-Bench, focusing on the correlation between abilities, the impact of the relative position of targets on performance, and the impact of Chain of Thought (CoT) prompting on performance. Moreover, inspired by multi-modal agents, we also explore which abilities of MLLMs need to be supplemented by specialist models. 

**Abstract (ZH)**: 面部和人类是社会互动中的关键元素，广泛出现在日常照片和视频中。因此，对面部和人类的深入理解将使多模态辅助系统能够提高回应质量并扩展应用范围。目前，多模态辅助社区缺乏对面部和人类理解能力的综合和科学评估。在本文中，我们首先提出了一种分层能力分类体系，包括三个层级的能力。然后，基于此分类体系，我们从面部和人类社区的公开数据集中收集图像和注释，并构建了一种半自动数据管道以生成新的基准测试的问题集。最终，获得的Face-Human-Bench 包含一个包含900个问题的开发集和一个包含1800个问题的测试集，支持英文和中文。我们使用Face-Human-Bench 对25种主流多模态大型语言模型（MLLMs）进行了评估，重点考察了能力之间的相关性、目标相对位置对性能的影响以及Chain of Thought（CoT）提示对性能的影响。此外，受到多模态代理的启发，我们还探讨了哪些多模态大型语言模型的能力需要通过专家模型进行补充。 

---
# Harnessing Multi-Agent LLMs for Complex Engineering Problem-Solving: A Framework for Senior Design Projects 

**Title (ZH)**: 利用多代理大型语言模型解决复杂工程问题：高级设计项目框架 

**Authors**: Abdullah Mushtaq, Muhammad Rafay Naeem, Ibrahim Ghaznavi, Muhammad Imran Taj, Imran Hashmi, Junaid Qadir  

**Link**: [PDF](https://arxiv.org/pdf/2501.01205)  

**Abstract**: Multi-Agent Large Language Models (LLMs) are gaining significant attention for their ability to harness collective intelligence in complex problem-solving, decision-making, and planning tasks. This aligns with the concept of the wisdom of crowds, where diverse agents contribute collectively to generating effective solutions, making it particularly suitable for educational settings. Senior design projects, also known as capstone or final year projects, are pivotal in engineering education as they integrate theoretical knowledge with practical application, fostering critical thinking, teamwork, and real-world problem-solving skills. In this paper, we explore the use of Multi-Agent LLMs in supporting these senior design projects undertaken by engineering students, which often involve multidisciplinary considerations and conflicting objectives, such as optimizing technical performance while addressing ethical, social, and environmental concerns. We propose a framework where distinct LLM agents represent different expert perspectives, such as problem formulation agents, system complexity agents, societal and ethical agents, or project managers, thus facilitating a holistic problem-solving approach. This implementation leverages standard multi-agent system (MAS) concepts such as coordination, cooperation, and negotiation, incorporating prompt engineering to develop diverse personas for each agent. These agents engage in rich, collaborative dialogues to simulate human engineering teams, guided by principles from swarm AI to efficiently balance individual contributions towards a unified solution. We adapt these techniques to create a collaboration structure for LLM agents, encouraging interdisciplinary reasoning and negotiation similar to real-world senior design projects. To assess the efficacy of this framework, we collected six proposals of engineering and computer science of... 

**Abstract (ZH)**: 多智能体大型语言模型（Multi-Agent Large Language Models, MALLMs）因其在复杂问题解决、决策制定和规划任务中的能力而获得了广泛关注，这种能力与群体智慧的概念相契合。群体智慧概念指的是，各种不同的个体共同努力，产生有效的解决方案，这特别适用于教育环境。在工程教育中，高级设计项目（Senior Design Projects，也称为毕业设计或毕业综合项目）非常重要，因为它们将理论知识与实践应用相结合，培养批判性思维、团队合作以及解决实际问题的能力。在本文中，我们探讨了在进行这些通常涉及多学科考虑和冲突目标（如优化技术性能同时解决伦理、社会和环境问题）的高级设计项目时，如何利用多智能体大型语言模型进行支持。我们提出了一种框架，其中不同的LLM代理代表不同的专家视角，如问题表述代理、系统复杂性代理、社会与伦理代理或项目经理，从而促进整体的问题解决方法。该实现利用了多智能体系统（Multi-Agent System, MAS）的基本概念，如协作、合作和谈判，并通过提示工程开发出各种各样的代理个性。这些代理通过富有成效的协作对话来模拟人类工程团队，受群智AI原则的指导，能够高效地平衡个体贡献以形成统一的解决方案。我们采用这些技术来创建LLM代理之间的协作框架，促进跨学科的推理和谈判，这类似于现实世界中的高级设计项目。为了评估该框架的有效性，我们收集了六个来自工程和计算机科学领域的... 

---
# MuQ: Self-Supervised Music Representation Learning with Mel Residual Vector Quantization 

**Title (ZH)**: MuQ：基于Mel残差向量量化的心自我监督音乐表示学习 

**Authors**: Haina Zhu, Yizhi Zhou, Hangting Chen, Jianwei Yu, Ziyang Ma, Rongzhi Gu, Wei Tan, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.01108)  

**Abstract**: Recent years have witnessed the success of foundation models pre-trained with self-supervised learning (SSL) in various music informatics understanding tasks, including music tagging, instrument classification, key detection, and more. In this paper, we propose a self-supervised music representation learning model for music understanding. Distinguished from previous studies adopting random projection or existing neural codec, the proposed model, named MuQ, is trained to predict tokens generated by Mel Residual Vector Quantization (Mel-RVQ). Our Mel-RVQ utilizes residual linear projection structure for Mel spectrum quantization to enhance the stability and efficiency of target extraction and lead to better performance. Experiments in a large variety of downstream tasks demonstrate that MuQ outperforms previous self-supervised music representation models with only 0.9K hours of open-source pre-training data. Scaling up the data to over 160K hours and adopting iterative training consistently improve the model performance. To further validate the strength of our model, we present MuQ-MuLan, a joint music-text embedding model based on contrastive learning, which achieves state-of-the-art performance in the zero-shot music tagging task on the MagnaTagATune dataset. Code and checkpoints are open source in this https URL. 

**Abstract (ZH)**: 近年来，通过自我监督学习（SSL）进行预训练的基础模型在多种音乐信息理解任务中取得了成功，包括音乐标记、乐器分类、调性检测等。本文提出了一种用于音乐理解的自我监督音乐表示学习模型。与以往采用随机投影或现有神经编解码器的研究不同，所提出的模型MuQ经过训练，能够预测由Mel残差矢量量化（Mel-RVQ）生成的令牌。我们的Mel-RVQ利用残差线性投影结构进行梅尔频谱量化，以增强目标提取的稳定性和效率，并提高性能。在各种下游任务中的实验表明，MuQ仅使用0.9K小时的开源预训练数据便超越了先前的自我监督音乐表示模型。通过增加数据量至超过160K小时并采用迭代训练，可以持续提高模型性能。为了进一步验证我们模型的优势，我们基于对比学习提出了MuQ-MuLan音乐-文本联合嵌入模型，在MagnaTagATune数据集的零样本音乐标记任务中实现了最佳性能。本文的相关代码和检查点已开源，可在以下链接访问：https://github.com/... 

---
# 2.5 Years in Class: A Multimodal Textbook for Vision-Language Pretraining 

**Title (ZH)**: 2.5年在校时间：一种用于视觉-语言预训练的多模态教材 

**Authors**: Wenqi Zhang, Hang Zhang, Xin Li, Jiashuo Sun, Yongliang Shen, Weiming Lu, Deli Zhao, Yueting Zhuang, Lidong Bing  

**Link**: [PDF](https://arxiv.org/pdf/2501.00958)  

**Abstract**: Compared to image-text pair data, interleaved corpora enable Vision-Language Models (VLMs) to understand the world more naturally like humans. However, such existing datasets are crawled from webpage, facing challenges like low knowledge density, loose image-text relations, and poor logical coherence between images. On the other hand, the internet hosts vast instructional videos (e.g., online geometry courses) that are widely used by humans to learn foundational subjects, yet these valuable resources remain underexplored in VLM training. In this paper, we introduce a high-quality \textbf{multimodal textbook} corpus with richer foundational knowledge for VLM pretraining. It collects over 2.5 years of instructional videos, totaling 22,000 class hours. We first use an LLM-proposed taxonomy to systematically gather instructional videos. Then we progressively extract and refine visual (keyframes), audio (ASR), and textual knowledge (OCR) from the videos, and organize as an image-text interleaved corpus based on temporal order. Compared to its counterparts, our video-centric textbook offers more coherent context, richer knowledge, and better image-text alignment. Experiments demonstrate its superb pretraining performance, particularly in knowledge- and reasoning-intensive tasks like ScienceQA and MathVista. Moreover, VLMs pre-trained on our textbook exhibit outstanding interleaved context awareness, leveraging visual and textual cues in their few-shot context for task solving~\footnote{Our code are available at \url{this https URL}}. 

**Abstract (ZH)**: 与图像-文本对数据相比，交错语料库使视觉-语言模型（VLMs）能够更自然地理解世界，类似于人类的理解方式。然而，现有的这类数据集是从网页上爬取的，面临着知识密度低、图像-文本关系松散和图像间的逻辑连贯性差等挑战。另一方面，互联网上存在大量的教学视频（例如，网上几何课程），这些视频在人类学习基础科目时被广泛使用，但这些宝贵资源在VLM训练中尚未得到充分挖掘。在本文中，我们介绍了一个高质量的多模态教材语料库，它为VLM预训练提供了更加丰富的基础知识。该语料库收集了超过两年半的教学视频，总计22,000课时。我们首先使用一个语言模型（LLM）提出的分类体系，系统地收集教学视频。然后，我们逐步从视频中提取和精炼视觉（关键帧）、音频（语音识别）和文本知识（光学字符识别），并根据时间顺序组织成交错的图像-文本语料库。与同类数据集相比，我们以视频为中心的教材提供了更为连贯的上下文、更丰富的内容以及更好的图像-文本对齐。实验结果表明，该语料库在科学问答（ScienceQA）和数学视野（MathVista）等知识密集和推理密集的任务上的预训练性能尤为出色。此外，基于我们教材预训练的VLM在处理任务时能够更好地利用视觉和文本提示来理解交错的上下文（注释\footnote{我们的代码可在\url{这个链接}下载}）。 

---
# Aligning Netlist to Source Code using SynAlign 

**Title (ZH)**: 使用SynAlign对网表进行源代码对齐 

**Authors**: Sakshi Garg, Jose Renau  

**Link**: [PDF](https://arxiv.org/pdf/2501.00921)  

**Abstract**: In current chip design processes, using multiple tools to obtain a gate-level netlist often results in the loss of source code correlation. SynAlign addresses this challenge by automating the alignment process, simplifying iterative design, reducing overhead, and maintaining correlation across various tools. This enhances the efficiency and effectiveness of chip design workflows.
Improving characteristics such as frequency through iterative design is essential for enhancing accelerators and chip designs. While synthesis tools produce netlists with critical path information, designers often lack the tools to trace these netlist cells back to their original source code. Mapping netlist components to source code provides early feedback on timing and power for frontend designers.
SynAlign automatically aligns post-optimized netlists with the original source code without altering compilers or synthesis processes. Its alignment strategy relies on the consistent design structure throughout the chip design cycle, even with changes in compiler flow. This consistency allows engineers to maintain a correlation between modified designs and the original source code across various tools. Remarkably, SynAlign can tolerate up to 61\% design net changes without impacting alignment accuracy. 

**Abstract (ZH)**: 在当前的芯片设计流程中，使用多种工具生成门级网表通常会导致源代码相关性的丢失。SynAlign 通过自动化对齐过程，简化迭代设计，减少开销，并在不同工具之间保持相关性，从而提高了芯片设计工作流的效率和效果。

通过迭代设计提高频率等特性是增强加速器和芯片设计的关键。虽然综合工具生成包含关键路径信息的网表，但设计者常常缺乏将网表单元回溯到原始源代码的工具。将网表组件映射到源代码可以为前端设计者提供早期的时间和功耗反馈。

SynAlign 自动对齐优化后的网表与原始源代码，而不修改编译器或综合流程。其对齐策略依赖于整个芯片设计周期中一致的设计结构，即使编译流程发生变化也是如此。这种一致性使得工程师能够在不同工具之间维护修改设计与原始源代码之间的相关性。令人remarkably的是，SynAlign 能够容忍高达 61% 的设计网表变化而不影响对齐准确性。 

---
# AutoPresent: Designing Structured Visuals from Scratch 

**Title (ZH)**: AutoPresent: 从头设计结构化可视化 

**Authors**: Jiaxin Ge, Zora Zhiruo Wang, Xuhui Zhou, Yi-Hao Peng, Sanjay Subramanian, Qinyue Tan, Maarten Sap, Alane Suhr, Daniel Fried, Graham Neubig, Trevor Darrell  

**Link**: [PDF](https://arxiv.org/pdf/2501.00912)  

**Abstract**: Designing structured visuals such as presentation slides is essential for communicative needs, necessitating both content creation and visual planning skills. In this work, we tackle the challenge of automated slide generation, where models produce slide presentations from natural language (NL) instructions. We first introduce the SlidesBench benchmark, the first benchmark for slide generation with 7k training and 585 testing examples derived from 310 slide decks across 10 domains. SlidesBench supports evaluations that are (i)reference-based to measure similarity to a target slide, and (ii)reference-free to measure the design quality of generated slides alone. We benchmark end-to-end image generation and program generation methods with a variety of models, and find that programmatic methods produce higher-quality slides in user-interactable formats. Built on the success of program generation, we create AutoPresent, an 8B Llama-based model trained on 7k pairs of instructions paired with code for slide generation, and achieve results comparable to the closed-source model GPT-4o. We further explore iterative design refinement where the model is tasked to self-refine its own output, and we found that this process improves the slide's quality. We hope that our work will provide a basis for future work on generating structured visuals. 

**Abstract (ZH)**: 设计结构化的视觉元素，如演示文稿，对于沟通需求至关重要，既需要内容创作能力，也需要视觉规划技能。在本工作中，我们面对从自然语言（NL）指令自动生成演示文稿的挑战。我们首先介绍了SlidesBench基准测试，这是首个用于演示文稿生成的基准测试，包含7000个训练样本和585个测试样本，这些样本来自310个包含10个领域内容的演示文稿集合。SlidesBench 支持两种评估方式：(i) 参考基于评估，用于衡量生成的幻灯片与目标幻灯片的相似度；(ii) 非参考评估，仅用于衡量生成幻灯片的设计质量。我们使用多种模型进行了端到端图像生成和程序生成方法的基准测试，发现程序生成方法能产出更高质量的可交互格式的幻灯片。基于程序生成方法的成功，我们创建了AutoPresent，一个基于8B参数的Llama模型，该模型在7000对指令与代码配对的数据集上进行训练，并实现了与专有模型GPT-4o 相媲美的结果。我们进一步探索了迭代设计细化，使模型能够自我细化其输出，发现这个过程可以提高幻灯片的质量。我们希望通过这项工作为未来生成结构化视觉的研究提供基础。 

---
# U-GIFT: Uncertainty-Guided Firewall for Toxic Speech in Few-Shot Scenario 

**Title (ZH)**: U-GIFT：面向少量样本场景的不确定性引导防火墙以应对有毒言论 

**Authors**: Jiaxin Song, Xinyu Wang, Yihao Wang, Yifan Tang, Ru Zhang, Jianyi Liu, Gongshen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.00907)  

**Abstract**: With the widespread use of social media, user-generated content has surged on online platforms. When such content includes hateful, abusive, offensive, or cyberbullying behavior, it is classified as toxic speech, posing a significant threat to the online ecosystem's integrity and safety. While manual content moderation is still prevalent, the overwhelming volume of content and the psychological strain on human moderators underscore the need for automated toxic speech detection. Previously proposed detection methods often rely on large annotated datasets; however, acquiring such datasets is both costly and challenging in practice. To address this issue, we propose an uncertainty-guided firewall for toxic speech in few-shot scenarios, U-GIFT, that utilizes self-training to enhance detection performance even when labeled data is limited. Specifically, U-GIFT combines active learning with Bayesian Neural Networks (BNNs) to automatically identify high-quality samples from unlabeled data, prioritizing the selection of pseudo-labels with higher confidence for training based on uncertainty estimates derived from model predictions. Extensive experiments demonstrate that U-GIFT significantly outperforms competitive baselines in few-shot detection scenarios. In the 5-shot setting, it achieves a 14.92\% performance improvement over the basic model. Importantly, U-GIFT is user-friendly and adaptable to various pre-trained language models (PLMs). It also exhibits robust performance in scenarios with sample imbalance and cross-domain settings, while showcasing strong generalization across various language applications. We believe that U-GIFT provides an efficient solution for few-shot toxic speech detection, offering substantial support for automated content moderation in cyberspace, thereby acting as a firewall to promote advancements in cybersecurity. 

**Abstract (ZH)**: 随着社交媒体的广泛使用，用户生成的内容在在线平台上急剧增加。当这些内容包含仇恨言论、辱骂内容、冒犯性言论或网络欺凌行为时，它们被归类为有毒言论，对在线生态系统的完整性和安全性构成了重大威胁。尽管手动内容审核仍然普遍，但内容量巨大和人类审核者的心理压力凸显了引入自动有毒言论检测系统的必要性。之前提出的一些检测方法通常依赖于大规模标注数据集；然而，在实践中获取这样的数据集既昂贵又具有挑战性。为了解决这一问题，我们提出了一种在少量标注样本情况下进行有毒言论检测的不确定性引导防火墙方法，即U-GIFT，该方法利用自我训练来增强检测性能，即使标注数据有限也是如此。具体而言，U-GIFT结合了主动学习和贝叶斯神经网络（BNNs），自动从未标注数据中识别高质量样本，并优先选择基于模型预测不确定性估计具有更高置信度的伪标签进行训练。实验结果表明，在少量标注样本的情境下，U-GIFT显著优于竞争基线方法。在5-shot设置中，相对于基础模型，其性能提高了14.92%。此外，U-GIFT操作简便且易于适应各种预训练语言模型（PLMs）。它还表现出在样本不平衡和跨域情境下的稳健性能，并在各种语言应用场景中展现出强大的泛化能力。我们认为，U-GIFT为少量标注样本下的有毒言论检测提供了一个高效解决方案，为网络空间中的自动化内容审核提供有力支持，从而成为保障网络安全性的防火墙，助力网络安全技术的发展。 

---
# Decoupling Knowledge and Reasoning in Transformers: A Modular Architecture with Generalized Cross-Attention 

**Title (ZH)**: 解耦知识和推理在变换器中的分离：一种采用通用交叉注意力的模块化架构 

**Authors**: Zhenyu Guo, Wenguang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.00823)  

**Abstract**: Transformers have achieved remarkable success across diverse domains, but their monolithic architecture presents challenges in interpretability, adaptability, and scalability. This paper introduces a novel modular Transformer architecture that explicitly decouples knowledge and reasoning through a generalized cross-attention mechanism to a shared knowledge base, specifically designed for effective knowledge retrieval. Critically, we provide a rigorous mathematical derivation demonstrating that the Feed-Forward Network (FFN) in a standard Transformer is a specialized case (a closure) of this generalized cross-attention, revealing its role in implicit knowledge retrieval and validating our design. This theoretical framework provides a new lens for understanding FFNs and lays the foundation for future research exploring enhanced interpretability, adaptability, and scalability, enabling richer interplay with external knowledge bases and other systems. 

**Abstract (ZH)**: Transformer架构已经在多种领域中取得了显著的成功，但其整体式的架构在可解释性、适应性和扩展性方面提出了挑战。本文介绍了一种新颖的模块化Transformer架构，通过一个广义的交叉注意力机制显式地将知识和推理分离到一个共享的知识库中，特别适用于有效知识检索。关键的是，我们提供了一种严格的数学推导，证明了标准Transformer中的前向网络（FFN）是该广义交叉注意力机制的一种特殊情形（闭包），揭示了其在隐式知识检索中的作用，并验证了我们的设计。该理论框架为理解FFN提供了一个新的视角，并为未来的研究奠定了基础，旨在探索增强的可解释性、适应性和扩展性，从而实现更加丰富的与外部知识库和其他系统的交互。 

---
# SLIDE: Integrating Speech Language Model with LLM for Spontaneous Spoken Dialogue Generation 

**Title (ZH)**: SLIDE：将语音语言模型与大规模语言模型集成以生成自发口语对话 

**Authors**: Haitian Lu, Gaofeng Cheng, Liuping Luo, Leying Zhang, Yanmin Qian, Pengyuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.00805)  

**Abstract**: Recently, ``textless" speech language models (SLMs) based on speech units have made huge progress in generating naturalistic speech, including non-verbal vocalizations. However, the generated speech samples often lack semantic coherence. In this paper, we propose SLM and LLM Integration for spontaneous spoken Dialogue gEneration (SLIDE). Specifically, we first utilize an LLM to generate the textual content of spoken dialogue. Next, we convert the textual dialogues into phoneme sequences and use a two-tower transformer-based duration predictor to predict the duration of each phoneme. Finally, an SLM conditioned on the spoken phoneme sequences is used to vocalize the textual dialogue. Experimental results on the Fisher dataset demonstrate that our system can generate naturalistic spoken dialogue while maintaining high semantic coherence. 

**Abstract (ZH)**: 近年来，“无文本”语音语言模型（SLMs）基于语音单元在生成自然语音方面取得了巨大进展，包括非言语声化。然而，生成的语音样本往往缺乏语义连贯性。在本文中，我们提出了一种将SLMs和LLMs集成用于自发性口语对话生成的方法（SLIDE）。具体而言，我们首先利用一个LLM生成口语对话的内容。接下来，我们将文本对话转换为音素序列，并使用基于Transformer的双塔模型预测每个音素的持续时间。最后，一个基于口语音素序列的SLM用于语音化文本对话。在Fisher数据集上的实验结果表明，我们的系统可以生成既自然又保持高语义连贯性的口语对话。 

---
# Automatic Text Pronunciation Correlation Generation and Application for Contextual Biasing 

**Title (ZH)**: 自动文本发音相关生成及其在情境偏差调整中的应用 

**Authors**: Gaofeng Cheng, Haitian Lu, Chengxu Yang, Xuyang Wang, Ta Li, Yonghong Yan  

**Link**: [PDF](https://arxiv.org/pdf/2501.00804)  

**Abstract**: Effectively distinguishing the pronunciation correlations between different written texts is a significant issue in linguistic acoustics. Traditionally, such pronunciation correlations are obtained through manually designed pronunciation lexicons. In this paper, we propose a data-driven method to automatically acquire these pronunciation correlations, called automatic text pronunciation correlation (ATPC). The supervision required for this method is consistent with the supervision needed for training end-to-end automatic speech recognition (E2E-ASR) systems, i.e., speech and corresponding text annotations. First, the iteratively-trained timestamp estimator (ITSE) algorithm is employed to align the speech with their corresponding annotated text symbols. Then, a speech encoder is used to convert the speech into speech embeddings. Finally, we compare the speech embeddings distances of different text symbols to obtain ATPC. Experimental results on Mandarin show that ATPC enhances E2E-ASR performance in contextual biasing and holds promise for dialects or languages lacking artificial pronunciation lexicons. 

**Abstract (ZH)**: 在语音声学领域，有效地区分不同书面文本之间的发音关联是一个重要问题。传统上，这些发音关联是通过手工设计的发音词典获得的。本文提出了一种数据驱动的方法，可以自动获取这些发音关联，称为自动文本发音关联（ATPC）。该方法所需的监督信息与训练端到端自动语音识别（E2E-ASR）系统所需的监督信息一致，即语音和相应的文本注释。首先，迭代训练的时间戳估计器（ITSE）算法被用来对齐语音与相应的注释文本符号。然后，使用语音编码器将语音转换为语音嵌入。最后，通过比较不同文本符号的语音嵌入距离来获得ATPC。实验结果表明，ATPC能够提高E2E-ASR的上下文偏向性，并且对于缺乏人工发音词典的方言或语言具有潜力。 

---
# Adjoint sharding for very long context training of state space models 

**Title (ZH)**: 状态空间模型中非常长上下文训练的伴随分片方法 

**Authors**: Xingzi Xu, Amir Tavanaei, Kavosh Asadi, Karim Bouyarmane  

**Link**: [PDF](https://arxiv.org/pdf/2501.00692)  

**Abstract**: Despite very fast progress, efficiently training large language models (LLMs) in very long contexts remains challenging. Existing methods fall back to training LLMs with short contexts (a maximum of a few thousands tokens in training) and use inference time techniques when evaluating on long contexts (above 1M tokens context window at inference). As opposed to long-context-inference, training on very long context input prompts is quickly limited by GPU memory availability and by the prohibitively long training times it requires on state-of-the-art hardware. Meanwhile, many real-life applications require not only inference but also training/fine-tuning with long context on specific tasks. Such applications include, for example, augmenting the context with various sources of raw reference information for fact extraction, fact summarization, or fact reconciliation tasks. We propose adjoint sharding, a novel technique that comprises sharding gradient calculation during training to reduce memory requirements by orders of magnitude, making training on very long context computationally tractable. Adjoint sharding is based on the adjoint method and computes equivalent gradients to backpropagation. We also propose truncated adjoint sharding to speed up the algorithm while maintaining performance. We provide a distributed version, and a paralleled version of adjoint sharding to further speed up training. Empirical results show the proposed adjoint sharding algorithm reduces memory usage by up to 3X with a 1.27B parameter large language model on 1M context length training. This allows to increase the maximum context length during training or fine-tuning of a 1.27B parameter model from 35K tokens to above 100K tokens on a training infrastructure composed of five AWS P4 instances. 

**Abstract (ZH)**: 尽管取得了非常快速的进展，但在极长上下文中高效训练大型语言模型（LLMs）仍然具有挑战性。现有方法选择在较短的上下文中训练LLMs（最多几千个标记），并在评估时使用推理时的技术（上下文窗口超过100万标记）。与长上下文推理不同，由于GPU内存可用性和在最新硬件上所需的数据延迟长的训练时间限制，极长上下文输入提示的训练很快就受到限制。同时，许多实际应用场景不仅需要推理，还需要在特定任务上进行训练/微调，涉及长上下文。例如，在事实提取、事实总结或事实校对任务中，增强上下文以包含各种原始参考资料信息。我们提出了一种新颖的技术——反向分片（adjoint sharding），该技术在训练过程中分片梯度计算，极大地减少了内存需求，使极长上下文训练变得计算上可行。反向分片基于反向法（adjoint method），计算与反向传播等效的梯度。我们还提出了一种截断反向分片，以加快算法速度并保持性能。我们还提供了分布式和并行版本的反向分片以进一步加快训练速度。实验证明，在一个包含100万标记长度的1.27亿参数的大语言模型训练中，所提出的反向分片算法能够将内存使用量最多减少3倍。这使得在训练基础设施由五台AWS P4实例组成的架构下，对于1.27亿参数模型的训练或微调，最大上下文长度可从35K标记增加到超过100K标记。 

---
# IGC: Integrating a Gated Calculator into an LLM to Solve Arithmetic Tasks Reliably and Efficiently 

**Title (ZH)**: IGC：将门控计算器集成到大规模语言模型中，以可靠且高效地完成算术任务 

**Authors**: Florian Dietz, Dietrich Klakow  

**Link**: [PDF](https://arxiv.org/pdf/2501.00684)  

**Abstract**: Solving arithmetic tasks is a simple and fundamental skill, yet modern Large Language Models (LLMs) have great difficulty with them. We introduce the Integrated Gated Calculator (IGC), a module that enables LLMs to perform arithmetic by emulating a calculator on the GPU. We finetune a Llama model with our module and test it on the BigBench Arithmetic benchmark, where it beats the State of the Art, outperforming all models on the benchmark, including models almost two orders of magnitude larger. Our approach takes only a single iteration to run and requires no external tools. It performs arithmetic operations entirely inside the LLM without the need to produce intermediate tokens. It is computationally efficient, interpretable, and avoids side-effects on tasks that do not require arithmetic operations. It reliably achieves 98\% to 99\% accuracy across multiple training runs and for all subtasks, including the substantially harder subtask of multiplication, which was previously unsolved. 

**Abstract (ZH)**: 解决算术任务是一项简单而基本的技能，然而现代大型语言模型（LLMs）在处理这类任务时却面临巨大困难。我们引入了综合门控计算器（IGC），这是一种模块，使LLMs能够通过在GPU上模拟计算器的方式执行算术计算。我们使用该模块对Llama模型进行了微调，并在BigBench Arithmetic基准测试中进行测试，结果显示该模型比现有最佳方法表现更优，即使在基准测试中，它也超过了所有其他模型，包括规模大了近两个数量级的模型。我们的方法只需要单次迭代即可运行，并不需要外部工具。该方法完全在LLM内部进行算术操作，无需生成中间词元。这种方法计算效率高、可解释性强，对于不需要进行算术操作的任务不会产生副作用。多次训练运行和所有子任务中，该方法都能可靠地达到98%到99%的准确性，甚至包括先前未被解决的乘法子任务。 

---
# Titans: Learning to Memorize at Test Time 

**Title (ZH)**: Titans: 在测试时学习记忆 

**Authors**: Ali Behrouz, Peilin Zhong, Vahab Mirrokni  

**Link**: [PDF](https://arxiv.org/pdf/2501.00663)  

**Abstract**: Over more than a decade there has been an extensive research effort on how to effectively utilize recurrent models and attention. While recurrent models aim to compress the data into a fixed-size memory (called hidden state), attention allows attending to the entire context window, capturing the direct dependencies of all tokens. This more accurate modeling of dependencies, however, comes with a quadratic cost, limiting the model to a fixed-length context. We present a new neural long-term memory module that learns to memorize historical context and helps attention to attend to the current context while utilizing long past information. We show that this neural memory has the advantage of fast parallelizable training while maintaining a fast inference. From a memory perspective, we argue that attention due to its limited context but accurate dependency modeling performs as a short-term memory, while neural memory due to its ability to memorize the data, acts as a long-term, more persistent, memory. Based on these two modules, we introduce a new family of architectures, called Titans, and present three variants to address how one can effectively incorporate memory into this architecture. Our experimental results on language modeling, common-sense reasoning, genomics, and time series tasks show that Titans are more effective than Transformers and recent modern linear recurrent models. They further can effectively scale to larger than 2M context window size with higher accuracy in needle-in-haystack tasks compared to baselines. 

**Abstract (ZH)**: 在过去的十多年里，关于如何有效地利用循环模型和注意力机制的研究取得了广泛进展。循环模型旨在将数据压缩到固定大小的记忆状态（称为隐藏状态）中，而注意力机制则允许关注整个上下文窗口，捕捉所有标记的直接依赖关系。然而，这种更准确的依赖建模伴随着平方时间的成本，限制了模型只能处理固定长度的上下文。我们提出了一种新的神经长期记忆模块，能够学习记住历史上下文，并在利用长期信息的同时帮助注意力机制关注当前的上下文。我们展示了这种神经记忆的优点在于可以快速并行训练，同时保持快速的推理速度。从存储器的角度来看，我们认为：由于上下文有限但依赖建模准确，注意力机制充当短期记忆；而由于其能够存储数据的特性，神经记忆则作为长期的、更具持久性的记忆。基于这两个模块，我们引入了一种新的架构家族，称为Titans，并提出了三种变体，以探讨如何有效地将记忆整合到这种架构中。我们在语言建模、常识推理、基因组学和时间序列任务上的实验结果表明，Titans在有效性上优于transformers和最近的现代线性循环模型。此外，与 baseline 相比，它们还能在具有超过200万个上下文窗口大小的任务中更有效地扩展，并保持更高的准确性。 

---
# Why Are Positional Encodings Nonessential for Deep Autoregressive Transformers? Revisiting a Petroglyph 

**Title (ZH)**: 为什么位置编码对深度自回归变压器模型而言并非不可或缺？重新审视一种象形文字

此翻译保持了原文的含义，并且符合学术论文的翻译规范，但在标题中使用“象形文字”可能不太直观。如果希望标题更加清晰，可以调整为：

为什么位置编码对深度自回归变压器模型而言并非不可或缺？重新审视一种古老标记

这样更符合学术标题的简洁性和易理解性。 

**Authors**: Kazuki Irie  

**Link**: [PDF](https://arxiv.org/pdf/2501.00659)  

**Abstract**: Do autoregressive Transformer language models require explicit positional encodings (PEs)? The answer is "no" as long as they have more than one layer -- they can distinguish sequences with permuted tokens without requiring explicit PEs. This property has been known since early efforts (those contemporary with GPT-2) adopting the Transformer for language modeling. However, this result does not appear to have been well disseminated and was even rediscovered recently. This may be partially due to a sudden growth of the language modeling community after the advent of GPT-2, but perhaps also due to the lack of a clear explanation in prior publications, despite being commonly understood by practitioners in the past. Here we review this long-forgotten explanation why explicit PEs are nonessential for multi-layer autoregressive Transformers (in contrast, one-layer models require PEs to discern order information of their input tokens). We also review the origin of this result, and hope to re-establish it as a common knowledge. 

**Abstract (ZH)**: 自回归Transformer语言模型是否需要显式的位置编码（PEs）？只要模型有多于一层的结构，答案是“不需要”——它们可以在不需要显式位置编码的情况下区分具有置换词元的序列。这一特性自早期采用Transformer进行语言建模的努力（与GPT-2同期）以来就已经为人所知。然而，这项结果并没有广泛传播，并且甚至最近又被重新发现。这可能部分归因于GPT-2问世后语言建模社区的突然增长，但也可能是由于早期出版物中缺少清晰的解释，尽管过去实践中普遍理解这一点。在这里，我们回顾了这种被遗忘的解释，说明了为什么多层自回归Transformer不需要显式位置编码（相比之下，单层模型需要位置编码来区分其输入词元的顺序信息）。我们还回顾了这一结果的起源，并希望能够重新确立其为常识。 

---
# ICONS: Influence Consensus for Vision-Language Data Selection 

**Title (ZH)**: ICONS：视觉-语言数据选择中的影响共识 

**Authors**: Xindi Wu, Mengzhou Xia, Rulin Shao, Zhiwei Deng, Pang Wei Koh, Olga Russakovsky  

**Link**: [PDF](https://arxiv.org/pdf/2501.00654)  

**Abstract**: Visual Instruction Tuning typically requires a large amount of vision-language training data. This data often containing redundant information that increases computational costs without proportional performance gains. In this work, we introduce ICONS, a gradient-driven Influence CONsensus approach for vision-language data Selection that selects a compact training dataset for efficient multi-task training. The key element of our approach is cross-task influence consensus, which uses majority voting across task-specific influence matrices to identify samples that are consistently valuable across multiple tasks, allowing us to effectively prioritize data that optimizes for overall performance. Experiments show that models trained on our selected data (20% of LLaVA-665K) achieve 98.6% of the relative performance obtained using the full dataset. Additionally, we release this subset, LLaVA-ICONS-133K, a compact yet highly informative subset of LLaVA-665K visual instruction tuning data, preserving high impact training data for efficient vision-language model development. 

**Abstract (ZH)**: 视觉语言指令调优通常需要大量的视觉-语言训练数据。这些数据往往包含冗余信息，这些信息增加了计算成本而未能带来相应的性能提升。在本文中，我们引入了一种基于梯度的Influence CONsensus（ICONS）选择方法，用于视觉-语言数据的选择，该方法可以为高效的多任务训练选择一个紧凑的训练数据集。我们方法的关键元素是跨任务影响共识，它使用特定任务的影响矩阵之间的多数投票来识别在多个任务中持续具有高价值的样本，从而能够有效优先选择优化整体性能的数据。实验结果显示，在我们选择的数据集（LLaVA-665K的20%）上训练的模型，实现了与使用完整数据集训练相比98.6%的相对性能。此外，我们还发布了这个子集LLaVA-ICONS-133K，这是LLaVA-665K视觉指令调优数据的一个紧凑而高度信息丰富的子集，保留了高效的视觉-语言模型开发所需的重要训练数据。 

---
# MCP-Solver: Integrating Language Models with Constraint Programming Systems 

**Title (ZH)**: MCP-Solver：将语言模型与约束编程系统集成 

**Authors**: Stefan Szeider  

**Link**: [PDF](https://arxiv.org/pdf/2501.00539)  

**Abstract**: While Large Language Models (LLMs) perform exceptionally well at natural language tasks, they often struggle with precise formal reasoning and the rigorous specification of problems. We present MCP-Solver, a prototype implementation of the Model Context Protocol that demonstrates the potential for systematic integration between LLMs and constraint programming systems. Our implementation provides interfaces for the creation, editing, and validation of a constraint model. Through an item-based editing approach with integrated validation, the system ensures model consistency at every modification step and enables structured iterative refinement. The system handles concurrent solving sessions and maintains a persistent knowledge base of modeling insights. Initial experiments suggest that this integration can effectively combine LLMs' natural language understanding with constraint-solving capabilities. Our open-source implementation is proof of concept for integrating formal reasoning systems with LLMs through standardized protocols. While further research is needed to establish comprehensive formal guarantees, this work takes a first step toward principled integration of natural language processing with constraint-based reasoning. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在自然语言任务上表现出色，但在精确的形式推理和问题的严格定义方面往往存在困难。我们提出了一种名为MCP-Solver的模型上下文协议原型实现，展示了LLMs与约束编程系统系统化集成的潜力。我们的实现提供了创建、编辑和验证约束模型的接口。通过基于项目的方法进行集成验证，在每个修改步骤中确保模型一致性，并支持结构化的逐步改进。该系统能够处理并发求解会话，并维护一个持久的知识库，记录建模见解。初步实验表明，这种集成可以有效地结合LLMs的自然语言理解和求解约束的能力。我们的开源实现证明了通过标准化协议整合形式推理系统与LLMs的可行性。尽管还需要进一步的研究来建立全面的形式保证，但这项工作迈出了一步，旨在原理上将自然语言处理与基于约束的推理相结合。 

---
# Two Cases of Deduction with Non-referring Descriptions 

**Title (ZH)**: 含有非指称描述符的两个演绎推理案例 

**Authors**: Jiří Raclavský  

**Link**: [PDF](https://arxiv.org/pdf/2501.00485)  

**Abstract**: Formal reasoning with non-denoting terms, esp. non-referring descriptions such as "the King of France", is still an under-investigated area. The recent exception being a series of papers e.g. by Indrzejczak, Zawidzki and Krbis. The present paper offers an alternative to their approach since instead of free logic and sequent calculus, it's framed in partial type theory with natural deduction in sequent style. Using a Montague- and Tichý-style formalization of  natural language, the paper successfully handles deduction with intensional transitives whose complements are non-referring descriptions, and derives Strawsonian rules for existential presuppositions of sentences with such descriptions. 

**Abstract (ZH)**: 非指称词项，尤其是非指称描述词，如“法国的国王”，形式推理仍然是一个研究不足的领域。近期，这一领域的例外是Indrzejczak、Zawidzki和Krbis等人的系列论文。本文提出了一种不同于他们方法的途径，因为本文采用的是部分类型论，并以序逻辑演绎的方式进行自然演绎。通过采用蒙塔古和提基风格对自然语言的形式化表示，本文成功地处理了含有意向性传递词的演绎，其补语是非指称描述词，并推导出了关于具有此类描述词的句子的 Strawson 规则，即存在预设规则。 

---
# Differentiable Prompt Learning for Vision Language Models 

**Title (ZH)**: 视觉语言模型中的可微提示学习 

**Authors**: Zhenhan Huang, Tejaswini Pedapati, Pin-Yu Chen, Jianxi Gao  

**Link**: [PDF](https://arxiv.org/pdf/2501.00457)  

**Abstract**: Prompt learning is an effective way to exploit the potential of large-scale pre-trained foundational models. Continuous prompts parameterize context tokens in prompts by turning them into differentiable vectors. Deep continuous prompts insert prompts not only in the input but also in the intermediate hidden representations. Manually designed deep continuous prompts exhibit a remarkable improvement compared to the zero-shot pre-trained model on downstream tasks. How to automate the continuous prompt design is an underexplored area, and a fundamental question arises, is manually designed deep prompt strategy optimal? To answer this question, we propose a method dubbed differentiable prompt learning (DPL). The DPL method is formulated as an optimization problem to automatically determine the optimal context length of the prompt to be added to each layer, where the objective is to maximize the performance. We test the DPL method on the pre-trained CLIP. We empirically find that by using only limited data, our DPL method can find deep continuous prompt configuration with high confidence. The performance on the downstream tasks exhibits the superiority of the automatic design: our method boosts the average test accuracy by 2.60% on 11 datasets compared to baseline methods. Besides, our method focuses only on the prompt configuration (i.e. context length for each layer), which means that our method is compatible with the baseline methods that have sophisticated designs to boost the performance. The DPL method can be deployed to large language models or computer vision models at no cost. 

**Abstract (ZH)**: prompt学习是一种有效利用大规模预训练基础模型潜在能力的方法。连续提示通过将提示中的上下文标记转换为可微分向量来参数化这些标记。深层连续提示不仅在输入中插入提示，还在中间隐藏表示中插入提示。与零-shot预训练模型相比，手工设计的深层连续提示在下游任务上表现出显著的改进。如何自动化设计连续提示是一个未被充分探索的领域，因此一个基本的问题浮现出来，即手工设计的深层提示策略是否是最优的？为回答这一问题，本文提出了一种名为可微提示学习（DPL）的方法。DPL方法被形式化为一个优化问题，以自动确定将添加到每一层的提示的最佳上下文长度，其目标是最大化性能。我们在预训练的CLIP上测试了DPL方法。实验证明，仅使用少量数据，我们的DPL方法能够以高置信度发现深层连续提示的配置。在下游任务上的性能表明自动设计的优势：与基线方法相比，我们的方法在11个数据集上将平均测试精度提高了2.60%。此外，我们的方法仅关注提示配置（即每层的上下文长度），这意味着我们的方法与那些通过复杂设计提升性能的基线方法兼容。DPL方法无需成本即可应用于大规模语言模型或计算机视觉模型。 

---
# Unleashing Text-to-Image Diffusion Prior for Zero-Shot Image Captioning 

**Title (ZH)**: 释放文本到图像扩散先验以实现零样本图像配描述 

**Authors**: Jianjie Luo, Jingwen Chen, Yehao Li, Yingwei Pan, Jianlin Feng, Hongyang Chao, Ting Yao  

**Link**: [PDF](https://arxiv.org/pdf/2501.00437)  

**Abstract**: Recently, zero-shot image captioning has gained increasing attention, where only text data is available for training. The remarkable progress in text-to-image diffusion model presents the potential to resolve this task by employing synthetic image-caption pairs generated by this pre-trained prior. Nonetheless, the defective details in the salient regions of the synthetic images introduce semantic misalignment between the synthetic image and text, leading to compromised results. To address this challenge, we propose a novel Patch-wise Cross-modal feature Mix-up (PCM) mechanism to adaptively mitigate the unfaithful contents in a fine-grained manner during training, which can be integrated into most of encoder-decoder frameworks, introducing our PCM-Net. Specifically, for each input image, salient visual concepts in the image are first detected considering the image-text similarity in CLIP space. Next, the patch-wise visual features of the input image are selectively fused with the textual features of the salient visual concepts, leading to a mixed-up feature map with less defective content. Finally, a visual-semantic encoder is exploited to refine the derived feature map, which is further incorporated into the sentence decoder for caption generation. Additionally, to facilitate the model training with synthetic data, a novel CLIP-weighted cross-entropy loss is devised to prioritize the high-quality image-text pairs over the low-quality counterparts. Extensive experiments on MSCOCO and Flickr30k datasets demonstrate the superiority of our PCM-Net compared with state-of-the-art VLMs-based approaches. It is noteworthy that our PCM-Net ranks first in both in-domain and cross-domain zero-shot image captioning. The synthetic dataset SynthImgCap and code are available at this https URL. 

**Abstract (ZH)**: 近年来，零样本图像标注得到了越来越多的关注，其中仅使用文本数据进行训练。文本到图像的扩散模型的显著进展为通过使用由预训练模型生成的合成图像-描述对来解决这一问题提供了可能。然而，合成图像中的缺陷细节在显著区域引入了合成图像与文本之间的语义错配，导致生成的结果不佳。为了解决这一挑战，我们提出了一种新颖的Patch-wise跨模态特征Mix-up (PCM)机制，在训练过程中以细粒度的方式自适应地减轻不忠实的内容，该机制可以整合到多数编码-解码框架中，形成我们的PCM-Net。具体来说，对于每个输入图像，首先在CLIP空间中考虑图像-文本相似性来检测图像中的显著视觉概念。接着，输入图像的局部视觉特征与显著视觉概念的文本特征选择性地融合，形成一个内容较少有缺陷的混合特征图。最后，利用视觉-语义编码器对生成的特征图进行细化，并进一步将其整合到句子解码器中进行描述生成。此外，为了便于使用合成数据进行模型训练，我们设计了一种新颖的CLIP加权交叉熵损失函数，优先选择高质量的图像-文本对。在MSCOCO和Flickr30k数据集上的 extensive 实验表明，我们提出的PCM-Net 在与最先进的基于语义-视觉模型的方法相比时表现出更好的性能。值得注意的是，我们的PCM-Net在领域内和跨领域零样本图像标注中均排名第一。合成数据集 SynthImgCap 和代码可通过以下链接获得：[提供的链接]。 

---
# TSPE: Task-Specific Prompt Ensemble for Improved Zero-Shot Audio Classification 

**Title (ZH)**: TSPE: 针对特定任务的提示集ensemble以提高零样本音频分类性能 

**Authors**: Nishit Anand, Ashish Seth, Ramani Duraiswami, Dinesh Manocha  

**Link**: [PDF](https://arxiv.org/pdf/2501.00398)  

**Abstract**: Audio-language models (ALMs) excel in zero-shot audio classification, a task where models classify previously unseen audio clips at test time by leveraging descriptive natural language prompts. We introduce TSPE (Task-Specific Prompt Ensemble), a simple, training-free hard prompting method that boosts ALEs' zero-shot performance by customizing prompts for diverse audio classification tasks. Rather than using generic template-based prompts like "Sound of a car" we generate context-rich prompts, such as "Sound of a car coming from a tunnel". Specifically, we leverage label information to identify suitable sound attributes, such as "loud" and "feeble", and appropriate sound sources, such as "tunnel" and "street" and incorporate this information into the prompts used by Audio-Language Models (ALMs) for audio classification. Further, to enhance audio-text alignment, we perform prompt ensemble across TSPE-generated task-specific prompts. When evaluated on 12 diverse audio classification datasets, TSPE improves performance across ALMs by showing an absolute improvement of 1.23-16.36% over vanilla zero-shot evaluation. 

**Abstract (ZH)**: 音频-语言模型（ALMs）在零样本音频分类任务中表现出色，即在测试时通过利用描述性的自然语言提示对未见过的音频片段进行分类。我们引入了一种简单且无需训练的定制提示方法TSPE（Task-Specific Prompt Ensemble），该方法通过为多样化的音频分类任务量身定制提示，提升ALMs的零样本性能。与使用通用模板提示如“汽车的声音”不同，我们生成了更具上下文信息的提示，例如“汽车从隧道里传来的声音”。具体来说，我们利用标签信息来识别适合的声音属性，如“大声”和“微弱”，以及适当的声音来源，如“隧道”和“街道”，并将这些信息融入到音频-语言模型（ALMs）用于音频分类的提示中。此外，为了增强音频-文本对齐，我们通过TSPE生成的任务特定提示进行提示组合。在对12个不同的音频分类数据集进行评估时，TSPE在ALMs上的性能有了显著提升，相比于标准的零样本评估，绝对改进幅度在1.23%-16.36%之间。 

---
# Efficient Relational Context Perception for Knowledge Graph Completion 

**Title (ZH)**: 高效的关系上下文感知方法在知识图谱完成中的应用 

**Authors**: Wenkai Tu, Guojia Wan, Zhengchun Shang, Bo Du  

**Link**: [PDF](https://arxiv.org/pdf/2501.00397)  

**Abstract**: Knowledge Graphs (KGs) provide a structured representation of knowledge but often suffer from challenges of incompleteness. To address this, link prediction or knowledge graph completion (KGC) aims to infer missing new facts based on existing facts in KGs. Previous knowledge graph embedding models are limited in their ability to capture expressive features, especially when compared to deeper, multi-layer models. These approaches also assign a single static embedding to each entity and relation, disregarding the fact that entities and relations can exhibit different behaviors in varying graph contexts. Due to complex context over a fact triple of a KG, existing methods have to leverage complex non-linear context encoder, like transformer, to project entity and relation into low dimensional representations, resulting in high computation cost. To overcome these limitations, we propose Triple Receptance Perception (TRP) architecture to model sequential information, enabling the learning of dynamic context of entities and relations. Then we use tensor decomposition to calculate triple scores, providing robust relational decoding capabilities. This integration allows for more expressive representations. Experiments on benchmark datasets such as YAGO3-10, UMLS, FB15k, and FB13 in link prediction and triple classification tasks demonstrate that our method performs better than several state-of-the-art models, proving the effectiveness of the integration. 

**Abstract (ZH)**: 知识图谱（KGs）提供了结构化的知识表示，但往往面临着不完整性的挑战。为了应对这一问题，链接预测或知识图谱补全（KGC）旨在根据KGs中现有的事实推断出缺失的新事实。之前的知识图谱嵌入模型在捕捉表达性特征方面能力有限，特别是在与深层次、多层模型相比时。这些方法还为每个实体和关系分配了一个静态嵌入，忽视了实体和关系在不同图上下文中可能会表现出不同的行为。由于KG中一个三元组的复杂上下文，现有方法必须利用如变压器这样的复杂非线性上下文编码器将实体和关系投影到低维表示，从而导致了高计算成本。为克服这些局限性，我们提出了Triple接收感知（TRP）架构，以建模序列信息，从而能够学习实体和关系的动态上下文。然后我们利用张量分解来计算三元组分数，提供强大的关系解码能力。这种集成使得表达性表示成为可能。在链路预测和三元组分类任务中的基准数据集（如YAGO3-10、UMLS、FB15k和FB13）上进行的实验表明，我们的方法优于几种最新的模型，证明了这种集成的有效性。 

---
# VoxVietnam: a Large-Scale Multi-Genre Dataset for Vietnamese Speaker Recognition 

**Title (ZH)**: VoxVietnam：一种大规模多类型数据集，用于越南语说话人识别 

**Authors**: Hoang Long Vu, Phuong Tuan Dat, Pham Thao Nhi, Nguyen Song Hao, Nguyen Thi Thu Trang  

**Link**: [PDF](https://arxiv.org/pdf/2501.00328)  

**Abstract**: Recent research in speaker recognition aims to address vulnerabilities due to variations between enrolment and test utterances, particularly in the multi-genre phenomenon where the utterances are in different speech genres. Previous resources for Vietnamese speaker recognition are either limited in size or do not focus on genre diversity, leaving studies in multi-genre effects unexplored. This paper introduces VoxVietnam, the first multi-genre dataset for Vietnamese speaker recognition with over 187,000 utterances from 1,406 speakers and an automated pipeline to construct a dataset on a large scale from public sources. Our experiments show the challenges posed by the multi-genre phenomenon to models trained on a single-genre dataset, and demonstrate a significant increase in performance upon incorporating the VoxVietnam into the training process. Our experiments are conducted to study the challenges of the multi-genre phenomenon in speaker recognition and the performance gain when the proposed dataset is used for multi-genre training. 

**Abstract (ZH)**: 近年来，发言者识别的研究致力于解决注册和测试片段之间变化带来的脆弱性问题，特别是在多体裁现象中，这些片段属于不同的语音体裁。之前有关越南语发言者识别的数据资源要么规模有限，要么没有关注体裁多样性，这导致对多体裁效应的研究未被探索。本文介绍了VoxVietnam，这是首个用于越南语发言者识别的多体裁数据集，包含超过187,000个来自1,406位发言者的片段，并提供了一个自动化的数据集构建管道，可以从公共来源大规模构建数据集。我们的实验表明，多体裁现象给单体裁数据集训练的模型带来了挑战，并证明了在训练过程中加入VoxVietnam后性能有了显著提升。我们的实验旨在研究多体裁现象对发言者识别的挑战及其在多体裁训练中使用提出的数据集所带来的性能提升。 

---
# Retrieval-Augmented Generation with Graphs (GraphRAG) 

**Title (ZH)**: 基于图的检索增强生成（GraphRAG） 

**Authors**: Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Jiayuan Ding, Yongjia Lei, Mahantesh Halappanavar, Ryan A. Rossi, Subhabrata Mukherjee, Xianfeng Tang, Qi He, Zhigang Hua, Bo Long, Tong Zhao, Neil Shah, Amin Javari, Yinglong Xia, Jiliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2501.00309)  

**Abstract**: Retrieval-augmented generation (RAG) is a powerful technique that enhances downstream task execution by retrieving additional information, such as knowledge, skills, and tools from external sources. Graph, by its intrinsic "nodes connected by edges" nature, encodes massive heterogeneous and relational information, making it a golden resource for RAG in tremendous real-world applications. As a result, we have recently witnessed increasing attention on equipping RAG with Graph, i.e., GraphRAG. However, unlike conventional RAG, where the retriever, generator, and external data sources can be uniformly designed in the neural-embedding space, the uniqueness of graph-structured data, such as diverse-formatted and domain-specific relational knowledge, poses unique and significant challenges when designing GraphRAG for different domains. Given the broad applicability, the associated design challenges, and the recent surge in GraphRAG, a systematic and up-to-date survey of its key concepts and techniques is urgently desired. Following this motivation, we present a comprehensive and up-to-date survey on GraphRAG. Our survey first proposes a holistic GraphRAG framework by defining its key components, including query processor, retriever, organizer, generator, and data source. Furthermore, recognizing that graphs in different domains exhibit distinct relational patterns and require dedicated designs, we review GraphRAG techniques uniquely tailored to each domain. Finally, we discuss research challenges and brainstorm directions to inspire cross-disciplinary opportunities. Our survey repository is publicly maintained at this https URL. 

**Abstract (ZH)**: 检索增强生成（RAG）是一种强大的技术，通过从外部来源检索额外信息（如知识、技能和工具），增强下游任务的执行。图，由于其本质上的“节点通过边连接”的特性，编码了大量的异构关系信息，使其成为RAG在众多实际应用中的一笔宝贵的资源。因此，我们最近见证了在RAG中增加图结构的越来越多的关注，即GraphRAG。然而，与传统的RAG不同，其中检索器、生成器和外部数据源可以在神经嵌入空间中均匀设计，图结构的数据，如形式多样且领域特定的关系知识，为设计不同领域的GraphRAG带来了独特且重大的挑战。鉴于其广泛的应用性、相关的设计挑战以及GraphRAG的近期发展，对GraphRAG的关键概念和技术进行全面和最新的综述变得至关重要。在此动机的驱使下，我们提供了一篇全面且最新的关于GraphRAG的综述。

我们的综述首先提出了一个全面的GraphRAG框架，通过定义其关键组件，包括查询处理器、检索器、组织器、生成器和数据源。进一步地，我们认识到不同领域的图展现出不同的关系特征并需要特定的设计，因此我们回顾了专为每个领域设计的独特GraphRAG技术。最后，我们讨论了研究挑战并提出了跨学科合作的方向。我们的综述库在此httpsURL上公开维护。 

---
# Automatically Planning Optimal Parallel Strategy for Large Language Models 

**Title (ZH)**: 自动规划大型语言模型的理想并行策略 

**Authors**: Zongbiao Li, Xiezhao Li, Yinghao Cui, Yijun Chen, Zhixuan Gu, Yuxuan Liu, Wenbo Zhu, Fei Jia, Ke Liu, Qifeng Li, Junyao Zhan, Jiangtao Zhou, Chenxi Zhang, Qike Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.00254)  

**Abstract**: The number of parameters in large-scale language models based on transformers is gradually increasing, and the scale of computing clusters is also growing. The technology of quickly mobilizing large amounts of computing resources for parallel computing is becoming increasingly important. In this paper, we propose an automatic parallel algorithm that automatically plans the parallel strategy with maximum throughput based on model and hardware information. By decoupling the training time into computation, communication, and overlap, we established a training duration simulation model. Based on this simulation model, we prune the parallel solution space to shorten the search time required. The multi-node experiment results show that the algorithm can estimate the parallel training duration in real time with an average accuracy of 96%. In our test, the recommendation strategy provided by the algorithm is always globally optimal. 

**Abstract (ZH)**: 基于变换器的大规模语言模型参数数量逐渐增加，计算集群的规模也在不断扩大。快速调动大量计算资源进行并行计算的技术变得越来越重要。本文提出了一种自动并行算法，该算法根据模型和硬件信息自动规划具有最大吞吐量的并行策略。通过将训练时间分解为计算时间、通信时间和重叠时间，我们建立了训练时长的仿真模型。基于该仿真模型，我们剪枝并行解决方案空间以缩短搜索时间。多节点实验结果表明，该算法可以实时估计并行训练时长，平均准确率为96%。在我们的测试中，算法提供的推荐策略始终是全局最优的。 

---
# Generative Emergent Communication: Large Language Model is a Collective World Model 

**Title (ZH)**: 生成式 emergent 通信：大规模语言模型是一个集体世界模型 

**Authors**: Tadahiro Taniguchi, Ryo Ueda, Tomoaki Nakamura, Masahiro Suzuki, Akira Taniguchi  

**Link**: [PDF](https://arxiv.org/pdf/2501.00226)  

**Abstract**: This study proposes a unifying theoretical framework called generative emergent communication (generative EmCom) that bridges emergent communication, world models, and large language models (LLMs) through the lens of collective predictive coding (CPC). The proposed framework formalizes the emergence of language and symbol systems through decentralized Bayesian inference across multiple agents, extending beyond conventional discriminative model-based approaches to emergent communication. This study makes the following two key contributions: First, we propose generative EmCom as a novel framework for understanding emergent communication, demonstrating how communication emergence in multi-agent reinforcement learning (MARL) can be derived from control as inference while clarifying its relationship to conventional discriminative approaches. Second, we propose a mathematical formulation showing the interpretation of LLMs as collective world models that integrate multiple agents' experiences through CPC. The framework provides a unified theoretical foundation for understanding how shared symbol systems emerge through collective predictive coding processes, bridging individual cognitive development and societal language evolution. Through mathematical formulations and discussion on prior works, we demonstrate how this framework explains fundamental aspects of language emergence and offers practical insights for understanding LLMs and developing sophisticated AI systems for improving human-AI interaction and multi-agent systems. 

**Abstract (ZH)**: 本研究提出了一种统一的理论框架，称为生成性涌现通信（Generative Emergent Communication，简称生成EmCom），该框架通过集体预测编码（CPC）的视角，将涌现通信、世界模型和大规模语言模型（LLMs）联系起来。所提出的框架通过分散贝叶斯推理在多个代理之间的应用，形式化了语言和符号系统的涌现，超越了传统的基于判别模型的方法对涌现通信的探讨。本研究做出了以下两大关键贡献：首先，我们提出了生成EmCom作为理解涌现通信的新框架，展示如何在网络强化学习（MARL）中的通信涌现可以从控制作为推理中导出，并澄清其与传统的判别方法之间的关系。其次，我们提出了一种数学公式，表明大规模语言模型可以被视为通过CPC整合多个代理经验的集体世界模型。该框架为理解通过集体预测编码过程共享符号系统的涌现提供了统一的理论基础，连接了个体内认知发展与社会语言演变。通过数学公式和对先前工作的讨论，我们展示了该框架如何解释语言涌现的基本方面，并为理解和开发促进人类-智能体互动和多智能体系统的复杂AI系统提供了实用见解。 

---
# MLLM-as-a-Judge for Image Safety without Human Labeling 

**Title (ZH)**: 将以下论文的内容或标题翻译成中文，同时确保符合学术规范：

“MLLM-as-a-Judge 无须人工标注的图像安全性评估”

说明：
- “MLLM”可能是指多模态大型语言模型（Multimodal Large Language Model），因此这里将其翻译为“MLLM”以保持原文缩写的完整性。
- “as a Judge”在这里指的是“作为评估者”。
- “无须人工标注”是对“without Human Labeling”的翻译，符合学术写作中的表达习惯。
- “图像安全性评估”是对“Image Safety”的翻译。 

**Authors**: Zhenting Wang, Shuming Hu, Shiyu Zhao, Xiaowen Lin, Felix Juefei-Xu, Zhuowei Li, Ligong Han, Harihar Subramanyam, Li Chen, Jianfa Chen, Nan Jiang, Lingjuan Lyu, Shiqing Ma, Dimitris N. Metaxas, Ankit Jain  

**Link**: [PDF](https://arxiv.org/pdf/2501.00192)  

**Abstract**: Image content safety has become a significant challenge with the rise of visual media on online platforms. Meanwhile, in the age of AI-generated content (AIGC), many image generation models are capable of producing harmful content, such as images containing sexual or violent material. Thus, it becomes crucial to identify such unsafe images based on established safety rules. Pre-trained Multimodal Large Language Models (MLLMs) offer potential in this regard, given their strong pattern recognition abilities. Existing approaches typically fine-tune MLLMs with human-labeled datasets, which however brings a series of drawbacks. First, relying on human annotators to label data following intricate and detailed guidelines is both expensive and labor-intensive. Furthermore, users of safety judgment systems may need to frequently update safety rules, making fine-tuning on human-based annotation more challenging. This raises the research question: Can we detect unsafe images by querying MLLMs in a zero-shot setting using a predefined safety constitution (a set of safety rules)? Our research showed that simply querying pre-trained MLLMs does not yield satisfactory results. This lack of effectiveness stems from factors such as the subjectivity of safety rules, the complexity of lengthy constitutions, and the inherent biases in the models. To address these challenges, we propose a MLLM-based method includes objectifying safety rules, assessing the relevance between rules and images, making quick judgments based on debiased token probabilities with logically complete yet simplified precondition chains for safety rules, and conducting more in-depth reasoning with cascaded chain-of-thought processes if necessary. Experiment results demonstrate that our method is highly effective for zero-shot image safety judgment tasks. 

**Abstract (ZH)**: 随着在线平台上视觉媒体的兴起，图像内容安全已成为一项重大挑战。同时，在人工智能生成内容（AIGC）的时代，许多图像生成模型能够生成有害内容，如包含色情或暴力的图像。因此，基于现有的安全规则识别此类不安全图像变得至关重要。预训练的多模态大型语言模型（MLLMs）在这方面具有潜力，因其具有强大的模式识别能力。现有的方法通常会使用人工标注的数据集对MLLMs进行微调，但这种方法带来了许多缺点。首先，依靠人工注释员按照复杂的详细指南标注数据既昂贵又耗时。此外，使用安全评估系统的用户可能需要频繁更新安全规则，使得基于人工标注的微调更加困难。这提出了一个问题：我们是否可以在零样本设置中通过查询预训练的MLLMs并使用预定义的安全宪法（一组安全规则）来检测不安全图像？我们的研究结果显示，仅仅通过查询预训练的MLLMs并不能取得满意的效果。这种无效性主要源于安全规则的主观性、宪法的复杂性以及模型本身的固有偏差。为解决这些挑战，我们提出了一种基于MLLMs的方法，该方法包括客观化安全规则、评估规则与图像的相关性、基于去偏置的令牌概率进行快速判断，并使用逻辑完整但简化的情境链来确定安全规则，必要时进行更深入的递进式推理。实验结果表明，我们的方法在零样本图像安全判断任务中非常有效。 

---
# DeepLL: Considering Linear Logic for the Analysis of Deep Learning Experiments 

**Title (ZH)**: DeepLL：考虑线性逻辑对深度学习实验的分析 

**Authors**: Nick Papoulias  

**Link**: [PDF](https://arxiv.org/pdf/2501.00169)  

**Abstract**: Deep Learning experiments have critical requirements regarding the careful handling of their datasets as well as the efficient and correct usage of APIs that interact with hardware accelerators. On the one hand, software mistakes during data handling can contaminate experiments and lead to incorrect results. On the other hand, poorly coded APIs that interact with the hardware can lead to sub-optimal usage and untrustworthy conclusions. In this work we investigate the use of Linear Logic for the analysis of Deep Learning experiments. We show that primitives and operators of Linear Logic can be used to express: (i) an abstract representation of the control flow of an experiment, (ii) a set of available experimental resources, such as API calls to the underlying data-structures and hardware as well as (iii) reasoning rules about the correct consumption of resources during experiments. Our proposed model is not only lightweight but also easy to comprehend having both a symbolic and a visual component. Finally, its artifacts are themselves proofs in Linear Logic that can be readily verified by off-the-shelf reasoners. 

**Abstract (ZH)**: 深度学习实验对数据集的细致处理及其与硬件加速器交互的API的高效且正确的使用有着严格的要求。一方面，数据处理中的软件错误会污染实验并导致错误的结果。另一方面，编码糟糕的与硬件交互的API可能导致次优利用和不可靠的结论。在本文中，我们探讨了使用线性逻辑来分析深度学习实验。我们展示了线性逻辑的原始符号和操作可以用于表达：(i) 实验控制流的抽象表示；(ii) 可用的实验资源集，包括底层数据结构和硬件的API调用等；以及(iii) 关于实验期间正确消耗资源的推理规则。我们提出的模型不仅轻量级，而且易于理解，具有符号和视觉两个组成部分。最后，其产生的结果本身就是线性逻辑中的证明，可以方便地通过现成的推理器进行验证。 

---
# LLM-Virus: Evolutionary Jailbreak Attack on Large Language Models 

**Title (ZH)**: LLM-Virus：大规模语言模型的进化型越狱攻击 

**Authors**: Miao Yu, Junfeng Fang, Yingjie Zhou, Xing Fan, Kun Wang, Shirui Pan, Qingsong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2501.00055)  

**Abstract**: While safety-aligned large language models (LLMs) are increasingly used as the cornerstone for powerful systems such as multi-agent frameworks to solve complex real-world problems, they still suffer from potential adversarial queries, such as jailbreak attacks, which attempt to induce harmful content. Researching attack methods allows us to better understand the limitations of LLM and make trade-offs between helpfulness and safety. However, existing jailbreak attacks are primarily based on opaque optimization techniques (e.g. token-level gradient descent) and heuristic search methods like LLM refinement, which fall short in terms of transparency, transferability, and computational cost. In light of these limitations, we draw inspiration from the evolution and infection processes of biological viruses and propose LLM-Virus, a jailbreak attack method based on evolutionary algorithm, termed evolutionary jailbreak. LLM-Virus treats jailbreak attacks as both an evolutionary and transfer learning problem, utilizing LLMs as heuristic evolutionary operators to ensure high attack efficiency, transferability, and low time cost. Our experimental results on multiple safety benchmarks show that LLM-Virus achieves competitive or even superior performance compared to existing attack methods. 

**Abstract (ZH)**: 当下的安全对齐大型语言模型（LLMs）正越来越多地被用作解决复杂现实世界问题的强大系统（如多智能体框架）的基石，但它们仍然面临潜在的对抗性查询，比如脱戒攻击，这种攻击试图诱导生成有害内容。研究攻击方法可以帮助我们更好地理解大型语言模型的局限性，并在有益性和安全性之间做出权衡。然而，现有的脱戒攻击主要基于不透明的优化技术（如标记级梯度下降）和启发式搜索方法（如LLM优化），在透明性、转移性和计算成本方面存在不足。鉴于这些局限性，我们从生物病毒的演化和感染过程汲取灵感，提出了基于进化算法的脱戒攻击方法——进化脱戒（Evolutionary Jailbreak），并将其命名为LLM-Virus。LLM-Virus 将脱戒攻击视为一个既包括进化学习又包括转移学习的问题，利用大型语言模型作为启发式的进化操作者，以确保高攻击效率、出色的转移能力以及较低的成本。我们在多个安全基准上的实验结果显示，LLM-Virus 达到了与现有攻击方法相当甚至更优的性能。 

---
# AdvAnchor: Enhancing Diffusion Model Unlearning with Adversarial Anchors 

**Title (ZH)**: AdvAnchor: 增强基于对抗锚点的扩散模型遗忘技术 

**Authors**: Mengnan Zhao, Lihe Zhang, Xingyi Yang, Tianhang Zheng, Baocai Yin  

**Link**: [PDF](https://arxiv.org/pdf/2501.00054)  

**Abstract**: Security concerns surrounding text-to-image diffusion models have driven researchers to unlearn inappropriate concepts through fine-tuning. Recent fine-tuning methods typically align the prediction distributions of unsafe prompts with those of predefined text anchors. However, these techniques exhibit a considerable performance trade-off between eliminating undesirable concepts and preserving other concepts. In this paper, we systematically analyze the impact of diverse text anchors on unlearning performance. Guided by this analysis, we propose AdvAnchor, a novel approach that generates adversarial anchors to alleviate the trade-off issue. These adversarial anchors are crafted to closely resemble the embeddings of undesirable concepts to maintain overall model performance, while selectively excluding defining attributes of these concepts for effective erasure. Extensive experiments demonstrate that AdvAnchor outperforms state-of-the-art methods. Our code is publicly available at this https URL. 

**Abstract (ZH)**: 围绕文本到图像扩散模型的安全性问题，研究者们通过微调来消除不适当的概念。近期的微调方法通常通过将不合适提示的预测分布与预定义的文本锚点的分布相匹配来实现这一点。然而，这些技术在消除不良概念和保持其他概念之间表现出相当大的性能权衡。在本文中，我们系统地分析了不同文本锚点对遗忘性能的影响。基于这一分析，我们提出了一种名为AdvAnchor的新方法，以减轻这种权衡问题。这些对抗性锚点被设计成与不良概念的嵌入高度相似，以保持整体模型性能，同时选择性地排除这些概念的定义属性，从而实现有效的删除。大量的实验表明，AdvAnchor 在性能上优于现有的最新方法。我们的代码已公开发布在 <this https URL>。 

---
# Speech Recognition With LLMs Adapted to Disordered Speech Using Reinforcement Learning 

**Title (ZH)**: 使用强化学习适应非规范语音的大型语言模型的语音识别 

**Authors**: Chirag Nagpal, Subhashini Venugopalan, Jimmy Tobin, Marilyn Ladewig, Katherine Heller, Katrin Tomanek  

**Link**: [PDF](https://arxiv.org/pdf/2501.00039)  

**Abstract**: We introduce a large language model (LLM) capable of processing speech inputs and show that tuning it further with reinforcement learning on human preference (RLHF) enables it to adapt better to disordered speech than traditional fine-tuning. Our method replaces low-frequency text tokens in an LLM's vocabulary with audio tokens and enables the model to recognize speech by fine-tuning it on speech with transcripts. We then use RL with rewards based on syntactic and semantic accuracy measures generalizing the LLM further to recognize disordered speech. While the resulting LLM does not outperform existing systems for speech recognition, we find that tuning with reinforcement learning using custom rewards leads to substantially better performance than supervised fine-tuning of the language model, specifically when adapting to speech in a different setting. This presents a compelling alternative tuning strategy for speech recognition using large language models. 

**Abstract (ZH)**: 我们介绍了一种大型语言模型（LLM），该模型能够处理语音输入，并表明通过强化学习（RLHF）进一步调整该模型使其能够更好地适应乱序语音，而传统的微调效果则较差。我们的方法用音频令牌替换LLM词汇表中低频的文字令牌，从而使模型能够在带有转录的语音数据上进行微调，进而识别语音。随后，我们使用基于语法和语义准确性的奖励进行RL，进一步推广LLM以识别乱序语音。尽管最终得到的LLM在语音识别方面并未胜过现有的系统，但我们发现使用自定义奖励进行强化学习的微调，与监督下的语言模型微调相比，其性能显著提升，尤其是在不同场景下的语音适应方面。这为使用大型语言模型进行语音识别提供了一种有吸引力的替代调优策略。 

---
# Highly Optimized Kernels and Fine-Grained Codebooks for LLM Inference on Arm CPUs 

**Title (ZH)**: 针对Arm CPU上大语言模型推理的高性能内核和细粒度代码本 

**Authors**: Dibakar Gope, David Mansell, Danny Loh, Ian Bratt  

**Link**: [PDF](https://arxiv.org/pdf/2501.00032)  

**Abstract**: Large language models (LLMs) have transformed the way we think about language understanding and generation, enthralling both researchers and developers. However, deploying LLMs for inference has been a significant challenge due to their unprecedented size and resource requirements. While quantizing model weights to sub-byte precision has emerged as a promising solution to ease memory pressure, the group quantization formats commonly used for LLM quantization have significant compute overheads and a resource-intensive dequantization process. As a result, a higher proportion of compute instructions do not perform multiplies, i.e., real work, rendering them unsuitable for meeting the required latency requirements for LLMs deployed on commodity CPUs. In this work, we propose a set of highly optimized kernels to accelerate LLM inference and unleash the full potential of CPUs, particularly Arm CPUs. These kernels amortize the cost of loading the operands and the cost of weight unpacking across multiple output rows. This, along with the introduction of an optimized interleaved group data layout for weights and decompression path optimizations to reduce unnecessary operations and dequantization overhead while maximizing the use of vector and matrix multiply operations, significantly improves the efficiency of MAC operations. Furthermore, we present a groupwise non-uniform codebook-based quantization method for ultra-low-precision quantization of LLMs to better match non-uniform patterns in their weight distributions, demonstrating better throughput during token generation while ensuring better quality than the state-of-the-art. Applying these improvements to 4-bit LLMs results in a 3-3.2x improvement in prompt processing and a 2x improvement in autoregressive decoding on Arm CPUs, compared to this http URL-based solution. The optimized kernels are available at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）已彻底改变我们对语言理解和生成的看法，吸引了研究人员和开发者的广泛关注。然而，由于其前所未有的规模和资源需求，将LLMs部署用于推理已成为一项重大挑战。虽然将模型权重量化至亚字节精度已被证明是一种减轻内存压力的有效解决方案，但LLMs量化中常用的分组量化格式在计算和去量化过程中存在显著的计算开销。这导致了更多的计算指令未执行实际工作（即未进行乘法运算），从而无法满足在商用CPU上部署LLMs时所需的延迟要求。在本项工作中，我们提出了一组高度优化的内核，以加速LLM推理，充分利用CPU的潜在性能，尤其是Arm CPU。这些内核通过将在多个输出行上分配负载算子的成本和权重解码的成本，实现成本的分摊。此外，我们引入了一种优化的交错分组数据布局，并对权重解压路径进行了优化，以减少不必要的操作和去量化开销，同时最大化了向量和矩阵乘法操作的使用，显著提高了MAC操作的效率。此外，我们提出了一种基于分组非均匀码本的超低精度量化方法，以更好地匹配LLMs权重分布中非均匀模式，在令牌生成过程中表现出更高的吞吐量，同时保持比现有最佳方法更好的质量。将这些改进应用于4位LLMs时，相比于该网址提供的解决方案，它们在Arm CPU上的提示处理提高了3-3.2倍，在自回归解码上提高了2倍。优化后的内核可在该网址获取。 

---
# NewsHomepages: Homepage Layouts Capture Information Prioritization Decisions 

**Title (ZH)**: 新闻主页：主页布局反映信息优先级决策 

**Authors**: Ben Welsh, Naitian Zhou, Arda Kaz, Michael Vu, Alexander Spangher  

**Link**: [PDF](https://arxiv.org/pdf/2501.00004)  

**Abstract**: Information prioritization plays an important role in how humans perceive and understand the world. Homepage layouts serve as a tangible proxy for this prioritization. In this work, we present NewsHomepages, a large dataset of over 3,000 new website homepages (including local, national and topic-specific outlets) captured twice daily over a three-year period. We develop models to perform pairwise comparisons between news items to infer their relative significance. To illustrate that modeling organizational hierarchies has broader implications, we applied our models to rank-order a collection of local city council policies passed over a ten-year period in San Francisco, assessing their "newsworthiness". Our findings lay the groundwork for leveraging implicit organizational cues to deepen our understanding of information prioritization. 

**Abstract (ZH)**: 信息优先级在人类感知和理解世界的过程中发挥着重要作用。主页布局是这一优先级的实体代表。在本研究中，我们推出了一个名为“NewsHomepages”的大型数据集，该数据集包含超过3,000个新闻网站主页（包括地方、国家级和专题特定的媒体），这些主页在过去三年中每天被捕捉两次。我们开发了模型来在新闻项目之间进行成对比较，以推断它们的相对重要性。为了说明建模组织层次结构具有更广泛的含义，我们将这些模型应用于按十年时间顺序排列的旧金山市议会政策集合，评估它们的“新闻价值”。我们的研究结果为利用潜在的组织线索以加深我们对信息优先级的理解奠定了基础。 

---
