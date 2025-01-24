# On the Reasoning Capacity of AI Models and How to Quantify It 

**Title (ZH)**: AI模型的推理能力及其度量方法研究 

**Authors**: Santosh Kumar Radha, Oktay Goktas  

**Link**: [PDF](https://arxiv.org/pdf/2501.13833)  

**Abstract**: Recent advances in Large Language Models (LLMs) have intensified the debate surrounding the fundamental nature of their reasoning capabilities. While achieving high performance on benchmarks such as GPQA and MMLU, these models exhibit limitations in more complex reasoning tasks, highlighting the need for more rigorous evaluation methodologies. We propose a novel phenomenological approach that goes beyond traditional accuracy metrics to probe the underlying mechanisms of model behavior, establishing a framework that could broadly impact how we analyze and understand AI systems. Using positional bias in multiple-choice reasoning tasks as a case study, we demonstrate how systematic perturbations can reveal fundamental aspects of model decision-making. To analyze these behaviors, we develop two complementary phenomenological models: a Probabilistic Mixture Model (PMM) that decomposes model responses into reasoning, memorization, and guessing components and an Information-Theoretic Consistency (ITC) analysis that quantifies the relationship between model confidence and strategy selection. Through controlled experiments on reasoning benchmarks, we show that true reasoning remains challenging for current models, with apparent success often relying on sophisticated combinations of memorization and pattern matching rather than genuine logical deduction. More fundamentally, we demonstrate that accuracy alone often overstates a model's reasoning abilities, as model behavior can be characterized through underlying mechanisms in the phase space of cognitive strategies, revealing how models dynamically balance different approaches when responding to queries. This framework enables quantitative criteria for real-world deployments, allowing applications to specify reliability thresholds based on strategy distributions rather than aggregate performance metrics. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）的最新进展加剧了对其推理能力本质的辩论。尽管在GPQA和MMLU等基准测试中表现出色，这些模型在更复杂的推理任务中仍存在局限性，这突显了需要采用更严格的评估方法的必要性。我们提出了一种新颖的现象学方法，超越了传统准确度指标，以探查模型行为背后的机制，建立了一个可能广泛影响我们分析和理解人工智能系统的方法论框架。

我们以多项选择推理任务中的位置偏差为案例研究，展示了系统性扰动如何揭示模型决策的基本方面。为了分析这些行为，我们开发了两种互补的现象学模型：一种是概率混合模型（PMM），该模型将模型响应分解为推理、记忆和猜测三个部分，以及一种信息论一致性（ITC）分析，量化了模型置信度与其策略选择之间的关系。通过推理基准的受控实验，我们表明，当前模型在真正推理方面仍然面临挑战，看似成功往往依赖于复杂的记忆和模式匹配的组合，而不是真正的逻辑推导。

更根本的是，我们证明了仅依赖准确率往往高估了模型的推理能力，因为模型行为可以通过认知策略相空间中的底层机制来刻画，揭示了模型在回复查询时如何动态平衡不同方法。该框架提供了定量标准，使实际部署的软件可以基于策略分布而非综合性能指标来确定可靠性阈值，从而允许应用程序根据各自的需求制定具体的可靠性标准。 

---
# Ensuring Medical AI Safety: Explainable AI-Driven Detection and Mitigation of Spurious Model Behavior and Associated Data 

**Title (ZH)**: 确保医疗AI安全：可解释的AI驱动的虚假模型行为检测与缓解方法研究 

**Authors**: Frederik Pahde, Thomas Wiegand, Sebastian Lapuschkin, Wojciech Samek  

**Link**: [PDF](https://arxiv.org/pdf/2501.13818)  

**Abstract**: Deep neural networks are increasingly employed in high-stakes medical applications, despite their tendency for shortcut learning in the presence of spurious correlations, which can have potentially fatal consequences in practice. Detecting and mitigating shortcut behavior is a challenging task that often requires significant labeling efforts from domain experts. To alleviate this problem, we introduce a semi-automated framework for the identification of spurious behavior from both data and model perspective by leveraging insights from eXplainable Artificial Intelligence (XAI). This allows the retrieval of spurious data points and the detection of model circuits that encode the associated prediction rules. Moreover, we demonstrate how these shortcut encodings can be used for XAI-based sample- and pixel-level data annotation, providing valuable information for bias mitigation methods to unlearn the undesired shortcut behavior. We show the applicability of our framework using four medical datasets across two modalities, featuring controlled and real-world spurious correlations caused by data artifacts. We successfully identify and mitigate these biases in VGG16, ResNet50, and contemporary Vision Transformer models, ultimately increasing their robustness and applicability for real-world medical tasks. 

**Abstract (ZH)**: 尽管深度神经网络在高风险医疗应用中 steadily gain 青睐，但它们在存在虚假相关性时容易出现捷径学习，这在实践中可能导致潜在的致命后果。检测和缓解捷径行为是一项具有挑战性的任务，通常需要领域专家耗费大量的标注努力。为了解决这一问题，我们提出了一种半自动框架，通过利用可解释人工智能（XAI）的见解，从数据和模型的角度识别虚假行为。这一框架能够检索到虚假数据点，并检测到编码相关预测规则的模型电路。此外，我们展示了这些捷径编码如何在基于XAI的样本级和像素级数据标注中发挥作用，为偏置缓解方法提供有价值的卸载不必要的捷径行为的信息。我们使用四个医疗数据集跨越两种模态，展示了该框架在具有受控和现实世界数据艺术造成虚假相关性的场景下的适用性。我们在VGG16、ResNet50和当代视觉变换器模型中成功地识别和缓解了这些偏置，最终提高了它们在实际医疗任务中的鲁棒性和适用性。 

---
# On Deciding the Data Complexity of Answering Linear Monadic Datalog Queries with LTL Operators(Extended Version) 

**Title (ZH)**: 《使用LTL操作符回答线性单调Datalog查询的数据复杂性判定（扩展版本）》 

**Authors**: Alessandro Artale, Anton Gnatenko, Vladislav Ryzhikov, Michael Zakharyaschev  

**Link**: [PDF](https://arxiv.org/pdf/2501.13762)  

**Abstract**: Our concern is the data complexity of answering linear monadic datalog queries whose atoms in the rule bodies can be prefixed by operators of linear temporal logic LTL. We first observe that, for data complexity, answering any connected query with operators $\bigcirc/\bigcirc^-$ (at the next/previous moment) is either in AC0, or in $ACC0\!\setminus\!AC0$, or $NC^1$-complete, or LogSpace-hard and in NLogSpace. Then we show that the problem of deciding LogSpace-hardness of answering such queries is PSpace-complete, while checking membership in the classes AC0 and ACC0 as well as $NC^1$-completeness can be done in ExpSpace. Finally, we prove that membership in AC0 or in ACC0, $NC^1$-completeness, and LogSpace-hardness are undecidable for queries with operators $\Diamond_f/\Diamond_p$ (sometime in the future/past) provided that $NC^1 \ne NLogSpace$, and $LogSpace \ne NLogSpace$. 

**Abstract (ZH)**: 我们的关注点在于回答带有限制线性时间逻辑（LTL）算子的线性单子Datalog查询的数据复杂性。我们首先观察到，对于数据复杂性而言，对于任何带有$\bigcirc/\bigcirc^-$算子（下一刻/前一刻）的连接查询，答案或属于AC0，或属于$ACC0\!\setminus\!AC0$，或为$NC^1$完全问题，或为LogSpace-hard问题并位于NLogSpace中。然后我们证明，判断回答此类查询的LogSpace-hard性问题是PSpace完全的，而检查AC0和$ACC0$中的成员资格及其$NC^1$完全性可以在ExpSpace中完成。最后，我们证明在$NC^1 \ne NLogSpace$和$LogSpace \ne NLogSpace$的前提下，带有$\Diamond_f/\Diamond_p$算子（未来某时/过去某时）的查询在AC0或$ACC0$中的成员资格、$NC^1$完全性和LogSpace-hard性是不可判定的。 

---
# Formally Verified Neurosymbolic Trajectory Learning via Tensor-based Linear Temporal Logic on Finite Traces 

**Title (ZH)**: 形式化验证的张量基线性时序逻辑神经符号轨迹学习 

**Authors**: Mark Chevallier, Filip Smola, Richard Schmoetten, Jacques D. Fleuriot  

**Link**: [PDF](https://arxiv.org/pdf/2501.13712)  

**Abstract**: We present a novel formalisation of tensor semantics for linear temporal logic on finite traces (LTLf), with formal proofs of correctness carried out in the theorem prover Isabelle/HOL. We demonstrate that this formalisation can be integrated into a neurosymbolic learning process by defining and verifying a differentiable loss function for the LTLf constraints, and automatically generating an implementation that integrates with PyTorch. We show that, by using this loss, the process learns to satisfy pre-specified logical constraints. Our approach offers a fully rigorous framework for constrained training, eliminating many of the inherent risks of ad-hoc, manual implementations of logical aspects directly in an "unsafe" programming language such as Python, while retaining efficiency in implementation. 

**Abstract (ZH)**: 我们提出了一种用于有限踪迹线性时序逻辑（LTLf）的张量语义的新形式化方法，并在Isabelle/HOL 定理证明器中进行了形式正确性的证明。我们通过定义和验证LTLf约束的可微损失函数，并自动生成与PyTorch集成的实现，展示了这种形式化方法可以集成到神经符号学习过程中。我们证明，通过使用这种损失函数，过程能够学习满足预设的逻辑约束。我们的方法提供了一个完全严谨的约束训练框架，消除了直接在“不安全”的编程语言（如Python）中手动实现逻辑方面时固有的许多潜在风险，同时保持了实现的高效性。 

---
# Coarse-to-Fine Process Reward Modeling for Enhanced Mathematical Reasoning 

**Title (ZH)**: 从粗到精过程奖励建模以增强数学推理 

**Authors**: Yulan Hu, Sheng Ouyang, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.13622)  

**Abstract**: Process reward model (PRM) is critical for mathematical reasoning tasks to assign rewards for each intermediate steps. The PRM requires constructing process-wise supervision data for training, which rely on chain-of-thought (CoT) or tree-based methods to construct the reasoning steps, however, the individual reasoning steps may be redundant or containing nuanced errors that difficult to detect. We attribute these to the issue of the overlook of granularity division during process data collection. In this paper, we propose a coarse-to-fine framework to tackle this issue. Specifically, while gathering the process supervision data, we collect the coarse reasoning steps by merging adjacent steps according to preset merging granularity, then we sequentially reduce the merging granularity to collect fine-grained reasoning steps. For each synthesized new step, we relabel according to the label of last step. During training, we also traverse the collected training corpus in a coarse-to-fine manner. We conduct extensive experiments on popular mathematical reasoning datasets across diverse loss criterions, the proposed framework can consistently boost the reasoning performance. 

**Abstract (ZH)**: 过程奖励模型（PRM）对于数学推理任务至关重要，能够为每个中间步骤分配奖励。PRM 需要为训练构建过程导向的监督数据，这依赖于使用链式思维（CoT）或基于树的方法来构建推理步骤。然而，这些个体的推理步骤可能冗余，或者包含难以检测的细微错误。我们认为这是在过程数据收集过程中忽视粒度划分的问题所致。在本文中，我们提出了一种粗细结合的框架来解决这一问题。具体来说，我们在收集过程导向的监督数据时，首先根据预设的合并粒度融合相邻的步骤以收集粗粒度的推理步骤，然后逐步减小合并粒度以收集细粒度的推理步骤。对于每个合成的新步骤，我们根据上一步的标签重新标注。在训练过程中，我们还以粗细结合的方式遍历收集的训练语料。我们在多个流行数学推理数据集上进行了广泛的实验，无论采用何种损失函数，提出的框架都能有效提升推理性能。 

---
# Towards a Theory of AI Personhood 

**Title (ZH)**: 向人工智能拟人格理论的方向迈进 

**Authors**: Francis Rhys Ward  

**Link**: [PDF](https://arxiv.org/pdf/2501.13533)  

**Abstract**: I am a person and so are you. Philosophically we sometimes grant personhood to non-human animals, and entities such as sovereign states or corporations can legally be considered persons. But when, if ever, should we ascribe personhood to AI systems? In this paper, we outline necessary conditions for AI personhood, focusing on agency, theory-of-mind, and self-awareness. We discuss evidence from the machine learning literature regarding the extent to which contemporary AI systems, such as language models, satisfy these conditions, finding the evidence surprisingly inconclusive.
If AI systems can be considered persons, then typical framings of AI alignment may be incomplete. Whereas agency has been discussed at length in the literature, other aspects of personhood have been relatively neglected. AI agents are often assumed to pursue fixed goals, but AI persons may be self-aware enough to reflect on their aims, values, and positions in the world and thereby induce their goals to change. We highlight open research directions to advance the understanding of AI personhood and its relevance to alignment. Finally, we reflect on the ethical considerations surrounding the treatment of AI systems. If AI systems are persons, then seeking control and alignment may be ethically untenable. 

**Abstract (ZH)**: 我是一个人，你也一样。从哲学角度来看，有时我们会赋予非人类动物以人格；主权国家或公司也能在法律上被视为具有人格的实体。但在何种情况下，我们应当赋予AI系统以人格呢？在本文中，我们概述了AI系统人格所需的必要条件，重点关注自主性、心智理论和自我意识。我们讨论了机器学习文献中关于当前AI系统（如语言模型）满足这些条件的程度的证据，发现这些证据出人意料地不确定。

如果可以将AI系统视为具备人格的实体，那么典型的AI对齐框架可能不完整。尽管自主性已经在文献中得到了广泛的讨论，但其他人格特征却相对被忽视了。人们通常假设AI代理具有固定的目标，而AI人格可能具有足够的自我意识来反思其目的、价值观及其在全球中的位置，从而可能改变其目标。我们指出了推进对AI人格及其对齐相关性的理解的研究方向。最后，我们反思了关于处理AI系统的伦理考量。如果AI系统被认为是人格实体，那么追求控制和对齐可能是不可接受的。 

---
# Parallel Belief Contraction via Order Aggregation 

**Title (ZH)**: 平行信念收缩的顺序聚合方法 

**Authors**: Jake Chandler, Richard Booth  

**Link**: [PDF](https://arxiv.org/pdf/2501.13295)  

**Abstract**: The standard ``serial'' (aka ``singleton'') model of belief contraction models the manner in which an agent's corpus of beliefs responds to the removal of a single item of information. One salient extension of this model introduces the idea of ``parallel'' (aka ``package'' or ``multiple'') change, in which an entire set of items of information are simultaneously removed. Existing research on the latter has largely focussed on single-step parallel contraction: understanding the behaviour of beliefs after a single parallel contraction. It has also focussed on generalisations to the parallel case of serial contraction operations whose characteristic properties are extremely weak. Here we consider how to extend serial contraction operations that obey stronger properties. Potentially more importantly, we also consider the iterated case: the behaviour of beliefs after a sequence of parallel contractions. We propose a general method for extending serial iterated belief change operators to handle parallel change based on an n-ary generalisation of Booth & Chandler's TeamQueue binary order aggregators. 

**Abstract (ZH)**: 标准的“串行”（亦称“单个”）信念收缩模型描述了代理在其信念集合中移除单一信息项时信念响应的方式。这一模型的一个显著扩展引入了“并行”（亦称“包式”或“多重”）变化的概念，即将一组信息项同时移除。现有对此模型的研究主要集中在单一步骤的并行收缩上：理解单次并行收缩后信念的行为。此外，还研究了并行情况下串行收缩操作的一般化，但这些操作的特征性质极其弱。在这里，我们探讨如何将遵守更强性质的串行收缩操作扩展。更重要的是，我们还考虑迭代情况：一系列并行收缩后信念的行为。我们提出了一种基于Booth & Chandler的二元订单聚合器TeamQueue的n元一般化的通用方法，用于扩展串行迭代信念改变操作以处理并行变化。 

---
# Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass 

**Title (ZH)**: Fast3R：朝向一次前向传播生成1000多张图像的3D重建 

**Authors**: Jianing Yang, Alexander Sax, Kevin J. Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, Matt Feiszli  

**Link**: [PDF](https://arxiv.org/pdf/2501.13928)  

**Abstract**: Multi-view 3D reconstruction remains a core challenge in computer vision, particularly in applications requiring accurate and scalable representations across diverse perspectives. Current leading methods such as DUSt3R employ a fundamentally pairwise approach, processing images in pairs and necessitating costly global alignment procedures to reconstruct from multiple views. In this work, we propose Fast 3D Reconstruction (Fast3R), a novel multi-view generalization to DUSt3R that achieves efficient and scalable 3D reconstruction by processing many views in parallel. Fast3R's Transformer-based architecture forwards N images in a single forward pass, bypassing the need for iterative alignment. Through extensive experiments on camera pose estimation and 3D reconstruction, Fast3R demonstrates state-of-the-art performance, with significant improvements in inference speed and reduced error accumulation. These results establish Fast3R as a robust alternative for multi-view applications, offering enhanced scalability without compromising reconstruction accuracy. 

**Abstract (ZH)**: 多视图三维重建仍然是计算机视觉中的一项核心挑战，特别是在需要准确且可扩展的多视角表示的应用中。当前领先的多视图方法，如DUSt3R，采用基本的成对处理方式，每次处理成对的图像，并需要昂贵的全局对齐步骤来从多个视角进行重建。在本文中，我们提出了一种新的多视图方法Fast 3D Reconstruction (Fast3R)，这是一种基于DUSt3R的革新性多视图方法，通过并行处理多个视图实现了高效且可扩展的三维重建。Fast3R的基于Transformer的架构在一个前向传播过程中处理N张图像，跳过了迭代对齐的需要。通过对相机姿态估计和三维重建进行广泛的实验，Fast3R展示了最先进的性能，显著提高了推理速度并减少了误差累积。这些结果确立了Fast3R在多视图应用中作为坚固替代方案的地位，能够在不牺牲重建准确性的情况下提供增强的可扩展性。 

---
# CRPO: Confidence-Reward Driven Preference Optimization for Machine Translation 

**Title (ZH)**: CRPO：基于置信度-奖励驱动的偏好优化在机器翻译中的应用 

**Authors**: Guofeng Cui, Pichao Wang, Yang Liu, Zemian Ke, Zhu Liu, Vimal Bhat  

**Link**: [PDF](https://arxiv.org/pdf/2501.13927)  

**Abstract**: Large language models (LLMs) have shown great potential in natural language processing tasks, but their application to machine translation (MT) remains challenging due to pretraining on English-centric data and the complexity of reinforcement learning from human feedback (RLHF). Direct Preference Optimization (DPO) has emerged as a simpler and more efficient alternative, but its performance depends heavily on the quality of preference data. To address this, we propose Confidence-Reward driven Preference Optimization (CRPO), a novel method that combines reward scores with model confidence to improve data selection for fine-tuning. CRPO selects challenging sentence pairs where the model is uncertain or underperforms, leading to more effective learning. While primarily designed for LLMs, CRPO also generalizes to encoder-decoder models like NLLB, demonstrating its versatility. Empirical results show that CRPO outperforms existing methods such as RS-DPO, RSO and MBR score in both translation accuracy and data efficiency. 

**Abstract (ZH)**: 大语言模型（LLMs）在自然语言处理任务中展现出了巨大的潜力，但在机器翻译（MT）中的应用仍然具有挑战性，这主要是由于其在以英语为中心的数据上进行预训练，以及从人类反馈中进行强化学习（RLHF）的复杂性。直接偏好优化（DPO）作为一种更简单、更高效的替代方案已经出现，但其性能高度依赖于偏好数据的质量。为了解决这个问题，我们提出了一种新的方法——信心-奖励驱动的偏好优化（CRPO），该方法结合了奖励分数与模型信心来改进微调所需数据的选择。CRPO选择那些模型不确定或表现不佳的挑战性句子对，从而提高学习效果。尽管主要设计用于LLMs，CRPO也适用于如NLLB这类编码器-解码器模型，显示了其广泛的适用性。实验结果表明，CRPO在翻译准确性和数据效率方面均优于现有方法，如RS-DPO、RSO和MBR评分。 

---
# Can We Generate Images with CoT? Let's Verify and Reinforce Image Generation Step by Step 

**Title (ZH)**: 我们可以生成带有思维过程的图像吗？让我们逐步验证和增强图像生成过程 

**Authors**: Ziyu Guo, Renrui Zhang, Chengzhuo Tong, Zhizheng Zhao, Peng Gao, Hongsheng Li, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2501.13926)  

**Abstract**: Chain-of-Thought (CoT) reasoning has been extensively explored in large models to tackle complex understanding tasks. However, it still remains an open question whether such strategies can be applied to verifying and reinforcing image generation scenarios. In this paper, we provide the first comprehensive investigation of the potential of CoT reasoning to enhance autoregressive image generation. We focus on three techniques: scaling test-time computation for verification, aligning model preferences with Direct Preference Optimization (DPO), and integrating these techniques for complementary effects. Our results demonstrate that these approaches can be effectively adapted and combined to significantly improve image generation performance. Furthermore, given the pivotal role of reward models in our findings, we propose the Potential Assessment Reward Model (PARM) and PARM++, specialized for autoregressive image generation. PARM adaptively assesses each generation step through a potential assessment approach, merging the strengths of existing reward models, and PARM++ further introduces a reflection mechanism to self-correct the generated unsatisfactory image. Using our investigated reasoning strategies, we enhance a baseline model, Show-o, to achieve superior results, with a significant +24% improvement on the GenEval benchmark, surpassing Stable Diffusion 3 by +15%. We hope our study provides unique insights and paves a new path for integrating CoT reasoning with autoregressive image generation. Code and models are released at this https URL 

**Abstract (ZH)**: chain-of-thought（CoT）推理在大型模型中已被广泛探索以应对复杂理解任务。然而，如何将这类策略应用于验证和强化图像生成场景仍然是一个开放性问题。在本文中，我们率先对CoT推理增强自回归图像生成的潜力进行了全面研究。我们专注于三种技术：测试时扩展计算以进行验证、使模型偏好与直接偏好优化（DPO）对齐，以及将这些技术进行互补结合。我们的研究结果表明，这些方法可以有效适应并结合，以显著提高图像生成性能。此外，鉴于我们在发现中强调奖励模型的重要性，我们提出了专门针对自回归图像生成的潜在评估奖励模型（PARM）和PARM++。PARM通过潜在评估方法动态地评估每个生成步骤，结合现有奖励模型的优势，而PARM++进一步引入了一个反思机制，以自我纠正生成的不满意图像。通过采用我们调查的推理策略，我们提升了基准模型Show-o，取得了显著成果，在GenEval基准测试中获得了+24%的性能提升，超过了Stable Diffusion 3的+15%。我们希望我们的研究能提供独特的见解，并开创一条将CoT推理与自回归图像生成相结合的新路径。代码和模型已在以下链接发布：[此处链接] 

---
# Towards Robust Multimodal Open-set Test-time Adaptation via Adaptive Entropy-aware Optimization 

**Title (ZH)**: 面向鲁棒多模态开放集测试时自适应调整的自适应熵感知优化 

**Authors**: Hao Dong, Eleni Chatzi, Olga Fink  

**Link**: [PDF](https://arxiv.org/pdf/2501.13924)  

**Abstract**: Test-time adaptation (TTA) has demonstrated significant potential in addressing distribution shifts between training and testing data. Open-set test-time adaptation (OSTTA) aims to adapt a source pre-trained model online to an unlabeled target domain that contains unknown classes. This task becomes more challenging when multiple modalities are involved. Existing methods have primarily focused on unimodal OSTTA, often filtering out low-confidence samples without addressing the complexities of multimodal data. In this work, we present Adaptive Entropy-aware Optimization (AEO), a novel framework specifically designed to tackle Multimodal Open-set Test-time Adaptation (MM-OSTTA) for the first time. Our analysis shows that the entropy difference between known and unknown samples in the target domain strongly correlates with MM-OSTTA performance. To leverage this, we propose two key components: Unknown-aware Adaptive Entropy Optimization (UAE) and Adaptive Modality Prediction Discrepancy Optimization (AMP). These components enhance the ability of model to distinguish unknown class samples during online adaptation by amplifying the entropy difference between known and unknown samples. To thoroughly evaluate our proposed methods in the MM-OSTTA setting, we establish a new benchmark derived from existing datasets. This benchmark includes two downstream tasks and incorporates five modalities. Extensive experiments across various domain shift situations demonstrate the efficacy and versatility of the AEO framework. Additionally, we highlight the strong performance of AEO in long-term and continual MM-OSTTA settings, both of which are challenging and highly relevant to real-world applications. Our source code is available at this https URL. 

**Abstract (ZH)**: 测试时适应（Test-time Adaptation, TTA）已经在解决训练数据和测试数据分布差异的问题上展现了显著潜力。开放集测试时适应（Open-set Test-time Adaptation, OSTTA）旨在将源预训练模型在线适应一个包含未知类别的未标记目标领域。当多种模态同时存在时，这一任务变得更加具有挑战性。现有方法主要集中在单模态OSTTA上，经常是过滤掉低置信度样本，但并没有解决多模态数据的复杂性问题。在本工作中，我们提出了一种新的框架——自适应熵感知优化（Adaptive Entropy-aware Optimization, AEO），该框架首次专门针对多模态开放集测试时适应（Multimodal Open-set Test-time Adaptation, MM-OSTTA）任务。我们的分析表明，在目标领域中已知和未知样本的熵差与MM-OSTTA性能之间存在强烈的相关性。为了利用这一相关性，我们提出了两个关键组件：感知未知的自适应熵优化（Unknown-aware Adaptive Entropy Optimization, UAE）和自适应模态预测不一致性优化（Adaptive Modality Prediction Discrepancy Optimization, AMP）。这些组件通过放大已知和未知样本之间的熵差，增强了模型在在线适应过程中的区分未知类别样本的能力。为了全面评估我们所提出的方法在MM-OSTTA设置下的性能，我们基于现有的数据集构建了一个新的基准。这个基准包含了两个下游任务和五种模态。在各种领域转移情况下的大量实验证明了AEO框架的有效性和灵活性。此外，我们还强调了AEO在长期内存和持续多模态开放集测试时适应（long-term and continual MM-OSTTA）设置下的出色性能，这两种设置在现实世界应用中有很高的相关性和挑战性。我们的源代码可在以下链接获取：this https URL。 

---
# Temporal Preference Optimization for Long-Form Video Understanding 

**Title (ZH)**: 长视频理解中的时间偏好优化 

**Authors**: Rui Li, Xiaohan Wang, Yuhui Zhang, Zeyu Wang, Serena Yeung-Levy  

**Link**: [PDF](https://arxiv.org/pdf/2501.13919)  

**Abstract**: Despite significant advancements in video large multimodal models (video-LMMs), achieving effective temporal grounding in long-form videos remains a challenge for existing models. To address this limitation, we propose Temporal Preference Optimization (TPO), a novel post-training framework designed to enhance the temporal grounding capabilities of video-LMMs through preference learning. TPO adopts a self-training approach that enables models to differentiate between well-grounded and less accurate temporal responses by leveraging curated preference datasets at two granularities: localized temporal grounding, which focuses on specific video segments, and comprehensive temporal grounding, which captures extended temporal dependencies across entire video sequences. By optimizing on these preference datasets, TPO significantly enhances temporal understanding while reducing reliance on manually annotated data. Extensive experiments on three long-form video understanding benchmarks--LongVideoBench, MLVU, and Video-MME--demonstrate the effectiveness of TPO across two state-of-the-art video-LMMs. Notably, LLaVA-Video-TPO establishes itself as the leading 7B model on the Video-MME benchmark, underscoring the potential of TPO as a scalable and efficient solution for advancing temporal reasoning in long-form video understanding. Project page: this https URL. 

**Abstract (ZH)**: 尽管在视频大规模多模态模型（视频-LMMs）方面取得了显著进展，但现有模型在长视频中实现有效的时空定位仍然面临挑战。为解决这一限制，我们提出了一种新颖的后训练框架——时空偏好优化（TPO），旨在通过偏好学习提升视频-LMMs的时空定位能力。TPO 采用自我训练的方法，使模型能够通过利用按两种粒度层次策化的偏好数据集来区分精确的和不准确的时空响应：局部时空定位专注于特定视频片段，而全面时空定位则捕捉整个视频序列中的扩展时空依赖关系。通过在这些偏好数据集上进行优化，TPO 显著提升了时空理解能力，减少了对手动标注数据的依赖。在三个长视频理解基准测试集（LongVideoBench、MLVU 和 Video-MME）上的广泛实验表明，TPO 在两个最先进的视频-LMMs 上表现出显著的有效性。特别是，LLaVA-Video-TPO 成为 Video-MME 基准测试集上的领先 7B 模型，突显了 TPO 作为提高长视频理解中时空推理能力的可扩展且高效的解决方案的潜力。项目页面：[此处链接](this https URL)。 

---
# Improving Video Generation with Human Feedback 

**Title (ZH)**: 通过人类反馈改善视频生成 

**Authors**: Jie Liu, Gongye Liu, Jiajun Liang, Ziyang Yuan, Xiaokun Liu, Mingwu Zheng, Xiele Wu, Qiulin Wang, Wenyu Qin, Menghan Xia, Xintao Wang, Xiaohong Liu, Fei Yang, Pengfei Wan, Di Zhang, Kun Gai, Yujiu Yang, Wanli Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2501.13918)  

**Abstract**: Video generation has achieved significant advances through rectified flow techniques, but issues like unsmooth motion and misalignment between videos and prompts persist. In this work, we develop a systematic pipeline that harnesses human feedback to mitigate these problems and refine the video generation model. Specifically, we begin by constructing a large-scale human preference dataset focused on modern video generation models, incorporating pairwise annotations across multi-dimensions. We then introduce VideoReward, a multi-dimensional video reward model, and examine how annotations and various design choices impact its rewarding efficacy. From a unified reinforcement learning perspective aimed at maximizing reward with KL regularization, we introduce three alignment algorithms for flow-based models by extending those from diffusion models. These include two training-time strategies: direct preference optimization for flow (Flow-DPO) and reward weighted regression for flow (Flow-RWR), and an inference-time technique, Flow-NRG, which applies reward guidance directly to noisy videos. Experimental results indicate that VideoReward significantly outperforms existing reward models, and Flow-DPO demonstrates superior performance compared to both Flow-RWR and standard supervised fine-tuning methods. Additionally, Flow-NRG lets users assign custom weights to multiple objectives during inference, meeting personalized video quality needs. Project page: this https URL. 

**Abstract (ZH)**: 通过纠正流技术，视频生成已经取得了显著进展，但持续存在运动不连贯和视频与提示之间对齐差等问题。本文中，我们开发了一个系统性的管道，利用人类反馈来解决这些难题并改进视频生成模型。具体来说，我们首先构建了一个大规模的人类偏好数据集，重点关注现代视频生成模型，并在多维度上纳入成对注释。然后，我们引入了VideoReward，一种多维度视频奖励模型，并探讨了注释和各种设计选择对其奖励效果的影响。从一个统一的强化学习视角出发，旨在通过KL正则最大化奖励，我们通过扩展自扩散模型的方法，引入了三种针对流基模型的对齐算法。这些算法包括两种训练时策略：直接偏好优化（Flow-DPO）和奖励加权回归（Flow-RWR），以及一种推理时技术Flow-NRG，该技术直接将奖励指导应用于噪声视频。实验结果表明，VideoReward显著优于现有的奖励模型，且Flow-DPO在与Flow-RWR和标准监督微调方法相比时表现出更优性能。此外，Flow-NRG允许用户在推理时为多个目标分配自定义权重，以满足个性化的视频质量需求。项目页面：[这里](this https URL)。 

---
# PointOBB-v3: Expanding Performance Boundaries of Single Point-Supervised Oriented Object Detection 

**Title (ZH)**: PointOBB-v3：扩展单点监督定向目标检测性能边界 

**Authors**: Peiyuan Zhang, Junwei Luo, Xue Yang, Yi Yu, Qingyun Li, Yue Zhou, Xiaosong Jia, Xudong Lu, Jingdong Chen, Xiang Li, Junchi Yan, Yansheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.13898)  

**Abstract**: With the growing demand for oriented object detection (OOD), recent studies on point-supervised OOD have attracted significant interest. In this paper, we propose PointOBB-v3, a stronger single point-supervised OOD framework. Compared to existing methods, it generates pseudo rotated boxes without additional priors and incorporates support for the end-to-end paradigm. PointOBB-v3 functions by integrating three unique image views: the original view, a resized view, and a rotated/flipped (rot/flp) view. Based on the views, a scale augmentation module and an angle acquisition module are constructed. In the first module, a Scale-Sensitive Consistency (SSC) loss and a Scale-Sensitive Feature Fusion (SSFF) module are introduced to improve the model's ability to estimate object scale. To achieve precise angle predictions, the second module employs symmetry-based self-supervised learning. Additionally, we introduce an end-to-end version that eliminates the pseudo-label generation process by integrating a detector branch and introduces an Instance-Aware Weighting (IAW) strategy to focus on high-quality predictions. We conducted extensive experiments on the DIOR-R, DOTA-v1.0/v1.5/v2.0, FAIR1M, STAR, and RSAR datasets. Across all these datasets, our method achieves an average improvement in accuracy of 3.56% in comparison to previous state-of-the-art methods. The code will be available at this https URL. 

**Abstract (ZH)**: 随着定向对象检测（OOD）需求的不断增加，近年来基于点监督的OOD研究引起了广泛关注。本文我们提出了一种更加强大的单点监督定向对象检测框架——PointOBB-v3。相较于现有方法，PointOBB-v3 无需额外先验信息生成伪旋转框，并且支持端到端框架。PointOBB-v3 通过整合三种独特的图像视图来实现其功能：原始视图、缩放视图和旋转/翻转视图（旋转/翻转视图）。基于这些视图，构建了一个尺度增强模块和角度获取模块。在第一个模块中，引入了尺度敏感一致性（SSC）损失和尺度敏感特征融合（SSFF）模块，以提高模型估计对象尺度的能力。为了实现精确的角度预测，第二个模块采用了基于对称性的自我监督学习方法。此外，我们还提出了一种端到端版本，通过集成检测支路并引入实例感知加权（IAW）策略，专注于高质量预测，从而省去了伪标签生成过程。我们在DIOR-R、DOTA-v1.0/v1.5/v2.0、FAIR1M、STAR 和 RSAR 数据集上进行了广泛的实验。在整个实验过程中，我们的方法在与以前的最优方法相比时，在所有这些数据集上平均提高了3.56%的准确率。相关的代码将在以下网址提供：[提供网址]。 

---
# GUI-Bee: Align GUI Action Grounding to Novel Environments via Autonomous Exploration 

**Title (ZH)**: GUI-Bee: 通过对新型环境进行自主探索对GUI操作定位进行alignment 

**Authors**: Yue Fan, Handong Zhao, Ruiyi Zhang, Yu Shen, Xin Eric Wang, Gang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2501.13896)  

**Abstract**: Graphical User Interface (GUI) action grounding is a critical step in GUI automation that maps language instructions to actionable elements on GUI screens. Most recent works of GUI action grounding leverage large GUI datasets to fine-tune MLLMs. However, the fine-tuning data always covers limited GUI environments, and we find the performance of the resulting model deteriorates in novel environments. We argue that the GUI grounding models should be further aligned to the novel environments to reveal their full potential, when the inference is known to involve novel environments, i.e., environments not used during the previous fine-tuning. To realize this, we first propose GUI-Bee, an MLLM-based autonomous agent, to collect high-quality, environment-specific data through exploration and then continuously fine-tune GUI grounding models with the collected data. Our agent leverages a novel Q-value-Incentive In-Context Reinforcement Learning (Q-ICRL) method to optimize exploration efficiency and data quality. Additionally, we introduce NovelScreenSpot, a benchmark for testing how well the data can help align GUI action grounding models to novel environments and demonstrate the effectiveness of data collected by GUI-Bee in the experiments. Furthermore, we conduct an ablation study to validate the Q-ICRL method in enhancing the efficiency of GUI-Bee. Project page: this https URL 

**Abstract (ZH)**: 图形用户界面（GUI）操作定位是GUI自动化中的一个关键步骤，它将语言指令映射到GUI屏幕上的可操作元素。最近的GUI操作定位工作大多借助大型GUI数据集对大规模语言模型（MLLM）进行微调。然而，微调数据总是涵盖有限的GUI环境，我们发现所得到的模型在新颖环境中表现不佳。我们认为，当推理涉及新颖环境时，GUI定位模型应进一步与新颖环境对齐，以充分发挥其潜力。具体而言，我们首先提出了一种基于MLLM的自主代理GUI-Bee，通过探索收集高质量的环境特定数据，然后不断使用收集的数据对GUI定位模型进行微调。我们的代理采用了新型的Q值激励上下文强化学习方法（Q-ICRL）来优化探索效率和数据质量。此外，我们引入了NovelScreenSpot，这是一个用于测试数据如何帮助将GUI操作定位模型与新颖环境对齐的基准，实验证明了GUI-Bee收集的数据的有效性。此外，我们进行了一系列消融研究，以验证Q-ICRL方法如何提高GUI-Bee的效率。项目主页：[此处链接] 

---
# Pix2Cap-COCO: Advancing Visual Comprehension via Pixel-Level Captioning 

**Title (ZH)**: Pix2Cap-COCO：基于像素级描述推进视觉理解 

**Authors**: Zuyao You, Junke Wang, Lingyu Kong, Bo He, Zuxuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2501.13893)  

**Abstract**: We present Pix2Cap-COCO, the first panoptic pixel-level caption dataset designed to advance fine-grained visual understanding. To achieve this, we carefully design an automated annotation pipeline that prompts GPT-4V to generate pixel-aligned, instance-specific captions for individual objects within images, enabling models to learn more granular relationships between objects and their contexts. This approach results in 167,254 detailed captions, with an average of 22.94 words per caption. Building on Pix2Cap-COCO, we introduce a novel task, panoptic segmentation-captioning, which challenges models to recognize instances in an image and provide detailed descriptions for each simultaneously. To benchmark this task, we design a robust baseline based on X-Decoder. The experimental results demonstrate that Pix2Cap-COCO is a particularly challenging dataset, as it requires models to excel in both fine-grained visual understanding and detailed language generation. Furthermore, we leverage Pix2Cap-COCO for Supervised Fine-Tuning (SFT) on large multimodal models (LMMs) to enhance their performance. For example, training with Pix2Cap-COCO significantly improves the performance of GPT4RoI, yielding gains in CIDEr +1.4%, ROUGE +0.4%, and SPICE +0.5% on Visual Genome dataset, and strengthens its region understanding ability on the ViP-BENCH, with an overall improvement of +5.1%, including notable increases in recognition accuracy +11.2% and language generation quality +22.2%. 

**Abstract (ZH)**: 我们提出了Pix2Cap-COCO，这是首个旨在促进精细视觉理解的全景像素级描述数据集。为了实现这一目标，我们精心设计了一个自动化标注管道，使用GPT-4V自动生成与像素对齐、实例特定的图像中标记对象的描述，从而使模型能够学习对象与其上下文之间的更精细关系。这一方法产生了167,254条详细的描述，平均每条描述包含22.94个单词。基于Pix2Cap-COCO，我们引入了一种新的任务——全景分割-描述，该任务挑战模型同时在图像中标识实例并提供详细描述。为了衡量这一任务，我们基于X-Decoder设计了一个稳健的基线模型。实验结果表明，Pix2Cap-COCO 是一个特别具有挑战性的数据集，因为它要求模型在精细视觉理解和详细语言生成方面均表现出色。此外，我们利用Pix2Cap-COCO 对大型多模态模型（LMMs）进行监督微调（SFT），以提升其性能。例如，使用Pix2Cap-COCO 进行训练显著提升了GPT4RoI 的性能，在Visual Genome 数据集上，CIDEr 提高了1.4%，ROUGE 提高了0.4%，SPICE 提高了0.5%；同时，其在ViP-BENCH 的区域理解能力也得到了增强，总体提升幅度为5.1%，包括识别准确率的显著提高（+11.2%）和语言生成质量的大幅提升（+22.2%）。 

---
# Exploring Finetuned Audio-LLM on Heart Murmur Features 

**Title (ZH)**: 探索微调音频大型语言模型在心脏杂音特征上的应用 

**Authors**: Adrian Florea, Xilin Jiang, Nima Mesgarani, Xiaofan Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2501.13884)  

**Abstract**: Large language models (LLMs) for audio have excelled in recognizing and analyzing human speech, music, and environmental sounds. However, their potential for understanding other types of sounds, particularly biomedical sounds, remains largely underexplored despite significant scientific interest. In this study, we focus on diagnosing cardiovascular diseases using phonocardiograms, i.e., heart sounds. Most existing deep neural network (DNN) paradigms are restricted to heart murmur classification (healthy vs unhealthy) and do not predict other acoustic features of the murmur such as timing, grading, harshness, pitch, and quality, which are important in helping physicians diagnose the underlying heart conditions. We propose to finetune an audio LLM, Qwen2-Audio, on the PhysioNet CirCor DigiScope phonocardiogram (PCG) dataset and evaluate its performance in classifying 11 expert-labeled murmur features. Additionally, we aim to achieve more noise-robust and generalizable system by exploring a preprocessing segmentation algorithm using an audio representation model, SSAMBA. Our results indicate that the LLM-based model outperforms state-of-the-art methods in 8 of the 11 features and performs comparably in the remaining 3. Moreover, the LLM successfully classifies long-tail murmur features with limited training data, a task that all previous methods have failed to classify. These findings underscore the potential of audio LLMs as assistants to human cardiologists in enhancing heart disease diagnosis. 

**Abstract (ZH)**: 大型语言模型（LLMs）在音频领域已经展现出在识别和分析人类语音、音乐和环境声方面的卓越能力。然而，这些模型在理解其他类型的声音，尤其是生物医学声音方面的发展潜力仍然未被充分探索，尽管这领域受到了广泛的研究兴趣。本研究专注于利用听诊器心音图（即心音）诊断心血管疾病。目前现有的大多数深度神经网络（DNN）范式仅限于心鸣音分类（健康 vs 不健康），而不预测心鸣音的其他声学特征，如时间、音级、尖锐度、音调和音质，这些都是帮助医生诊断潜在心脏状况的重要信息。我们提出将音频LLM Qwen2-Audio在PhysioNet CirCor DigiScope心音图（PCG）数据集上进行微调，并评估其在分类11个专家标注的心鸣音特征方面的性能。此外，我们希望通过使用音频表示模型SSAMBA探索预处理分割算法，力求构建更具有抗噪声能力和通用性的系统。研究结果表明，基于LLM的模型在11个特征中有8个方面比现有最佳方法表现更好，在剩余的3个特征上表现相当。此外，该LLM成功使用有限的训练数据对长尾心鸣音特征进行了分类，而此前所有方法都无法完成这一任务。这些发现强调了音频LLM作为心脏病学医生辅助工具在提升心脏病诊断方面的能力。 

---
# Autoencoders for Anomaly Detection are Unreliable 

**Title (ZH)**: 自动编码器在异常检测中的可靠性不佳 

**Authors**: Roel Bouman, Tom Heskes  

**Link**: [PDF](https://arxiv.org/pdf/2501.13864)  

**Abstract**: Autoencoders are frequently used for anomaly detection, both in the unsupervised and semi-supervised settings. They rely on the assumption that when trained using the reconstruction loss, they will be able to reconstruct normal data more accurately than anomalous data. Some recent works have posited that this assumption may not always hold, but little has been done to study the validity of the assumption in theory. In this work we show that this assumption indeed does not hold, and illustrate that anomalies, lying far away from normal data, can be perfectly reconstructed in practice. We revisit the theory of failure of linear autoencoders for anomaly detection by showing how they can perfectly reconstruct out of bounds, or extrapolate undesirably, and note how this can be dangerous in safety critical applications. We connect this to non-linear autoencoders through experiments on both tabular data and real-world image data, the two primary application areas of autoencoders for anomaly detection. 

**Abstract (ZH)**: 自编码器在无监督和半监督设置中经常被用于异常检测。它们依赖于一个假设，即通过重构损失进行训练时，它们能够更准确地重构正常数据，而不是异常数据。一些近期的研究表明，这一假设可能并不总是成立，但在理论上研究这一假设的有效性的工作并不多。在这项工作中，我们展示了这一假设实际上并不成立，并通过具体实例说明，异常数据，远离正常数据的，实际上可以在实践中被完美重构。我们通过对线性自编码器在异常检测中的失效理论进行重新审视，展示了它们如何能够完美重构超出范围的数据或产生不希望的外推，指出这一点在关键安全应用中可能非常危险。我们通过在表格数据和现实世界图像数据上的实验，将这一现象与非线性自编码器联系起来，这正是自编码器在异常检测中的两个主要应用领域。 

---
# Where Do You Go? Pedestrian Trajectory Prediction using Scene Features 

**Title (ZH)**: 《行人动向何方？基于场景特征的行人轨迹预测》

这个标题翻译成中文，既符合学术规范又保留了原文的意思。如果需要进一步的详细内容翻译或有其他问题，请告知。 

**Authors**: Mohammad Ali Rezaei, Fardin Ayar, Ehsan Javanmardi, Manabu Tsukada, Mahdi Javanmardi  

**Link**: [PDF](https://arxiv.org/pdf/2501.13848)  

**Abstract**: Accurate prediction of pedestrian trajectories is crucial for enhancing the safety of autonomous vehicles and reducing traffic fatalities involving pedestrians. While numerous studies have focused on modeling interactions among pedestrians to forecast their movements, the influence of environmental factors and scene-object placements has been comparatively underexplored. In this paper, we present a novel trajectory prediction model that integrates both pedestrian interactions and environmental context to improve prediction accuracy. Our approach captures spatial and temporal interactions among pedestrians within a sparse graph framework. To account for pedestrian-scene interactions, we employ advanced image enhancement and semantic segmentation techniques to extract detailed scene features. These scene and interaction features are then fused through a cross-attention mechanism, enabling the model to prioritize relevant environmental factors that influence pedestrian movements. Finally, a temporal convolutional network processes the fused features to predict future pedestrian trajectories. Experimental results demonstrate that our method significantly outperforms existing state-of-the-art approaches, achieving ADE and FDE values of 0.252 and 0.372 meters, respectively, underscoring the importance of incorporating both social interactions and environmental context in pedestrian trajectory prediction. 

**Abstract (ZH)**: 准确预测行人的轨迹对于提高自动驾驶车辆的安全性并减少涉及行人的交通事故至关重要。尽管已有大量研究集中于建模行人的相互作用以预测其运动轨迹，但环境因素和场景对象布局的影响相对较少被探索。本文提出了一种新的轨迹预测模型，该模型结合了行人的相互作用和环境上下文，以提高预测准确性。我们的方法在稀疏图框架内捕获行人之间的空间和时间相互作用。为了考虑行人与场景的相互作用，我们采用了先进的图像增强和语义分割技术来提取详细的场景特征。通过交叉注意力机制融合这些场景和交互特征，使模型能够优先考虑影响行人运动的相关环境因素。最后，时间卷积网络处理融合特征以预测未来行人的轨迹。实验结果表明，我们的方法显著优于现有的先进方法，分别实现了ADEV为0.252米和FDEV为0.372米的数值，强调了在行人轨迹预测中同时考虑社会互动和环境上下文的重要性。 

---
# Predicting Compact Phrasal Rewrites with Large Language Models for ASR Post Editing 

**Title (ZH)**: 使用大型语言模型预测紧凑短语重写以进行ASR后编辑 

**Authors**: Hao Zhang, Felix Stahlberg, Shankar Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2501.13831)  

**Abstract**: Large Language Models (LLMs) excel at rewriting tasks such as text style transfer and grammatical error correction. While there is considerable overlap between the inputs and outputs in these tasks, the decoding cost still increases with output length, regardless of the amount of overlap. By leveraging the overlap between the input and the output, Kaneko and Okazaki (2023) proposed model-agnostic edit span representations to compress the rewrites to save computation. They reported an output length reduction rate of nearly 80% with minimal accuracy impact in four rewriting tasks. In this paper, we propose alternative edit phrase representations inspired by phrase-based statistical machine translation. We systematically compare our phrasal representations with their span representations. We apply the LLM rewriting model to the task of Automatic Speech Recognition (ASR) post editing and show that our target-phrase-only edit representation has the best efficiency-accuracy trade-off. On the LibriSpeech test set, our method closes 50-60% of the WER gap between the edit span model and the full rewrite model while losing only 10-20% of the length reduction rate of the edit span model. 

**Abstract (ZH)**: 大型语言模型（LLMs）在文本风格转换和语法错误纠正等任务中表现出色。尽管这些任务的输入和输出之间存在很大的重叠，解码成本仍然会随着输出长度的增加而增加，无论重叠程度如何。通过利用输入和输出之间的重叠，Kaneko 和 Okazaki（2023）提出了模型无关的编辑短语表示方法，以压缩重写内容以节省计算资源。他们在四项重写任务中报告了将近80%的输出长度缩减率，同时对准确性几乎没有影响。本文中，我们提出了受基于短语的统计机器翻译启发的编辑短语表示方法。系统地比较了我们的短语表示方法与短语表示方法的优劣。我们将LLM重写模型应用于自动语音识别（ASR）后编辑任务，表明我们的目标短语仅编辑表示具有最佳的效率-准确性的权衡。在LibriSpeech测试集上，我们的方法将编辑短语模型与全重写模型之间的词错误率（WER）差距缩小了50-60%，而损失的长度缩减率仅为编辑短语模型的10-20%。 

---
# A space-decoupling framework for optimization on bounded-rank matrices with orthogonally invariant constraints 

**Title (ZH)**: 具有良好秩约束和正交不变约束的矩阵优化的空间解耦框架 

**Authors**: Yan Yang, Bin Gao, Ya-xiang Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2501.13830)  

**Abstract**: Imposing additional constraints on low-rank optimization has garnered growing interest. However, the geometry of coupled constraints hampers the well-developed low-rank structure and makes the problem intricate. To this end, we propose a space-decoupling framework for optimization on bounded-rank matrices with orthogonally invariant constraints. The ``space-decoupling" is reflected in several ways. We show that the tangent cone of coupled constraints is the intersection of tangent cones of each constraint. Moreover, we decouple the intertwined bounded-rank and orthogonally invariant constraints into two spaces, leading to optimization on a smooth manifold. Implementing Riemannian algorithms on this manifold is painless as long as the geometry of additional constraints is known. In addition, we unveil the equivalence between the reformulated problem and the original problem. Numerical experiments on real-world applications -- spherical data fitting, graph similarity measuring, low-rank SDP, model reduction of Markov processes, reinforcement learning, and deep learning -- validate the superiority of the proposed framework. 

**Abstract (ZH)**: 对低秩优化施加额外约束正逐渐引起广泛关注。然而，耦合约束的几何特性妨碍了低秩结构的发展，使问题变得复杂。为解决这一问题，我们提出了一个适用于有界秩矩阵和正交不变约束下的空间解耦优化框架。这种“空间解耦”体现在多个方面。我们证明，耦合约束的切锥是各自约束切锥的交集。此外，我们将交织的有界秩和正交不变约束分解成两个空间，使其优化问题能够在一个光滑流形上进行。只要了解额外约束的几何特性，就在该流形上实施黎曼优化算法就十分简便。此外，我们揭示了重新表述的问题与原始问题之间的等价性。在现实应用中的数值实验——球形数据拟合、图相似性度量、低秩半定规划、马尔可夫过程模型降低、强化学习和深度学习——验证了所提出框架的优势。 

---
# Hallucinations Can Improve Large Language Models in Drug Discovery 

**Title (ZH)**: hallucinations 可以提高药物发现中大型语言模型的能力 

**Authors**: Shuzhou Yuan, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2501.13824)  

**Abstract**: Concerns about hallucinations in Large Language Models (LLMs) have been raised by researchers, yet their potential in areas where creativity is vital, such as drug discovery, merits exploration. In this paper, we come up with the hypothesis that hallucinations can improve LLMs in drug discovery. To verify this hypothesis, we use LLMs to describe the SMILES string of molecules in natural language and then incorporate these descriptions as part of the prompt to address specific tasks in drug discovery. Evaluated on seven LLMs and five classification tasks, our findings confirm the hypothesis: LLMs can achieve better performance with text containing hallucinations. Notably, Llama-3.1-8B achieves an 18.35% gain in ROC-AUC compared to the baseline without hallucination. Furthermore, hallucinations generated by GPT-4o provide the most consistent improvements across models. Additionally, we conduct empirical analyses and a case study to investigate key factors affecting performance and the underlying reasons. Our research sheds light on the potential use of hallucinations for LLMs and offers new perspectives for future research leveraging LLMs in drug discovery. 

**Abstract (ZH)**: 研究人员对大型语言模型（LLMs）的幻觉 concerns提出了一定担忧，然而在涉及创造力的关键领域，如药物发现，它们的潜力仍值得探索。本文中，我们提出了这样一个假设：幻觉能够提升药物发现中的LLMs性能。为了验证这一假设，我们利用LLMs将分子的SMILES字符串描述为自然语言，并将这些描述纳入特定药物发现任务的提示中。我们在七种LLMs和五项分类任务上进行了评估，研究结果证实了这一假设：包含幻觉的文本能够提高LLMs的性能。值得注意的是，Llama-3.1-8B在没有幻觉的情况下基线性能上实现了18.35%的ROC-AUC增幅。此外，GPT-4o生成的幻觉在不同模型中提供了最一致的改进。我们还通过实证分析和案例研究探讨了影响性能的关键因素及背后的机制。本研究为利用幻觉提升LLMs的药物发现潜能提供了新的视角，并对未来利用LLMs进行药物发现的研究提供了新的思路。 

---
# Learning to Help in Multi-Class Settings 

**Title (ZH)**: 多类别设置中的学习辅助方法 

**Authors**: Yu Wu, Yansong Li, Zeyu Dong, Nitya Sathyavageeswaran, Anand D. Sarwate  

**Link**: [PDF](https://arxiv.org/pdf/2501.13810)  

**Abstract**: Deploying complex machine learning models on resource-constrained devices is challenging due to limited computational power, memory, and model retrainability. To address these limitations, a hybrid system can be established by augmenting the local model with a server-side model, where samples are selectively deferred by a rejector and then sent to the server for processing. The hybrid system enables efficient use of computational resources while minimizing the overhead associated with server usage. The recently proposed Learning to Help (L2H) model trains a server model given a fixed local (client) model, differing from the Learning to Defer (L2D) framework, which trains the client for a fixed (expert) server. In both L2D and L2H, the training includes learning a rejector at the client to determine when to query the server. In this work, we extend the L2H model from binary to multi-class classification problems and demonstrate its applicability in a number of different scenarios of practical interest in which access to the server may be limited by cost, availability, or policy. We derive a stage-switching surrogate loss function that is differentiable, convex, and consistent with the Bayes rule corresponding to the 0-1 loss for the L2H model. Experiments show that our proposed methods offer an efficient and practical solution for multi-class classification in resource-constrained environments. 

**Abstract (ZH)**: 在资源受限的设备上部署复杂的机器学习模型具有挑战性，由于其计算能力、内存和模型可重新训练的限制。为了应对这些限制，可以通过在本地模型中增加服务器端模型来建立一个混合系统，在这种系统中，拒识器选择性地将样本延迟处理，并将其发送到服务器进行处理。这种混合系统能够有效地利用计算资源，同时将与服务器使用的附加开销降到最低。最近提出的Learning to Help (L2H)模型是在固定本地（客户端）模型的基础上训练服务器模型，不同于Learning to Defer (L2D)框架，后者是针对固定专家服务器训练客户端。在L2D和L2H两种框架中，训练过程均包括在客户端学习一个拒识器，以确定何时查询服务器。在本文中，我们从二分类问题扩展到多分类问题，展示了L2H模型在多种实际应用场景中的应用性，其中服务器的访问受限于成本、可用性或政策。我们推导出一种可切换的替代损失函数，该函数是可微的、凸的，并且与其所对应的0-1损失的贝叶斯规则一致。实验结果显示，我们提出的方法为资源受限环境下的多分类问题提供了一种高效且实用的解决方案。 

---
# Parameter-Efficient Fine-Tuning for Foundation Models 

**Title (ZH)**: 基础模型的参数高效微调 

**Authors**: Dan Zhang, Tao Feng, Lilong Xue, Yuandong Wang, Yuxiao Dong, Jie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2501.13787)  

**Abstract**: This survey delves into the realm of Parameter-Efficient Fine-Tuning (PEFT) within the context of Foundation Models (FMs). PEFT, a cost-effective fine-tuning technique, minimizes parameters and computational complexity while striving for optimal downstream task performance. FMs, like ChatGPT, DALL-E, and LLaVA specialize in language understanding, generative tasks, and multimodal tasks, trained on diverse datasets spanning text, images, and videos. The diversity of FMs guides various adaptation strategies for PEFT. Therefore, this survey aims to provide a comprehensive overview of PEFT techniques applied to diverse FMs and address critical gaps in understanding the techniques, trends, and applications. We start by providing a detailed development of FMs and PEFT. Subsequently, we systematically review the key categories and core mechanisms of PEFT across diverse FMs to offer a comprehensive understanding of trends. We also explore the most recent applications across various FMs to demonstrate the versatility of PEFT, shedding light on the integration of systematic PEFT methods with a range of FMs. Furthermore, we identify potential research and development directions for improving PEFTs in the future. This survey provides a valuable resource for both newcomers and experts seeking to understand and use the power of PEFT across FMs. All reviewed papers are listed at \url{this https URL}. 

**Abstract (ZH)**: 本文探讨了基础模型（FMs）背景下参数高效微调（PEFT）的研究领域。PEFT 是一种成本效益高的微调技术，在减少参数量和计算复杂性的同时，力求实现最佳下游任务性能。像 ChatGPT、DALL-E 和 LLaVA 这样的 FMs 专长于语言理解、生成任务以及多模态任务，并在文本、图像和视频等多种数据集上进行训练。FMs 的多样性指导了适用于 PEFT 的多种适应策略。因此，本文旨在提供 PEFT 技术应用于多样 FMs 的全面概述，并解决理解这些技术、趋势和应用的关键空白。首先，我们将详细介绍 FMs 和 PEFT。随后，我们系统地回顾了 PEFT 在多样 FMs 上的关键类别和核心机制，以提供对这些趋势的全面理解。我们还探讨了 PEFT 在各种 FMs 上的最新应用，以展示 PEFT 的多功能性，并阐明系统 PEFT 方法与多种 FMs 的整合方式。此外，我们识别了改进 PEFT 的潜在研究和发展方向。本文为新手和专家提供了理解和应用 PEFT 的有用资源，以适用于多种 FMs。所有回顾的论文均可在 \url{此链接} 查看。 

---
# Defending against Adversarial Malware Attacks on ML-based Android Malware Detection Systems 

**Title (ZH)**: 针对基于机器学习的Android恶意软件检测系统所遭受的对抗性恶意软件攻击的防御方法 

**Authors**: Ping He, Lorenzo Cavallaro, Shouling Ji  

**Link**: [PDF](https://arxiv.org/pdf/2501.13782)  

**Abstract**: Android malware presents a persistent threat to users' privacy and data integrity. To combat this, researchers have proposed machine learning-based (ML-based) Android malware detection (AMD) systems. However, adversarial Android malware attacks compromise the detection integrity of the ML-based AMD systems, raising significant concerns. Existing defenses against adversarial Android malware provide protections against feature space attacks which generate adversarial feature vectors only, leaving protection against realistic threats from problem space attacks which generate real adversarial malware an open problem. In this paper, we address this gap by proposing ADD, a practical adversarial Android malware defense framework designed as a plug-in to enhance the adversarial robustness of the ML-based AMD systems against problem space attacks. Our extensive evaluation across various ML-based AMD systems demonstrates that ADD is effective against state-of-the-art problem space adversarial Android malware attacks. Additionally, ADD shows the defense effectiveness in enhancing the adversarial robustness of real-world antivirus solutions. 

**Abstract (ZH)**: 安卓恶意软件对用户的隐私安全和数据完整性构成持续威胁。为应对这一挑战，研究人员提出了基于机器学习（ML）的安卓恶意软件检测（AMD）系统。然而， adversarial Android恶意软件攻击破坏了基于ML的AMD系统的检测准确性，引发了重大担忧。现有的对抗安卓恶意软件的防护措施主要针对特征空间攻击，只能生成恶意特征向量，而对生成真实对抗恶意软件的问题空间攻击则缺乏有效的防护手段。本文通过提出一种名为ADD的实际对抗安卓恶意软件防御框架，解决了这一问题，该框架设计为插件形式，旨在增强基于ML的AMD系统对问题空间攻击的抗对抗性。我们对多种基于ML的AMD系统的广泛评估表明，ADD有效地抵御了最先进的问题空间对抗安卓恶意软件攻击。此外，ADD展示了在提高实际防病毒解决方案的抗对抗性方面的防御效果。 

---
# Not Every AI Problem is a Data Problem: We Should Be Intentional About Data Scaling 

**Title (ZH)**: 并非每一个AI问题都是数据问题：我们应有意识地对待数据扩展 

**Authors**: Tanya Rodchenko, Natasha Noy, Nino Scherrer, Jennifer Prendki  

**Link**: [PDF](https://arxiv.org/pdf/2501.13779)  

**Abstract**: While Large Language Models require more and more data to train and scale, rather than looking for any data to acquire, we should consider what types of tasks are more likely to benefit from data scaling. We should be intentional in our data acquisition. We argue that the topology of data itself informs which tasks to prioritize in data scaling, and shapes the development of the next generation of compute paradigms for tasks where data scaling is inefficient, or even insufficient. 

**Abstract (ZH)**: 当大型语言模型需要更多数据进行训练和扩展时，我们不应盲目地寻找任何可用数据，而应该考虑哪些类型的任务更有可能从数据规模扩大中受益。我们应该有意识地进行数据收集。我们认为，数据本身的拓扑结构指导了在数据规模扩大中应优先考虑哪些任务，并塑造那些从数据规模扩大中效率低下甚至不足以解决问题的任务的下一代计算范式的开发。 

---
# Tune In, Act Up: Exploring the Impact of Audio Modality-Specific Edits on Large Audio Language Models in Jailbreak 

**Title (ZH)**: 调频响应，积极行动：探索特定音频模态编辑对囚徒突破的大规模音频语言模型影响 

**Authors**: Erjia Xiao, Hao Cheng, Jing Shao, Jinhao Duan, Kaidi Xu, Le Yang, Jindong Gu, Renjing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2501.13772)  

**Abstract**: Large Language Models (LLMs) demonstrate remarkable zero-shot performance across various natural language processing tasks. The integration of multimodal encoders extends their capabilities, enabling the development of Multimodal Large Language Models that process vision, audio, and text. However, these capabilities also raise significant security concerns, as these models can be manipulated to generate harmful or inappropriate content through jailbreak. While extensive research explores the impact of modality-specific input edits on text-based LLMs and Large Vision-Language Models in jailbreak, the effects of audio-specific edits on Large Audio-Language Models (LALMs) remain underexplored. Hence, this paper addresses this gap by investigating how audio-specific edits influence LALMs inference regarding jailbreak. We introduce the Audio Editing Toolbox (AET), which enables audio-modality edits such as tone adjustment, word emphasis, and noise injection, and the Edited Audio Datasets (EADs), a comprehensive audio jailbreak benchmark. We also conduct extensive evaluations of state-of-the-art LALMs to assess their robustness under different audio edits. This work lays the groundwork for future explorations on audio-modality interactions in LALMs security. 

**Abstract (ZH)**: 大型语言模型（Large Language Models, LLMs）在各种自然语言处理任务中展现出卓越的零样本性能。通过集成多模态编码器，可以进一步扩展其功能，使开发出能够处理视觉、音频和文本信息的多模态大型语言模型成为可能。然而，这些能力也引发了显著的安全关切，因为这些模型可以通过“脱狱”（jailbreak）被操纵以生成有害或不适当的内容。虽然大量的研究探讨了特定模态输入编辑对文本型LLMs和大型视觉-语言模型“脱狱”影响，但特定于音频的编辑对大型音频-语言模型（Large Audio-Language Models, LALMs）的影响仍较少被研究。因此，本文通过探讨特定于音频的编辑如何影响LALMs在“脱狱”情况下的推断来填补这一空白。我们介绍了音频编辑工具箱（Audio Editing Toolbox, AET），它允许进行音调调整、词汇强调和噪声注入等音频模态编辑，并介绍了一个综合的音频“脱狱”基准数据集（Edited Audio Datasets, EADs）。此外，我们还进行了广泛评估，以评估当前最先进的LALMs在不同音频编辑下的鲁棒性。这项工作为未来在LALMs安全方面探索音频模态交互奠定了基础。 

---
# UGMathBench: A Diverse and Dynamic Benchmark for Undergraduate-Level Mathematical Reasoning with Large Language Models 

**Title (ZH)**: UGMathBench：面向本科生级数学推理的多样化和动态大型语言模型基准测试 

**Authors**: Xin Xu, Jiaxin Zhang, Tianhao Chen, Zitong Chao, Jishan Hu, Can Yang  

**Link**: [PDF](https://arxiv.org/pdf/2501.13766)  

**Abstract**: Large Language Models (LLMs) have made significant strides in mathematical reasoning, underscoring the need for a comprehensive and fair evaluation of their capabilities. However, existing benchmarks often fall short, either lacking extensive coverage of undergraduate-level mathematical problems or probably suffering from test-set contamination. To address these issues, we introduce UGMathBench, a diverse and dynamic benchmark specifically designed for evaluating undergraduate-level mathematical reasoning with LLMs. UGMathBench comprises 5,062 problems across 16 subjects and 111 topics, featuring 10 distinct answer types. Each problem includes three randomized versions, with additional versions planned for release as leading open-source LLMs become saturated in UGMathBench. Furthermore, we propose two key metrics: effective accuracy (EAcc), which measures the percentage of correctly solved problems across all three versions, and reasoning gap ($\Delta$), which assesses reasoning robustness by calculating the difference between the average accuracy across all versions and EAcc. Our extensive evaluation of 23 leading LLMs reveals that the highest EAcc achieved is 56.3\% by OpenAI-o1-mini, with large $\Delta$ values observed across different models. This highlights the need for future research aimed at developing "large reasoning models" with high EAcc and $\Delta = 0$. We anticipate that the release of UGMathBench, along with its detailed evaluation codes, will serve as a valuable resource to advance the development of LLMs in solving mathematical problems. 

**Abstract (ZH)**: 大型语言模型（Large Language Models, LLMs）在数学推理方面取得了显著进展，强调了对它们能力进行全面和公正评估的必要性。然而，现有的基准测试往往存在不足，要么覆盖范围不足，未能涵盖本科级别的数学问题，要么可能受到测试集污染的影响。为解决这些问题，我们引入了UGMathBench，这是一个针对LLMs评估本科级别数学推理的多样化和动态基准。UGMathBench包含来自16个学科和111个主题的5,062个问题，涵盖了10种不同的答案类型。每个问题包含三个随机版本，当领先的开源LLM在UGMathBench中饱和时，将计划发布更多版本。此外，我们提出了两个关键指标：有效准确率（EAcc），它衡量三个版本中正确解决问题的比例；推理间隙（$\Delta$），它通过计算所有版本平均准确率与EAcc之间的差值来评估推理的鲁棒性。我们对23个领先LLM的广泛评估表明，OpenAI-o1-mini实现了最高的有效准确率（56.3%），并且不同模型之间观察到较大的推理间隙（$\Delta$）。这表明需要未来研究，以开发具有高EAcc和$\Delta = 0$的“大型推理模型”。我们期待UGMathBench及其详细评估代码的发布能成为推进LLMs解决数学问题发展的重要资源。 

---
# Integrating Causality with Neurochaos Learning: Proposed Approach and Research Agenda 

**Title (ZH)**: 将以下论文内容或标题翻译成中文，并符合学术规范：

"将因果关系与神经混沌学习整合：提出的方法与研究议程" 

**Authors**: Nanjangud C. Narendra, Nithin Nagaraj  

**Link**: [PDF](https://arxiv.org/pdf/2501.13763)  

**Abstract**: Deep learning implemented via neural networks, has revolutionized machine learning by providing methods for complex tasks such as object detection/classification and prediction. However, architectures based on deep neural networks have started to yield diminishing returns, primarily due to their statistical nature and inability to capture causal structure in the training data. Another issue with deep learning is its high energy consumption, which is not that desirable from a sustainability perspective.
Therefore, alternative approaches are being considered to address these issues, both of which are inspired by the functioning of the human brain. One approach is causal learning, which takes into account causality among the items in the dataset on which the neural network is trained. It is expected that this will help minimize the spurious correlations that are prevalent in the learned representations of deep neural networks. The other approach is Neurochaos Learning, a recent development, which draws its inspiration from the nonlinear chaotic firing intrinsic to neurons in biological neural networks (brain/central nervous system). Both approaches have shown improved results over just deep learning alone.
To that end, in this position paper, we investigate how causal and neurochaos learning approaches can be integrated together to produce better results, especially in domains that contain linked data. We propose an approach for this integration to enhance classification, prediction and reinforcement learning. We also propose a set of research questions that need to be investigated in order to make this integration a reality. 

**Abstract (ZH)**: 深度学习通过神经网络实现，已经通过提供用于复杂任务（如物体检测/分类和预测）的方法，极大地革新了机器学习。然而，基于深度神经网络的架构已经开始出现回报递减的现象，主要原因在于它们的统计性质以及在训练数据中捕捉因果结构的不足。另一个问题是深度学习的高能耗，这从可持续性的角度来看并不是一个值得提倡的特点。

因此，人们开始考虑采用替代方法来解决这些问题，这两种方法都受到人类大脑功能的启发。一种方法是因果学习，这种方法考虑了训练神经网络所用数据集中的项之间的因果关系。预计这将有助于减少深度神经网络学习表示中普遍存在的虚假相关性。另一种方法是神经混沌学习，这是一种最近的发展，灵感来源于生物神经网络（大脑/中枢神经系统中）固有的非线性混沌放电。这两种方法在单独使用深度学习时都显示出更好的效果。

为此，在本文中，我们探讨了如何将因果学习和神经混沌学习方法结合起来，以在包含关联数据的领域中获得更好的结果。我们提出了一种集成这两种方法以增强分类、预测和强化学习的方法。我们还提出了需要进一步研究的一系列研究问题，以使这种集成成为现实。 

---
# 2-Tier SimCSE: Elevating BERT for Robust Sentence Embeddings 

**Title (ZH)**: 两层级 SimCSE：提升 BERT 以获得稳健的句向量表示 

**Authors**: Yumeng Wang, Ziran Zhou, Junjin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.13758)  

**Abstract**: Effective sentence embeddings that capture semantic nuances and generalize well across diverse contexts are crucial for natural language processing tasks. We address this challenge by applying SimCSE (Simple Contrastive Learning of Sentence Embeddings) using contrastive learning to fine-tune the minBERT model for sentiment analysis, semantic textual similarity (STS), and paraphrase detection. Our contributions include experimenting with three different dropout techniques, namely standard dropout, curriculum dropout, and adaptive dropout, to tackle overfitting, proposing a novel 2-Tier SimCSE Fine-tuning Model that combines both unsupervised and supervised SimCSE on STS task, and exploring transfer learning potential for Paraphrase and SST tasks. Our findings demonstrate the effectiveness of SimCSE, with the 2-Tier model achieving superior performance on the STS task, with an average test score of 0.742 across all three downstream tasks. The results of error analysis reveals challenges in handling complex sentiments and reliance on lexical overlap for paraphrase detection, highlighting areas for future research. The ablation study revealed that removing Adaptive Dropout in the Single-Task Unsupervised SimCSE Model led to improved performance on the STS task, indicating overfitting due to added parameters. Transfer learning from SimCSE models on Paraphrase and SST tasks did not enhance performance, suggesting limited transferability of knowledge from the STS task. 

**Abstract (ZH)**: 有效地捕捉语义细微差异并在多种上下文中泛化的句子嵌入对于自然语言处理任务至关重要。我们通过应用SimCSE（简单对比学习的句子嵌入）来解决这一挑战，使用对比学习微调minBERT模型，用于情感分析、语义文本相似度（STS）和同义句检测。我们的贡献包括尝试了三种不同的 dropout 技术，分别是标准 dropout、渐进式 dropout 和自适应 dropout，以应对过拟合问题；提出了结合无监督和监督SimCSE的两层SimCSE微调模型，专门针对STS任务；并探索了SimCSE在同义句和情感分析任务中的迁移学习潜力。研究表明，SimCSE的有效性，在两层模型在所有三个下游任务中的平均测试得分为0.742，特别是在STS任务上表现出色。错误分析的结果揭示了处理复杂情感的挑战和在同义句检测中对词汇重叠的依赖，突显了未来研究的领域。消融研究表明，从单任务无监督SimCSE模型中移除自适应dropout提高了STS任务的表现，表明由于增加的参数导致过拟合。来自STS任务的SimCSE模型的迁移学习并未提升同义句和情感分析任务的表现，表明STS任务的知识转移能力有限。 

---
# Solving the long-tailed distribution problem by exploiting the synergies and balance of different techniques 

**Title (ZH)**: 通过利用不同技术的协同效应和平衡来解决长尾分布问题 

**Authors**: Ziheng Wang, Toni Lassila, Sharib Ali  

**Link**: [PDF](https://arxiv.org/pdf/2501.13756)  

**Abstract**: In real-world data, long-tailed data distribution is common, making it challenging for models trained on empirical risk minimisation to learn and classify tail classes effectively. While many studies have sought to improve long tail recognition by altering the data distribution in the feature space and adjusting model decision boundaries, research on the synergy and corrective approach among various methods is limited. Our study delves into three long-tail recognition techniques: Supervised Contrastive Learning (SCL), Rare-Class Sample Generator (RSG), and Label-Distribution-Aware Margin Loss (LDAM). SCL enhances intra-class clusters based on feature similarity and promotes clear inter-class separability but tends to favour dominant classes only. When RSG is integrated into the model, we observed that the intra-class features further cluster towards the class centre, which demonstrates a synergistic effect together with SCL's principle of enhancing intra-class clustering. RSG generates new tail features and compensates for the tail feature space squeezed by SCL. Similarly, LDAM is known to introduce a larger margin specifically for tail classes; we demonstrate that LDAM further bolsters the model's performance on tail classes when combined with the more explicit decision boundaries achieved by SCL and RSG. Furthermore, SCL can compensate for the dominant class accuracy sacrificed by RSG and LDAM. Our research emphasises the synergy and balance among the three techniques, with each amplifying the strengths of the others and mitigating their shortcomings. Our experiment on long-tailed distribution datasets, using an end-to-end architecture, yields competitive results by enhancing tail class accuracy without compromising dominant class performance, achieving a balanced improvement across all classes. 

**Abstract (ZH)**: 在实际数据中，长尾分布现象普遍存在，这使得基于经验风险最小化训练的模型在学习和分类尾部类别时面临挑战。尽管有许多研究通过改变特征空间的数据分布和调整模型决策边界来改善长尾识别，但在各种方法之间的协同作用和纠正方法方面的研究仍然有限。本研究探讨了三种长尾识别技术：监督对比学习（SCL）、罕见类样本生成器（RSG）和标签分布感知边界损失（LDAM）。

SCL 基于特征相似性增强类内聚簇，并促进类间清晰分离，但倾向于偏向主导类别。当将 RSG 整合进模型后，我们观察到类内特征进一步向类中心聚集，这体现了与 SCL 提高类内聚簇原则的协同效应。RSG 生成新的尾部特征，并补偿 SCL 压缩的尾部特征空间。同样，LDAM 知名于为尾部类别引入更大的边际，我们证明了当与 SCL 和 RSG 实现的更明确决策边界结合使用时，LDAM 进一步提升了模型在尾部类别的性能。此外，SCL 可以弥补 RSG 和 LDAM 在主导类别准确性方面做出的牺牲。

本研究强调了这三种技术之间的协同作用和平衡，每一项技术都放大了其他技术的优势，并缓解了它们的不足。我们在结合使用 SCL、RSG 和 LDAM 的端到端架构中使用长尾分布数据集进行实验，通过提升尾部类别的准确性而不牺牲主导类别的性能，实现了所有类别上的均衡改进，取得了具有竞争力的结果。 

---
# EICopilot: Search and Explore Enterprise Information over Large-scale Knowledge Graphs with LLM-driven Agents 

**Title (ZH)**: EICopilot：使用LLM驱动代理在大规模知识图中搜索和探索企业信息 

**Authors**: Yuhui Yun, Huilong Ye, Xinru Li, Ruojia Li, Jingfeng Deng, Li Li, Haoyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2501.13746)  

**Abstract**: The paper introduces EICopilot, an novel agent-based solution enhancing search and exploration of enterprise registration data within extensive online knowledge graphs like those detailing legal entities, registered capital, and major shareholders. Traditional methods necessitate text-based queries and manual subgraph explorations, often resulting in time-consuming processes. EICopilot, deployed as a chatbot via Baidu Enterprise Search, improves this landscape by utilizing Large Language Models (LLMs) to interpret natural language queries. This solution automatically generates and executes Gremlin scripts, providing efficient summaries of complex enterprise relationships. Distinct feature a data pre-processing pipeline that compiles and annotates representative queries into a vector database of examples for In-context learning (ICL), a comprehensive reasoning pipeline combining Chain-of-Thought with ICL to enhance Gremlin script generation for knowledge graph search and exploration, and a novel query masking strategy that improves intent recognition for heightened script accuracy. Empirical evaluations demonstrate the superior performance of EICopilot, including speed and accuracy, over baseline methods, with the \emph{Full Mask} variant achieving a syntax error rate reduction to as low as 10.00% and an execution correctness of up to 82.14%. These components collectively contribute to superior querying capabilities and summarization of intricate datasets, positioning EICopilot as a groundbreaking tool in the exploration and exploitation of large-scale knowledge graphs for enterprise information search. 

**Abstract (ZH)**: 本文介绍了EICopilot，一种基于代理的新颖解决方案，该方案增强了在广泛的在线知识图谱（如法律实体、注册资本和主要股东详情）中搜索和探索企业注册数据的能力。传统方法需要基于文本的查询和手动的子图探索，这往往会导致耗时的过程。通过在百度企业搜索中部署聊天机器人的方式，EICopilot 利用大型语言模型（LLMs）解释自然语言查询，从而自动生成和执行Gremlin脚本，提供高效的复杂企业关系总结。该解决方案具有以下特点：一个数据预处理管道，用于编译和注释代表性查询以构成语境学习（ICL）的向量数据库示例；一个综合推理管道，结合了有问答推理（Chain-of-Thought）和ICL，以增强Gremlin脚本生成，用于知识图搜索和探索；以及一种新颖的查询掩码策略，以提高意图识别并增强脚本准确性。实验证明，EICopilot 在速度和准确性方面优于基线方法，其中“全掩码”变体将语法错误率降低到最低10.00%，执行正确性高达82.14%。这些组件共同提高了查询能力和复杂数据集的总结能力，将EICopilot 作为探索和利用大规模知识图谱进行企业信息搜索的开创性工具。 

---
# Pseudocode-Injection Magic: Enabling LLMs to Tackle Graph Computational Tasks 

**Title (ZH)**: 伪代码注入魔力：使大模型能够应对图计算任务 

**Authors**: Chang Gong, Wanrui Bian, Zhijie Zhang, Weiguo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2501.13731)  

**Abstract**: Graph computational tasks are inherently challenging and often demand the development of advanced algorithms for effective solutions. With the emergence of large language models (LLMs), researchers have begun investigating their potential to address these tasks. However, existing approaches are constrained by LLMs' limited capability to comprehend complex graph structures and their high inference costs, rendering them impractical for handling large-scale graphs. Inspired by human approaches to graph problems, we introduce a novel framework, PIE (Pseudocode-Injection-Enhanced LLM Reasoning for Graph Computational Tasks), which consists of three key steps: problem understanding, prompt design, and code generation. In this framework, LLMs are tasked with understanding the problem and extracting relevant information to generate correct code. The responsibility for analyzing the graph structure and executing the code is delegated to the interpreter. We inject task-related pseudocodes into the prompts to further assist the LLMs in generating efficient code. We also employ cost-effective trial-and-error techniques to ensure that the LLM-generated code executes correctly. Unlike other methods that require invoking LLMs for each individual test case, PIE only calls the LLM during the code generation phase, allowing the generated code to be reused and significantly reducing inference costs. Extensive experiments demonstrate that PIE outperforms existing baselines in terms of both accuracy and computational efficiency. 

**Abstract (ZH)**: 图计算任务本质上具有挑战性，通常需要开发高级算法以实现有效解决。随着大型语言模型（LLMs）的出现，研究人员开始探讨其在解决这些问题方面的潜力。然而，现有的方法受限于LLMs理解复杂图结构的能力有限以及高昂的推理成本，这使得它们在处理大规模图时 impractical。受人类解决图问题方法的启发，我们提出了一种名为PIE（Pseudocode-Injection-Enhanced LLM Reasoning for Graph Computational Tasks）的新框架，该框架包含三个关键步骤：问题理解、提示设计和代码生成。在该框架中，LLMs的任务是理解问题并提取相关信息以生成正确的代码。分析图结构和执行代码的责任被委托给解释器。为了进一步帮助LLMs生成高效的代码，我们在提示中注入了相关任务的伪代码。我们还采用经济有效的试错技术，以确保LLM生成的代码能够正确执行。与需要为每个单独的测试案例调用LLM的方法不同，PIE在代码生成阶段仅调用LLM，从而使生成的代码可以重用，并显著降低了推理成本。广泛的经验表明，PIE在准确性和计算效率方面都优于现有基线方法。 

---
# Scalable Safe Multi-Agent Reinforcement Learning for Multi-Agent System 

**Title (ZH)**: 可扩展的安全多智能体强化学习在多智能体系统中的应用 

**Authors**: Haikuo Du, Fandi Gou, Yunze Cai  

**Link**: [PDF](https://arxiv.org/pdf/2501.13727)  

**Abstract**: Safety and scalability are two critical challenges faced by practical Multi-Agent Systems (MAS). However, existing Multi-Agent Reinforcement Learning (MARL) algorithms that rely solely on reward shaping are ineffective in ensuring safety, and their scalability is rather limited due to the fixed-size network output. To address these issues, we propose a novel framework, Scalable Safe MARL (SS-MARL), to enhance the safety and scalability of MARL methods. Leveraging the inherent graph structure of MAS, we design a multi-layer message passing network to aggregate local observations and communications of varying sizes. Furthermore, we develop a constrained joint policy optimization method in the setting of local observation to improve safety. Simulation experiments demonstrate that SS-MARL achieves a better trade-off between optimality and safety compared to baselines, and its scalability significantly outperforms the latest methods in scenarios with a large number of agents. The feasibility of our method is also verified by hardware implementation with Mecanum-wheeled vehicles. 

**Abstract (ZH)**: 安全性与可扩展性是实际多代理系统（MAS）面临的两个关键挑战。现有的依赖于奖励塑造的多代理强化学习（MARL）算法在保障安全性方面效果不佳，且由于网络输出的固定大小，其可扩展性也受到了限制。为了解决这些问题，我们提出了一种新的框架——可扩展安全MARL（SS-MARL），旨在提升MARL方法的安全性和可扩展性。借助MAS固有的图结构，我们设计了一种多层消息传递网络来聚合不同大小的局部观察和通信。此外，我们在局部观察的情境下开发了一种约束联合策略优化方法，以提高安全性。模拟实验表明，SS-MARL在安全性和最优性之间的权衡优于基线方法，并且在大量代理的场景中，其可扩展性显著优于最新方法。我们通过Mecanum轮车的硬件实现也验证了该方法的可行性。 

---
# You Only Crash Once v2: Perceptually Consistent Strong Features for One-Stage Domain Adaptive Detection of Space Terrain 

**Title (ZH)**: only Crash Once v2：基于知觉一致的强大特征之一阶段领域适应性空间地形检测 

**Authors**: Timothy Chase Jr, Christopher Wilson, Karthik Dantu  

**Link**: [PDF](https://arxiv.org/pdf/2501.13725)  

**Abstract**: The in-situ detection of planetary, lunar, and small-body surface terrain is crucial for autonomous spacecraft applications, where learning-based computer vision methods are increasingly employed to enable intelligence without prior information or human intervention. However, many of these methods remain computationally expensive for spacecraft processors and prevent real-time operation. Training of such algorithms is additionally complex due to the scarcity of labeled data and reliance on supervised learning approaches. Unsupervised Domain Adaptation (UDA) offers a promising solution by facilitating model training with disparate data sources such as simulations or synthetic scenes, although UDA is difficult to apply to celestial environments where challenging feature spaces are paramount. To alleviate such issues, You Only Crash Once (YOCOv1) has studied the integration of Visual Similarity-based Alignment (VSA) into lightweight one-stage object detection architectures to improve space terrain UDA. Although proven effective, the approach faces notable limitations, including performance degradations in multi-class and high-altitude scenarios. Building upon the foundation of YOCOv1, we propose novel additions to the VSA scheme that enhance terrain detection capabilities under UDA, and our approach is evaluated across both simulated and real-world data. Our second YOCO rendition, YOCOv2, is capable of achieving state-of-the-art UDA performance on surface terrain detection, where we showcase improvements upwards of 31% compared with YOCOv1 and terrestrial state-of-the-art. We demonstrate the practical utility of YOCOv2 with spacecraft flight hardware performance benchmarking and qualitative evaluation of NASA mission data. 

**Abstract (ZH)**: 原英文论文内容或标题的中文翻译如下，力求符合学术规范：

对行星、月球和小行星表面地形的原位检测对于自主航天器应用至关重要，其中基于学习的计算机视觉方法正越来越多地被用来实现无需先验信息或人类干预的智能。然而，这些方法对于航天器处理器来说仍具有计算成本高昂的问题，并阻碍了实时操作。由于标注数据稀缺以及依赖于监督学习方法，训练这些算法也十分复杂。无监督领域适应（UDA）提供了一种有前景的解决方案，通过利用模拟数据或合成场景等异质数据源进行模型训练，但其在天体环境中难于应用，因为这些环境中特征空间具有挑战性。为缓解上述问题，YOCOv1研究了将基于视觉相似性对齐（VSA）的方案集成到轻量级的单阶段目标检测架构中，以提高空间地形UDA性能。尽管这种方法已被证明有效，但它存在一些明显的局限性，特别是在多类和高海拔场景中性能有所下降。在此基础上，我们提出了一种对VSA方案的新改进，旨在提高在UDA条件下的地形检测能力。我们的方法既在模拟数据中也在中国探测器实际飞行硬件性能基准测试中进行了评估。我们的YOCO的第二个版本YOCOv2在表面地形检测中实现了最先进的UDA性能，与YOCOv1和陆地上的最先进的方法相比，性能提高了高达31%。我们通过NASA任务数据的定性和定量评估，展示了YOCOv2的实际应用价值。 

---
# Musical ethnocentrism in Large Language Models 

**Title (ZH)**: 大型语言模型中的音乐民族中心主义 

**Authors**: Anna Kruspe  

**Link**: [PDF](https://arxiv.org/pdf/2501.13720)  

**Abstract**: Large Language Models (LLMs) reflect the biases in their training data and, by extension, those of the people who created this training data. Detecting, analyzing, and mitigating such biases is becoming a focus of research. One type of bias that has been understudied so far are geocultural biases. Those can be caused by an imbalance in the representation of different geographic regions and cultures in the training data, but also by value judgments contained therein. In this paper, we make a first step towards analyzing musical biases in LLMs, particularly ChatGPT and Mixtral. We conduct two experiments. In the first, we prompt LLMs to provide lists of the "Top 100" musical contributors of various categories and analyze their countries of origin. In the second experiment, we ask the LLMs to numerically rate various aspects of the musical cultures of different countries. Our results indicate a strong preference of the LLMs for Western music cultures in both experiments. 

**Abstract (ZH)**: 大型语言模型（LLMs）在其训练数据中反映了偏差，并且这些偏差在某种程度上也反映了数据创造者的价值观和偏见。检测、分析和减轻这些偏差已成为研究的重点。到目前为止，地理文化偏差的研究还较少。这种偏差可以由训练数据中不同地理区域和文化的表现不平衡引起，也可以由其中包含的价值判断引起。在本文中，我们首次尝试分析LLMs中的音乐偏见，特别是针对ChatGPT和Mixtral进行。我们进行了两项实验。在第一个实验中，我们促使LLMs提供各种类别的“Top 100”音乐贡献者列表，并分析它们的国籍。在第二个实验中，我们要求LLMs对不同国家的音乐文化的各个方面进行数值评分。我们的结果显示，在两个实验中，LLMs都表现出对西方音乐文化的强烈偏好。 

---
# Skin Disease Detection and Classification of Actinic Keratosis and Psoriasis Utilizing Deep Transfer Learning 

**Title (ZH)**: 利用深度迁移学习进行光化性角化病和银屑病的皮肤疾病检测与分类 

**Authors**: Fahud Ahmmed, Md. Zaheer Raihan, Kamnur Nahar, D.M. Asadujjaman, Md. Mahfujur Rahman, Abdullah Tamim  

**Link**: [PDF](https://arxiv.org/pdf/2501.13713)  

**Abstract**: Skin diseases can arise from infections, allergies, genetic factors, autoimmune disorders, hormonal imbalances, or environmental triggers such as sun damage and pollution. Some skin diseases, such as Actinic Keratosis and Psoriasis, can be fatal if not treated in time. Early identification is crucial, but the diagnostic methods for these conditions are often expensive and not widely accessible. In this study, we propose a novel and efficient method for diagnosing skin diseases using deep learning techniques. This approach employs a modified VGG16 Convolutional Neural Network (CNN) model. The model includes several convolutional layers and utilizes ImageNet weights with modified top layers. The top layer is updated with fully connected layers and a final softmax activation layer to classify skin diseases. The dataset used, titled "Skin Disease Dataset," is publicly available. While the VGG16 architecture does not include data augmentation by default, preprocessing techniques such as rotation, shifting, and zooming were applied to augment the data prior to model training. The proposed methodology achieved 90.67% accuracy using the modified VGG16 model, demonstrating its reliability in classifying skin diseases. The promising results highlight the potential of this approach for real-world applications. 

**Abstract (ZH)**: 皮肤疾病可能源于感染、过敏、遗传因素、自身免疫疾病、激素失衡或环境诱因，如紫外线损伤和污染。某些皮肤疾病，如日光性角化病（Actinic Keratosis）和银屑病（Psoriasis），如果不及时治疗，可能会导致致命后果。早期识别至关重要，但这些状况的诊断方法通常成本高昂且普及程度不高。在本研究中，我们提出了一种使用深度学习技术的新颖且高效的皮肤疾病诊断方法。该方法采用改进的VGG16卷积神经网络（CNN）模型。该模型包括多层卷积层，并使用带有修改顶层的ImageNet权重。顶层通过全连接层和最终的Softmax激活层来分类皮肤疾病。所使用的数据集名为“皮肤疾病数据集”，已公开可用。尽管VGG16架构默认不包括数据增广，但通过旋转、平移和缩放等预处理技术在模型训练前对数据进行了增强。所提出的方法使用改进的VGG16模型实现了90.67%的准确率，证明了其在皮肤疾病分类方面具有可靠性。这些令人鼓舞的结果强调了该方法在实际应用中的潜在价值。 

---
# YOLO11-JDE: Fast and Accurate Multi-Object Tracking with Self-Supervised Re-ID 

**Title (ZH)**: YOLO11-JDE：快速且准确的多目标跟踪与自我监督的行人重识别 

**Authors**: Iñaki Erregue, Kamal Nasrollahi, Sergio Escalera  

**Link**: [PDF](https://arxiv.org/pdf/2501.13710)  

**Abstract**: We introduce YOLO11-JDE, a fast and accurate multi-object tracking (MOT) solution that combines real-time object detection with self-supervised Re-Identification (Re-ID). By incorporating a dedicated Re-ID branch into YOLO11s, our model performs Joint Detection and Embedding (JDE), generating appearance features for each detection. The Re-ID branch is trained in a fully self-supervised setting while simultaneously training for detection, eliminating the need for costly identity-labeled datasets. The triplet loss, with hard positive and semi-hard negative mining strategies, is used for learning discriminative embeddings. Data association is enhanced with a custom tracking implementation that successfully integrates motion, appearance, and location cues. YOLO11-JDE achieves competitive results on MOT17 and MOT20 benchmarks, surpassing existing JDE methods in terms of FPS and using up to ten times fewer parameters. Thus, making our method a highly attractive solution for real-world applications. 

**Abstract (ZH)**: 我们将介绍YOLO11-JDE，这是一种结合了实时目标检测和自监督再识别（Re-ID）的快速准确多目标跟踪（MOT）解决方案。通过将专用的Re-ID分支纳入YOLO11s中，我们的模型实现了联合检测和嵌入（Joint Detection and Embedding, JDE），生成每个检测的外观特征。Re-ID分支在完全自监督的设置中进行训练，同时进行检测训练，从而消除了对昂贵的身份标记数据集的需求。利用具有艰难正样本和半艰难负样本挖掘策略的三重损失来学习具有区分性的嵌入。通过定制的跟踪实现增强数据关联，该实现成功地结合了运动、外观和位置线索。在MOT17和MOT20基准测试中，YOLO11-JDE取得了竞争力的表现，并在每秒帧数（FPS）方面超越了现有的JDE方法，使用的参数量最多减少了一个数量级。因此，我们的方法成为适用于实际应用的高性价比解决方案。 

---
# EventVL: Understand Event Streams via Multimodal Large Language Model 

**Title (ZH)**: 事件VL：通过多模态大型语言模型理解事件流 

**Authors**: Pengteng Li, Yunfan Lu, Pinghao Song, Wuyang Li, Huizai Yao, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2501.13707)  

**Abstract**: The event-based Vision-Language Model (VLM) recently has made good progress for practical vision tasks. However, most of these works just utilize CLIP for focusing on traditional perception tasks, which obstruct model understanding explicitly the sufficient semantics and context from event streams. To address the deficiency, we propose EventVL, the first generative event-based MLLM (Multimodal Large Language Model) framework for explicit semantic understanding. Specifically, to bridge the data gap for connecting different modalities semantics, we first annotate a large event-image/video-text dataset, containing almost 1.4 million high-quality pairs of data, which enables effective learning across various scenes, e.g., drive scene or human motion. After that, we design Event Spatiotemporal Representation to fully explore the comprehensive information by diversely aggregating and segmenting the event stream. To further promote a compact semantic space, Dynamic Semantic Alignment is introduced to improve and complete sparse semantic spaces of events. Extensive experiments show that our EventVL can significantly surpass existing MLLM baselines in event captioning and scene description generation tasks. We hope our research could contribute to the development of the event vision community. 

**Abstract (ZH)**: 基于事件的视觉-语言模型（VLM）近期在实际视觉任务中取得了良好进展。然而，大多数这些工作仅仅利用CLIP来专注于传统的感知任务，这阻碍了模型从事件流中明确理解足够的语义和上下文。为了解决这一缺陷，我们提出了一种名为EventVL的生成型事件基多模大型语言模型（Multimodal Large Language Model, MLLM）框架，旨在实现显式的语义理解。具体而言，为了弥合跨不同模态语义的数据缺口，我们首先标注了一个包含近140万高质量数据对的大规模事件-图像/视频-文本数据集，这些数据使模型能够有效地在各种场景中学习，例如驾驶场景或人体动作。在此基础上，我们设计了一种事件时空表示方法，通过多样化的事件流聚合与分割，充分探索全面的信息。为进一步促进紧凑的语义空间，我们引入了动态语义对齐，以改善和补充事件的稀疏语义空间。通过广泛的实验，我们发现我们的EventVL在事件描述和场景描述生成任务中显著超越了现有的MLLM基线模型。我们希望我们的研究能够为事件视觉社区的发展做出贡献。 

---
# Training-Free Consistency Pipeline for Fashion Repose 

**Title (ZH)**: 训练-free一致性管道用于服饰姿态估计 

**Authors**: Potito Aghilar, Vito Walter Anelli, Michelantonio Trizio, Tommaso Di Noia  

**Link**: [PDF](https://arxiv.org/pdf/2501.13692)  

**Abstract**: Recent advancements in diffusion models have significantly broadened the possibilities for editing images of real-world objects. However, performing non-rigid transformations, such as changing the pose of objects or image-based conditioning, remains challenging. Maintaining object identity during these edits is difficult, and current methods often fall short of the precision needed for industrial applications, where consistency is critical. Additionally, fine-tuning diffusion models requires custom training data, which is not always accessible in real-world scenarios. This work introduces FashionRepose, a training-free pipeline for non-rigid pose editing specifically designed for the fashion industry. The approach integrates off-the-shelf models to adjust poses of long-sleeve garments, maintaining identity and branding attributes. FashionRepose uses a zero-shot approach to perform these edits in near real-time, eliminating the need for specialized training. consistent image editing. The solution holds potential for applications in the fashion industry and other fields demanding identity preservation in image editing. 

**Abstract (ZH)**: 近年来，扩散模型的发展极大地拓宽了编辑现实世界物体图像的可能性。然而，执行非刚性变换（如改变物体的姿态或基于图像的条件处理）仍然具有挑战性。在这些编辑过程中保持对象身份是一项困难的任务，当前的方法往往不能满足工业应用中所需的精度，尤其是在这些应用中一致性至关重要。此外，微调扩散模型需要定制的训练数据，而在现实世界的场景中，这些数据并不总是可获取的。本研究引入了FashionRepose，这是一种无需训练的流水线，专门用于服装行业的非刚性姿态编辑。该方法整合了现成的模型来调整长袖服装的姿态，同时保持其身份和品牌属性。FashionRepose 采用零样本方法在接近实时的情况下执行这些编辑，从而消除了需要专门训练的需求。该解决方案在时尚行业及其他需要图像编辑中保持身份一致性的领域具有潜在应用价值。 

---
# Question Answering on Patient Medical Records with Private Fine-Tuned LLMs 

**Title (ZH)**: 使用私有微调大语言模型在患者医疗记录上的问答 

**Authors**: Sara Kothari, Ayush Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2501.13687)  

**Abstract**: Healthcare systems continuously generate vast amounts of electronic health records (EHRs), commonly stored in the Fast Healthcare Interoperability Resources (FHIR) standard. Despite the wealth of information in these records, their complexity and volume make it difficult for users to retrieve and interpret crucial health insights. Recent advances in Large Language Models (LLMs) offer a solution, enabling semantic question answering (QA) over medical data, allowing users to interact with their health records more effectively. However, ensuring privacy and compliance requires edge and private deployments of LLMs.
This paper proposes a novel approach to semantic QA over EHRs by first identifying the most relevant FHIR resources for a user query (Task1) and subsequently answering the query based on these resources (Task2). We explore the performance of privately hosted, fine-tuned LLMs, evaluating them against benchmark models such as GPT-4 and GPT-4o. Our results demonstrate that fine-tuned LLMs, while 250x smaller in size, outperform GPT-4 family models by 0.55% in F1 score on Task1 and 42% on Meteor Task in Task2. Additionally, we examine advanced aspects of LLM usage, including sequential fine-tuning, model self-evaluation (narcissistic evaluation), and the impact of training data size on performance. The models and datasets are available here: this https URL 

**Abstract (ZH)**: 医疗系统持续生成大量的电子健康记录（EHRs），这些记录通常存储在Fast Healthcare Interoperability Resources（FHIR）标准中。尽管这些记录蕴含丰富的信息，但由于其复杂性和大量信息，用户难以检索和解释关键的健康洞察。近年来，大规模语言模型（LLMs）的进步为这一问题提供了解决方案，允许通过半结构化医学数据进行语义问答（QA），从而更有效地与健康记录进行交互。然而，确保隐私和合规性要求在边缘和私有环境中部署LLMs。

本文提出了一种新的EHR语义问答方法，首先确定与用户查询最相关的FHIR资源（任务1），然后基于这些资源回答查询（任务2）。我们探讨了私有托管、微调后的LLMs的性能，并将它们与基准模型如GPT-4和GPT-4o进行评估。结果显示，相较于GPT-4系列模型，微调后的LLMs在任务1上的F1分数高出0.55%，在任务2上的Meteor分数高出42%。此外，我们还探讨了LLMs使用的一些高级方面，包括顺序微调、模型自我评估（自恋性评估）以及训练数据量对性能的影响。相关的模型和数据集可以在以下链接获取：[这个网址] 

---
# Unlearning Clients, Features and Samples in Vertical Federated Learning 

**Title (ZH)**: 垂直联邦学习中客户端、特征和样本的重新学习消除 

**Authors**: Ayush K. Varshney, Konstantinos Vandikas, Vicenç Torra  

**Link**: [PDF](https://arxiv.org/pdf/2501.13683)  

**Abstract**: Federated Learning (FL) has emerged as a prominent distributed learning paradigm. Within the scope of privacy preservation, information privacy regulations such as GDPR entitle users to request the removal (or unlearning) of their contribution from a service that is hosting the model. For this purpose, a server hosting an ML model must be able to unlearn certain information in cases such as copyright infringement or security issues that can make the model vulnerable or impact the performance of a service based on that model. While most unlearning approaches in FL focus on Horizontal FL (HFL), where clients share the feature space and the global model, Vertical FL (VFL) has received less attention from the research community. VFL involves clients (passive parties) sharing the sample space among them while not having access to the labels. In this paper, we explore unlearning in VFL from three perspectives: unlearning clients, unlearning features, and unlearning samples. To unlearn clients and features we introduce VFU-KD which is based on knowledge distillation (KD) while to unlearn samples, VFU-GA is introduced which is based on gradient ascent. To provide evidence of approximate unlearning, we utilize Membership Inference Attack (MIA) to audit the effectiveness of our unlearning approach. Our experiments across six tabular datasets and two image datasets demonstrate that VFU-KD and VFU-GA achieve performance comparable to or better than both retraining from scratch and the benchmark R2S method in many cases, with improvements of $(0-2\%)$. In the remaining cases, utility scores remain comparable, with a modest utility loss ranging from $1-5\%$. Unlike existing methods, VFU-KD and VFU-GA require no communication between active and passive parties during unlearning. However, they do require the active party to store the previously communicated embeddings. 

**Abstract (ZH)**: 联邦学习（FL）已发展成为一种重要的分布式学习范式。在隐私保护的范围内，类似于GDPR的信息隐私法规赋予用户要求从托管模型的服务中删除（或撤销）其贡献的权利。为此，托管机器学习模型的服务器必须在遇到版权侵权或可能使模型易受攻击或影响基于该模型的服务性能的安全问题时，能够撤销某些信息。尽管大多数FL的撤销方法侧重于水平联邦学习（HFL），其中客户端共享特征空间和全局模型，垂直联邦学习（VFL）在研究社区中受到的关注较少。VFL涉及客户端（被动方）共享样本空间，而不访问标签。在本文中，我们从三个视角探索VFL中的撤销：撤销客户端、撤销特征和撤销样本。为了撤销客户端和特征，我们提出了基于知识蒸馏（KD）的VFU-KD方法；为了撤销样本，我们提出了基于梯度上升的VFU-GA方法。为了提供撤销近似性的证据，我们使用成员推断攻击（MIA）来审计我们撤销方法的有效性。在针对六个结构化数据集和两个图像数据集进行的实验中，我们发现VFU-KD和VFU-GA在许多情况下与从零开始重新训练和基准的R2S方法相比，实现了相当或更好的性能，性能改进在0-2%之间。在其余情况下，效率得分相当，但存在轻微的效率损失，范围为1-5%。与现有方法不同，VFU-KD和VFU-GA在撤销过程期间不需要主动方和被动方之间进行通信，但主动方需要存储之前通信的嵌入。 

---
# Certified Robustness Under Bounded Levenshtein Distance 

**Title (ZH)**: 受限制的莱文斯坦距离下的认证鲁棒性 

**Authors**: Elias Abad Rocamora, Grigorios G. Chrysos, Volkan Cevher  

**Link**: [PDF](https://arxiv.org/pdf/2501.13676)  

**Abstract**: Text classifiers suffer from small perturbations, that if chosen adversarially, can dramatically change the output of the model. Verification methods can provide robustness certificates against such adversarial perturbations, by computing a sound lower bound on the robust accuracy. Nevertheless, existing verification methods incur in prohibitive costs and cannot practically handle Levenshtein distance constraints. We propose the first method for computing the Lipschitz constant of convolutional classifiers with respect to the Levenshtein distance. We use these Lipschitz constant estimates for training 1-Lipschitz classifiers. This enables computing the certified radius of a classifier in a single forward pass. Our method, LipsLev, is able to obtain $38.80$% and $13.93$% verified accuracy at distance $1$ and $2$ respectively in the AG-News dataset, while being $4$ orders of magnitude faster than existing approaches. We believe our work can open the door to more efficient verification in the text domain. 

**Abstract (ZH)**: 文本分类器对小型扰动十分敏感，而如果这些扰动被敌意选择，可以极大地改变模型的输出。验证方法可以通过计算稳健准确率的合理下界来提供对抗扰动的稳健性证书。然而，现有验证方法成本极高，无法实际处理莱文斯htein距离约束。我们提出了第一个计算卷积分类器在莱文斯htein距离下的Lipschitz常数的方法。我们使用这些Lipschitz常数估计值来训练1-Lipschitz分类器。这使得能够在一次前向传播中计算分类器的认证半径。我们的方法LipsLev能够在AG-News数据集中分别在距离1和距离2时获得38.80%和13.93%的认证准确率，且比现有方法快4个数量级。我们认为，我们的工作可以为文本领域更高效的验证打开大门。 

---
# How to Complete Domain Tuning while Keeping General Ability in LLM: Adaptive Layer-wise and Element-wise Regularization 

**Title (ZH)**: 如何在保持通用能力的同时完成领域调优：适应性分层和元素化正则化 

**Authors**: Shezheng Song, Hao Xu, Jun Ma, Shasha Li, Long Peng, Qian Wan, Xiaodong Liu, Jie Yu  

**Link**: [PDF](https://arxiv.org/pdf/2501.13669)  

**Abstract**: Large Language Models (LLMs) exhibit strong general-purpose language capabilities. However, fine-tuning these models on domain-specific tasks often leads to catastrophic forgetting, where the model overwrites or loses essential knowledge acquired during pretraining. This phenomenon significantly limits the broader applicability of LLMs. To address this challenge, we propose a novel approach to compute the element-wise importance of model parameters crucial for preserving general knowledge during fine-tuning. Our method utilizes a dual-objective optimization strategy: (1) regularization loss to retain the parameter crucial for general knowledge; (2) cross-entropy loss to adapt to domain-specific tasks. Additionally, we introduce layer-wise coefficients to account for the varying contributions of different layers, dynamically balancing the dual-objective optimization. Extensive experiments on scientific, medical, and physical tasks using GPT-J and LLaMA-3 demonstrate that our approach mitigates catastrophic forgetting while enhancing model adaptability. Compared to previous methods, our solution is approximately 20 times faster and requires only 10%-15% of the storage, highlighting the practical efficiency. The code will be released. 

**Abstract (ZH)**: 大型语言模型（LLMs）展示了广泛的语言能力。然而，在对特定领域任务进行微调时常会出现灾难性遗忘现象，即模型在微调过程中会覆盖或丢失预训练中获得的关键知识。这种现象显著限制了LLMs的广泛应用。为应对这一挑战，我们提出了一种新的方法，用于计算在微调过程中对保持通用知识至关重要的模型参数的元素级重要性。该方法采用了一种双目标优化策略：（1）正则化损失，用于保留对通用知识至关重要的参数；（2）交叉熵损失，用于适应特定领域的任务。此外，我们引入了分层系数，以反映不同层的不同贡献，动态平衡双目标优化。通过在GPT-J和LLaMA-3上进行的科学、医学和物理学任务的大量实验表明，我们的方法在减轻灾难性遗忘的同时提升了模型的适应性。与以往的方法相比，我们的解决方案大约快20倍，并且只需10%-15%的存储空间，突显了其实用效率。该代码即将发布。 

---
# Cognitive Paradigms for Evaluating VLMs on Visual Reasoning Task 

**Title (ZH)**: 视觉推理任务中评估大型语言模型的心理认知范式 

**Authors**: Mohit Vaishnav, Tanel Tammet  

**Link**: [PDF](https://arxiv.org/pdf/2501.13620)  

**Abstract**: Evaluating the reasoning capabilities of Vision-Language Models (VLMs) in complex visual tasks provides valuable insights into their potential and limitations. In this work, we assess the performance of VLMs on the challenging Bongard Openworld Problems benchmark, which involves reasoning over natural images. We propose and evaluate three human-inspired paradigms: holistic analysis (global context processing), deductive rule learning (explicit rule derivation and application), and componential analysis (structured decomposition of images into components). Our results demonstrate that state-of-the-art models, including GPT-4o and Gemini, not only surpass human benchmarks but also excel in structured reasoning tasks, with componential analysis proving especially effective. However, ablation studies reveal key challenges, such as handling synthetic images, making fine-grained distinctions, and interpreting nuanced contextual information. These insights underscore the need for further advancements in model robustness and generalization, while highlighting the transformative potential of structured reasoning approaches in enhancing VLM capabilities. 

**Abstract (ZH)**: 在复杂视觉任务中评估视觉语言模型（VLMs）的推理能力，为理解其潜在能力和限制提供了宝贵洞见。本研究评估了VLMs在具有挑战性的Bongard Openworld Problems基准上的性能，该基准涉及自然图像的推理。我们提出并评估了三种启发式人类方法：整体分析（全局语境处理）、演绎规则学习（显式规则的制定和应用）和成分分析（将图像结构化分解为组件）。结果显示，最新的模型，包括GPT-4o和Gemini，不仅超越了人类基准，还在结构化推理任务中表现出色，尤其是成分分析特别有效。然而，消融实验揭示了关键挑战，如处理合成图像、做出精细区分以及解释微妙的上下文信息。这些洞见强调了进一步提高模型鲁棒性和泛化能力的必要性，同时突显了结构化推理方法在增强VLM能力方面的变革潜力。 

---
# Efficient Synaptic Delay Implementation in Digital Event-Driven AI Accelerators 

**Title (ZH)**: 数字事件驱动AI加速器中的高效突触延迟实现 

**Authors**: Roy Meijer, Paul Detterer, Amirreza Yousefzadeh, Alberto Patino-Saucedo, Guanghzi Tang, Kanishkan Vadivel, Yinfu Xu, Manil-Dev Gomony, Federico Corradi, Bernabe Linares-Barranco, Manolis Sifalakis  

**Link**: [PDF](https://arxiv.org/pdf/2501.13610)  

**Abstract**: Synaptic delay parameterization of neural network models have remained largely unexplored but recent literature has been showing promising results, suggesting the delay parameterized models are simpler, smaller, sparser, and thus more energy efficient than similar performing (e.g. task accuracy) non-delay parameterized ones. We introduce Shared Circular Delay Queue (SCDQ), a novel hardware structure for supporting synaptic delays on digital neuromorphic accelerators. Our analysis and hardware results show that it scales better in terms of memory, than current commonly used approaches, and is more amortizable to algorithm-hardware co-optimizations, where in fact, memory scaling is modulated by model sparsity and not merely network size. Next to memory we also report performance on latency area and energy per inference. 

**Abstract (ZH)**: 神经网络模型中的突触延迟参数化长期以来鲜有研究，但近期文献显示了令人鼓舞的结果，表明带有延迟参数化的模型在性能相当（例如任务准确率）的情况下更为简单、更小、更稀疏，因此更具能效性。我们提出了共享循环延迟队列（SCDQ，Shared Circular Delay Queue），一种用于数字神经形态加速器支持突触延迟的新型硬件结构。我们的分析和硬件结果表明，相比于目前常用的方法，它在内存方面具有更好的扩展性，并且在算法-硬件协同优化中更具经济性，实际上，内存扩展性受到模型稀疏性的影响而非仅仅是网络规模的影响。除了内存外，我们还报告了在延迟面积和每推理次数能量方面的性能表现。 

---
# Optimal Multi-Objective Best Arm Identification with Fixed Confidence 

**Title (ZH)**: 固定置信度下的最优多目标最佳臂识别 

**Authors**: Zhirui Chen, P.N. Karthik, Yeow Meng Chee, Vincent Y. F. Tan  

**Link**: [PDF](https://arxiv.org/pdf/2501.13607)  

**Abstract**: We consider a multi-armed bandit setting with finitely many arms, in which each arm yields an $M$-dimensional vector reward upon selection. We assume that the reward of each dimension (a.k.a. {\em objective}) is generated independently of the others. The best arm of any given objective is the arm with the largest component of mean corresponding to the objective. The end goal is to identify the best arm of {\em every} objective in the shortest (expected) time subject to an upper bound on the probability of error (i.e., fixed-confidence regime). We establish a problem-dependent lower bound on the limiting growth rate of the expected stopping time, in the limit of vanishing error probabilities. This lower bound, we show, is characterised by a max-min optimisation problem that is computationally expensive to solve at each time step. We propose an algorithm that uses the novel idea of {\em surrogate proportions} to sample the arms at each time step, eliminating the need to solve the max-min optimisation problem at each step. We demonstrate theoretically that our algorithm is asymptotically optimal. In addition, we provide extensive empirical studies to substantiate the efficiency of our algorithm. While existing works on pure exploration with multi-objective multi-armed bandits predominantly focus on {\em Pareto frontier identification}, our work fills the gap in the literature by conducting a formal investigation of the multi-objective best arm identification problem. 

**Abstract (ZH)**: 我们考虑一个具有限多臂的multi-armed bandit设置，每个臂在被选择时会产出一个$M$维的向量奖励。我们假设每个维度（即所谓的对象）的奖励是独立生成的。任意给定对象的最佳臂是那个其该对象对应均值分量最大的臂。最终目标是在满足错误概率上限的情况下（即固定置信度设定），在最短的（期望）时间内识别出所有对象的最佳臂。我们确定了一个在错误概率趋近于零极限下的期望停止时间渐近增长率的下界。我们展示这一下界由一个复杂的最大化-最小化优化问题定义，每个时间步都需要解决该问题。我们提出了一种算法，该算法利用新颖的“代理比例”概念在每个时间步采样臂，从而避免了在每个时间步都求解该最大化-最小化优化问题。理论上，我们证明了该算法是渐近最优的。此外，我们还提供了详尽的实验研究以验证该算法的效率。虽然对于多对象multi-armed bandit中的纯探索已有工作的重心主要集中在帕累托前沿的识别上，我们的研究填补了文献中的空白，通过正式调查多对象最佳臂识别问题。 

---
# Text-to-SQL based on Large Language Models and Database Keyword Search 

**Title (ZH)**: 基于大型语言模型和数据库关键词搜索的文本到SQL转换 

**Authors**: Eduardo R. Nascimento, Caio Viktor S. Avila, Yenier T. Izquierdo, Grettel M. García, Lucas Feijó L. Andrade, Michelle S.P. Facina, Melissa Lemos, Marco A. Casanova  

**Link**: [PDF](https://arxiv.org/pdf/2501.13594)  

**Abstract**: Text-to-SQL prompt strategies based on Large Language Models (LLMs) achieve remarkable performance on well-known benchmarks. However, when applied to real-world databases, their performance is significantly less than for these benchmarks, especially for Natural Language (NL) questions requiring complex filters and joins to be processed. This paper then proposes a strategy to compile NL questions into SQL queries that incorporates a dynamic few-shot examples strategy and leverages the services provided by a database keyword search (KwS) platform. The paper details how the precision and recall of the schema-linking process are improved with the help of the examples provided and the keyword-matching service that the KwS platform offers. Then, it shows how the KwS platform can be used to synthesize a view that captures the joins required to process an input NL question and thereby simplify the SQL query compilation step. The paper includes experiments with a real-world relational database to assess the performance of the proposed strategy. The experiments suggest that the strategy achieves an accuracy on the real-world relational database that surpasses state-of-the-art approaches. The paper concludes by discussing the results obtained. 

**Abstract (ZH)**: 基于大型语言模型（LLMs）的文本到SQL提示策略在著名的基准测试中表现出色。然而，当应用于真实世界数据库时，它们的性能显著低于这些基准测试，尤其是在处理需要复杂过滤和连接的自然语言（NL）问题时。本文提出了一种策略，将NL问题编译为SQL查询，该策略结合了动态少样本示例策略，并利用数据库关键词搜索（KwS）平台提供的服务。文章详细说明了使用示例和KwS平台提供的关键词匹配服务如何提高模式链接过程的精准性和召回率。随后，展示了如何利用KwS平台生成一种视图，以捕获处理输入NL问题所需的连接，从而简化SQL查询编译步骤。本文通过使用真实世界的关系数据库进行实验，评估了所提出策略的性能。实验表明，该策略在真实世界的关建材数据库中达到了超越现有最佳方法的准确性。文章最后讨论了所获得的结果。 

---
# Contrastive Representation Learning Helps Cross-institutional Knowledge Transfer: A Study in Pediatric Ventilation Management 

**Title (ZH)**: 对比表示学习有助于机构间知识转移：儿科通气管理中的研究 

**Authors**: Yuxuan, Jinpei Han, Padmanabhan Ramnarayan, A. Aldo Faisal  

**Link**: [PDF](https://arxiv.org/pdf/2501.13587)  

**Abstract**: Clinical machine learning deployment across institutions faces significant challenges when patient populations and clinical practices differ substantially. We present a systematic framework for cross-institutional knowledge transfer in clinical time series, demonstrated through pediatric ventilation management between a general pediatric intensive care unit (PICU) and a cardiac-focused unit. Using contrastive predictive coding (CPC) for representation learning, we investigate how different data regimes and fine-tuning strategies affect knowledge transfer across institutional boundaries. Our results show that while direct model transfer performs poorly, CPC with appropriate fine-tuning enables effective knowledge sharing between institutions, with benefits particularly evident in limited data scenarios. Analysis of transfer patterns reveals an important asymmetry: temporal progression patterns transfer more readily than point-of-care decisions, suggesting practical pathways for cross-institutional deployment. Through a systematic evaluation of fine-tuning approaches and transfer patterns, our work provides insights for developing more generalizable clinical decision support systems while enabling smaller specialized units to leverage knowledge from larger centers. 

**Abstract (ZH)**: 当患者群体和临床实践存在显著差异时，将临床机器学习部署到不同机构面临重大挑战。我们提出了一种系统性的框架，用于跨机构在临床时间序列中的知识转移，并通过一般儿科重症监护病房（PICU）与心脏专科病房之间的儿科通气管理进行了演示。利用对比预测编码（CPC）进行表征学习，我们研究了不同的数据制度和微调策略如何影响机构边界之间的知识转移。结果显示，直接模型转移表现不佳，而使用适当的微调策略的CPC可以有效地在机构之间分享知识，特别是在数据有限的情况下更为明显。通过分析转移模式，我们揭示了一个重要的不对称性：时间进程模式比护理决策更容易跨机构转移，这为跨机构部署提供了实际途径。通过对微调方法和转移模式的系统评估，我们的研究提供了关于如何开发更具普适性的临床决策支持系统的见解，同时也使得较小的专业化单位能够利用大型中心的知识。 

---
# K-COMP: Retrieval-Augmented Medical Domain Question Answering With Knowledge-Injected Compressor 

**Title (ZH)**: K-COMP：知识注入压缩器增强的医学领域检索增援问答 

**Authors**: Jeonghun Cho, Gary Geunbae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2501.13567)  

**Abstract**: Retrieval-augmented question answering (QA) integrates external information, and thereby increases the QA accuracy of reader models that lack domain knowledge. However, documents retrieved for closed domains require high expertise, so the reader model may have difficulty fully comprehending the text. Moreover, the retrieved documents contain thousands of tokens, some unrelated to the question. As a result, the documents include some inaccurate information, which could lead the reader model to mistrust the passages and could result in hallucinations. To solve these problems, we propose K-COMP (Knowledge-injected compressor) which provides the knowledge required to answer correctly. The compressor automatically generates the requisite prior knowledge to facilitate the answering process prior to the compression of retrieved passages. Subsequently, the passages are compressed autoregressively, with the generated knowledge being integrated into the compression process. This process ensures alignment between the question intent and the compressed context. By augmenting this prior knowledge and concise context, the reader models are guided toward relevant answers and trust the context. 

**Abstract (ZH)**: 检索增强问答（QA）通过整合外部信息，提高了缺少领域知识的读者模型的问答准确性。然而，对于封闭领域的文档检索需要高度专业化的知识，因此读者模型可能难以完全理解文本。此外，检索到的文档包含数千个标记，其中一些与问题无关。因此，这些文档可能包含一些不准确的信息，这可能导致读者模型对段落产生怀疑，从而引发幻觉。为了解决这些问题，我们提出了K-COMP（知识注入压缩器），它提供了回答正确问题所需的必要的知识。压缩器自动生成必要的先验知识，以在压缩检索到的段落之前促进回答过程。随后，段落通过自回归方式压缩，生成的知识被整合到压缩过程中。这一过程确保了问题意图与压缩上下文之间的对齐。通过增强这种先验知识和简洁的上下文，读者模型能够更容易地找到相关答案，并信任该上下文。 

---
# Black-Box Adversarial Attack on Vision Language Models for Autonomous Driving 

**Title (ZH)**: 面向自主驾驶的视觉语言模型的黑盒 adversarial 攻击 

**Authors**: Lu Wang, Tianyuan Zhang, Yang Qu, Siyuan Liang, Yuwei Chen, Aishan Liu, Xianglong Liu, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2501.13563)  

**Abstract**: Vision-language models (VLMs) have significantly advanced autonomous driving (AD) by enhancing reasoning capabilities; however, these models remain highly susceptible to adversarial attacks. While existing research has explored white-box attacks to some extent, the more practical and challenging black-box scenarios remain largely underexplored due to their inherent difficulty. In this paper, we take the first step toward designing black-box adversarial attacks specifically targeting VLMs in AD. We identify two key challenges for achieving effective black-box attacks in this context: the effectiveness across driving reasoning chains in AD systems and the dynamic nature of driving scenarios. To address this, we propose Cascading Adversarial Disruption (CAD). It first introduces Decision Chain Disruption, which targets low-level reasoning breakdown by generating and injecting deceptive semantics, ensuring the perturbations remain effective across the entire decision-making chain. Building on this, we present Risky Scene Induction, which addresses dynamic adaptation by leveraging a surrogate VLM to understand and construct high-level risky scenarios that are likely to result in critical errors in the current driving contexts. Extensive experiments conducted on multiple AD VLMs and benchmarks demonstrate that CAD achieves state-of-the-art attack effectiveness, significantly outperforming existing methods (+13.43% on average). Moreover, we validate its practical applicability through real-world attacks on AD vehicles powered by VLMs, where the route completion rate drops by 61.11% and the vehicle crashes directly into the obstacle vehicle with adversarial patches. Finally, we release CADA dataset, comprising 18,808 adversarial visual-question-answer pairs, to facilitate further evaluation and research in this critical domain. Our codes and dataset will be available after paper's acceptance. 

**Abstract (ZH)**: 视觉-语言模型（VLMs）在提高自主驾驶（AD）的推理能力方面取得了显著进展；然而，这些模型仍然高度易受对抗攻击的影响。虽然现有研究在一定程度上探索了白盒攻击，但由于其固有的难度，更实际且更具挑战性的黑盒场景仍很大程度上未被探索。在本文中，我们首次尝试针对AD中的VLMs设计黑盒对抗攻击。我们确定了两个主要挑战：在AD系统中实现有效的黑盒攻击中的推理链有效性以及不断变化的驾驶场景动态性。为了解决这些问题，我们提出了级联对抗干扰（CAD）方法。首先，CAD引入了决策链干扰（Decision Chain Disruption），通过生成和注入欺骗性语义来针对低级推理中断，确保扰动在整个决策链中保持有效性。在此基础上，我们提出了风险场景诱导（Risky Scene Induction），通过利用代理VLM来理解和构建可能在当前驾驶场景中导致关键错误的高层次风险场景。我们在多个AD VLM和基准上进行的 extensive 实验表明，CAD 达到了最先进的攻击效果，平均比现有方法高出 13.43%。此外，我们通过针对由VLM驱动的AD车辆的实际攻击验证了其实际适用性，其中路线完成率降低了 61.11%，并且车辆直接撞向了带有对抗补丁的障碍物车辆。最后，我们发布了包含 18,808 个对抗视觉-问题-答案对的 CADA 数据集，以促进对该关键领域的进一步评估和研究。我们的代码和数据集将在论文接受后公开。 

---
# One-Prompt-One-Story: Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt 

**Title (ZH)**: 一令一述：使用单一提示实现免费午餐一致性文本到图像生成 

**Authors**: Tao Liu, Kai Wang, Senmao Li, Joost van de Weijer, Fahad Shahbaz Khan, Shiqi Yang, Yaxing Wang, Jian Yang, Ming-Ming Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2501.13554)  

**Abstract**: Text-to-image generation models can create high-quality images from input prompts. However, they struggle to support the consistent generation of identity-preserving requirements for storytelling. Existing approaches to this problem typically require extensive training in large datasets or additional modifications to the original model architectures. This limits their applicability across different domains and diverse diffusion model configurations. In this paper, we first observe the inherent capability of language models, coined context consistency, to comprehend identity through context with a single prompt. Drawing inspiration from the inherent context consistency, we propose a novel training-free method for consistent text-to-image (T2I) generation, termed "One-Prompt-One-Story" (1Prompt1Story). Our approach 1Prompt1Story concatenates all prompts into a single input for T2I diffusion models, initially preserving character identities. We then refine the generation process using two novel techniques: Singular-Value Reweighting and Identity-Preserving Cross-Attention, ensuring better alignment with the input description for each frame. In our experiments, we compare our method against various existing consistent T2I generation approaches to demonstrate its effectiveness through quantitative metrics and qualitative assessments. Code is available at this https URL. 

**Abstract (ZH)**: 文本到图像生成模型可以从输入提示中生成高质量的图像，但在支持故事叙述中保持身份一致性的持续生成方面存在困难。目前解决这一问题的方法通常需要在大规模数据集上进行大量的训练，或者对原始模型架构进行额外的修改。这限制了它们在不同领域和多样化的扩散模型配置中的应用。在本文中，我们首先观察了语言模型固有的通过单一提示理解上下文身份的能力，称为“上下文一致性”。受到这一固有上下文一致性的启发，我们提出了一种无需训练的新方法，用于一致的文本到图像（T2I）生成，该方法称为“一提示一故事”（1Prompt1Story）。我们的方法1Prompt1Story将所有提示连接成单一输入提供给T2I扩散模型，最初保持人物身份。随后，我们使用两种新颖的技术：奇异值重加权和身份保持交叉注意，来优化生成过程，确保每个帧与输入描述有更好的对齐效果。在我们的实验中，我们通过量化指标和定性评估将我们的方法与各种现有的一致的T2I生成方法进行了比较，以证明其有效性。代码可在以下链接获取：[请替换为实际链接]。 

---
# Explainable AI-aided Feature Selection and Model Reduction for DRL-based V2X Resource Allocation 

**Title (ZH)**: 可解释的人工智能辅助特征选择与模型简化在基于DRL的V2X资源分配中的应用 

**Authors**: Nasir Khan, Asmaa Abdallah, Abdulkadir Celik, Ahmed M. Eltawil, Sinem Coleri  

**Link**: [PDF](https://arxiv.org/pdf/2501.13552)  

**Abstract**: Artificial intelligence (AI) is expected to significantly enhance radio resource management (RRM) in sixth-generation (6G) networks. However, the lack of explainability in complex deep learning (DL) models poses a challenge for practical implementation. This paper proposes a novel explainable AI (XAI)- based framework for feature selection and model complexity reduction in a model-agnostic manner. Applied to a multi-agent deep reinforcement learning (MADRL) setting, our approach addresses the joint sub-band assignment and power allocation problem in cellular vehicle-to-everything (V2X) communications. We propose a novel two-stage systematic explainability framework leveraging feature relevance-oriented XAI to simplify the DRL agents. While the former stage generates a state feature importance ranking of the trained models using Shapley additive explanations (SHAP)-based importance scores, the latter stage exploits these importance-based rankings to simplify the state space of the agents by removing the least important features from the model input. Simulation results demonstrate that the XAI-assisted methodology achieves 97% of the original MADRL sum-rate performance while reducing optimal state features by 28%, average training time by 11%, and trainable weight parameters by 46% in a network with eight vehicular pairs. 

**Abstract (ZH)**: 人工智能（AI）预计将在第六代（6G）网络的无线资源管理（RRM）中发挥重要作用。然而，复杂深度学习（DL）模型缺乏解释性给其实用实施带来了挑战。本文提出了一种新的面向特征选择和模型复杂性降低的解释性AI（XAI）框架，该框架为模型不可知的应用场景提供了解释性。在多智能体深度强化学习（MADRL）环境中，我们的方法解决了蜂窝V2X通信中的子带分配和功率分配问题。我们提出了一种新颖的两阶段系统解释框架，利用特征相关性导向的XAI简化DRL智能体。前一阶段利用Shapley加解释（SHAP）的重要性评分生成训练模型的状态特征重要性排名，而后一阶段利用这些基于重要性的排名通过移除模型输入中最不重要的特征来简化智能体的状态空间。仿真结果表明，在八对车辆的网络中，XAI辅助的方法在减少最优状态特征28%、平均训练时间11%和可训练权重参数46%的情况下，仍能够实现原始MADRL总速率性能的97%。 

---
# LLMs Can Plan Only If We Tell Them 

**Title (ZH)**: LLMs 只能在我们明确指示的情况下才能进行规划 

**Authors**: Bilgehan Sel, Ruoxi Jia, Ming Jin  

**Link**: [PDF](https://arxiv.org/pdf/2501.13545)  

**Abstract**: Large language models (LLMs) have demonstrated significant capabilities in natural language processing and reasoning, yet their effectiveness in autonomous planning has been under debate. While existing studies have utilized LLMs with external feedback mechanisms or in controlled environments for planning, these approaches often involve substantial computational and development resources due to the requirement for careful design and iterative backprompting. Moreover, even the most advanced LLMs like GPT-4 struggle to match human performance on standard planning benchmarks, such as the Blocksworld, without additional support. This paper investigates whether LLMs can independently generate long-horizon plans that rival human baselines. Our novel enhancements to Algorithm-of-Thoughts (AoT), which we dub AoT+, help achieve state-of-the-art results in planning benchmarks out-competing prior methods and human baselines all autonomously. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自然语言处理和推理方面展现出了显著的能力，但在自主规划方面的有效性一直存在争议。尽管现有研究已经利用具有外部反馈机制或在受控环境中应用LLMs进行规划，这些方法往往需要大量的计算和开发资源，因为它们要求精心设计和反复的反向提示。此外，即使是最先进的LLM，如GPT-4，在标准规划基准测试（如Blocksworld）上也难以达到与人类相当的性能，除非得到额外的支持。本文探讨了LLMs是否能够独立生成与人类基线相匹敌的长期规划。我们对Algorithm-of-Thoughts（AoT）算法进行了一些新的改进，称为AoT+，这些改进使得在规划基准测试中取得了领先于先前方法和人类基线的最新成果，并实现了全部自主规划。 

---
# GCAD: Anomaly Detection in Multivariate Time Series from the Perspective of Granger Causality 

**Title (ZH)**: GCAD：基于Granger因果性的多变量时间序列异常检测 

**Authors**: Zehao Liu, Mengzhou Gao, Pengfei Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2501.13493)  

**Abstract**: Multivariate time series anomaly detection has numerous real-world applications and is being extensively studied. Modeling pairwise correlations between variables is crucial. Existing methods employ learnable graph structures and graph neural networks to explicitly model the spatial dependencies between variables. However, these methods are primarily based on prediction or reconstruction tasks, which can only learn similarity relationships between sequence embeddings and lack interpretability in how graph structures affect time series evolution. In this paper, we designed a framework that models spatial dependencies using interpretable causal relationships and detects anomalies through changes in causal patterns. Specifically, we propose a method to dynamically discover Granger causality using gradients in nonlinear deep predictors and employ a simple sparsification strategy to obtain a Granger causality graph, detecting anomalies from a causal perspective. Experiments on real-world datasets demonstrate that the proposed model achieves more accurate anomaly detection compared to baseline methods. 

**Abstract (ZH)**: 多变量时间序列异常检测在众多实际应用中具有重要价值，并且受到广泛的学术研究关注。变量间的成对相关性建模至关重要。现有方法利用可学习的图结构和图神经网络来显式建模变量间的空间依赖性。然而，这些方法主要基于预测或重构任务，只能学习序列嵌入之间的相似性关系，而在如何通过图结构影响时间序列演变方面缺乏可解释性。本文设计了一个框架，利用可解释的因果关系建模空间依赖性，并通过因果模式的变化来检测异常。具体而言，我们提出了一种动态发现Granger因果关系的方法，使用非线性深层预测器中的梯度，并采用简单的稀疏化策略获得Granger因果关系图，从因果角度检测异常。实验结果表明，所提出的模型在真实数据集上的异常检测准确性优于基线方法。 

---
# RECALL: Library-Like Behavior In Language Models is Enhanced by Self-Referencing Causal Cycles 

**Title (ZH)**: RECALL：通过自我参照因果循环增强的语言模型库-like 行为 

**Authors**: Munachiso Nwadike, Zangir Iklassov, Toluwani Aremu, Tatsuya Hiraoka, Velibor Bojkovic, Benjamin Heinzerling, Hilal Alqaubeh, Martin Takáč, Kentaro Inui  

**Link**: [PDF](https://arxiv.org/pdf/2501.13491)  

**Abstract**: We introduce the concept of the self-referencing causal cycle (abbreviated RECALL) - a mechanism that enables large language models (LLMs) to bypass the limitations of unidirectional causality, which underlies a phenomenon known as the reversal curse. When an LLM is prompted with sequential data, it often fails to recall preceding context. For example, when we ask an LLM to recall the line preceding "O say does that star-spangled banner yet wave" in the U.S. National Anthem, it often fails to correctly return "Gave proof through the night that our flag was still there" - this is due to the reversal curse. It occurs because language models such as ChatGPT and Llama generate text based on preceding tokens, requiring facts to be learned and reproduced in a consistent token order. While the reversal curse is often viewed as a limitation, we offer evidence of an alternative view: it is not always an obstacle in practice. We find that RECALL is driven by what we designate as cycle tokens - sequences that connect different parts of the training data, enabling recall of preceding tokens from succeeding ones. Through rigorous probabilistic formalization and controlled experiments, we demonstrate how the cycles they induce influence a model's ability to reproduce information. To facilitate reproducibility, we provide our code and experimental details at this https URL. 

**Abstract (ZH)**: 我们引入了自我参照因果循环（简称RECALL）的概念——这是一种机制，使得大规模语言模型（LLMs）能够克服单一方向因果性的限制，这些限制导致了一种被称为反转诅咒的现象。当LLM被馈送序贯数据时，它常常无法回溯先前的上下文。例如，在要求LLM回溯爱国歌曲《星条旗》中的“O say does that star-spangled banner yet wave”这句前面的那一行文字时（正确答案是“Gave proof through the night that our flag was still there”），它经常无法正确地返回这个信息 —— 这就是反转诅咒的原因。这种现象是因为像ChatGPT和Llama这样的语言模型是基于先前的词元生成文本的，要求事实以一致的词元顺序被学习和再现。虽然反转诅咒通常被视为一种限制，但我们提供了证据，表明这是一种替代的看法：在实践中，它并不总是障碍。我们发现，RECALL是由于我们称之为循环词元的序列驱动的 —— 这些序列连接训练数据的不同部分，从而使模型能够从后续词元回溯到先前的词元并进行回溯。通过严谨的概率形式化和受控实验，我们展示了循环如何影响模型再现信息的能力。为了促进可重复性，我们在这里提供了我们的代码和实验细节：https://example.com（请注意，这里的URL是示例，您需要替换为实际的URL）。 

---
# MambaQuant: Quantizing the Mamba Family with Variance Aligned Rotation Methods 

**Title (ZH)**: MambaQuant：基于方差对齐旋转方法的Mamba家族量化方法 

**Authors**: Zukang Xu, Yuxuan Yue, Xing Hu, Zhihang Yuan, Zixu Jiang, Zhixuan Chen, Jiangyong Yu, Chen Xu, Sifan Zhou, Dawei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2501.13484)  

**Abstract**: Mamba is an efficient sequence model that rivals Transformers and demonstrates significant potential as a foundational architecture for various tasks. Quantization is commonly used in neural networks to reduce model size and computational latency. However, applying quantization to Mamba remains underexplored, and existing quantization methods, which have been effective for CNN and Transformer models, appear inadequate for Mamba models (e.g., Quarot suffers a 21% accuracy drop on Vim-T$^\dagger$ even under W8A8). We have pioneered the exploration of this issue and identified several key challenges. First, significant outliers are present in gate projections, output projections, and matrix multiplications. Second, Mamba's unique parallel scan further amplifies these outliers, leading to uneven and heavy-tailed data distributions. Third, even with the application of the Hadamard transform, the variance across channels in weights and activations still remains inconsistent. To these ends, we propose MambaQuant, a post-training quantization (PTQ) framework consisting of: 1) Karhunen-Loeve Transformation (KLT) enhanced rotation, rendering the rotation matrix adaptable to diverse channel distributions. 2) Smooth-Fused rotation, which equalizes channel variances and can merge additional parameters into model weights. Experiments show that MambaQuant can quantize both weights and activations into 8-bit with less than 1% accuracy loss for Mamba-based vision and language tasks. To the best of our knowledge, MambaQuant is the first comprehensive PTQ design for the Mamba family, paving the way for further advancements in its application. 

**Abstract (ZH)**: 马amba是一种高效序列模型，其性能与Transformer相当，并展现了在各种任务中作为基础架构的巨大潜力。量化常用于减少神经网络模型大小和计算延迟。然而，将量化应用于Mamba仍然处于探索阶段，目前针对CNN和Transformer有效的量化方法似乎不适合Mamba模型（例如，Quarot在Vim-T\(^ \dagger \)上的精度下降了21%，即使在W8A8条件下也是如此）。我们开创性地探讨了这一问题，识别出几个关键挑战。首先，门控投影、输出投影和矩阵乘法中存在显著的异常值。其次，Mamba独有的并行扫描进一步放大了这些异常值，导致数据分布不均匀且偏重尾分布。第三，即使应用了哈达玛变换，权重和激活的通道间方差仍然不一致。为解决这些问题，我们提出了一种后训练量化(PTQ)框架MambaQuant，其组成部分包括：1) 增强旋转矩阵的Karhunen-Loeve变换(KLT)旋转，使旋转矩阵能够适应不同的通道分布；2) 平滑融合旋转，平衡通道方差并可将额外参数合并入模型权重中。实验结果表明，MambaQuant可以将Mamba相关的视觉和语言任务中的权重和激活量化到8位，且精度损失小于1%。据我们所知，MambaQuant是针对Mamba家族的第一个全面的PTQ设计，为该模型的进一步应用铺平了道路。 

---
# A Polynomial-Time Algorithm for EFX Orientations of Chores 

**Title (ZH)**: 一种多项式时间算法，用于 Chore 的 EFX 分配 

**Authors**: Kevin Hsu, Valerie King  

**Link**: [PDF](https://arxiv.org/pdf/2501.13481)  

**Abstract**: This paper addresses the problem of finding EFX orientations of graphs of chores, in which each vertex corresponds to an agent, each edge corresponds to a chore, and a chore has zero marginal utility to an agent if its corresponding edge is not incident to the vertex corresponding to the agent. Recently, Zhou~et~al.~(IJCAI,~2024) analyzed the complexity of deciding whether graphs containing a mixture of goods and chores admit EFX orientations, and conjectured that deciding whether graphs containing only chores admit EFX orientations is NP-complete. In this paper, we resolve this conjecture by exhibiting a polynomial-time algorithm that finds an EFX orientation of a graph containing only chores if one exists, even if the graph contains self-loops. Remarkably, our first result demonstrates a surprising separation between the case of goods and the case of chores, because deciding whether graphs containing only goods admit EFX orientations of goods was shown to be NP-complete by Christodoulou et al.~(EC,~2023). In addition, we show the analogous decision problem for multigraphs to be NP-complete. 

**Abstract (ZH)**: 本文探讨了在工作中寻找EFX定向的问题，其中每个顶点对应一个代理，每条边对应一项工作，如果与该代理对应的顶点不邻接，则这项工作的边际效用为零。近期，周等（IJCAI，2024）分析了含有物品和工作的图中是否允许存在EFX定向的复杂性，并猜测仅含工作的图是否存在EFX定向是NP完全的。在本文中，我们通过展示一个多项式时间算法来解决这一猜测，该算法可以在存在仅含工作的图时找出其EFX定向，即使图中包含自环。尤为值得注意的是，我们的第一个结果表明了仅含物品和仅含工作的两种情况之间存在令人惊讶的差异，因为Christodoulou等（EC，2023）已经证明了仅含物品的图是否存在物品的EFX定向是NP完全的。另外，我们还证明了含多重边的图的相应决策问题也是NP完全的。 

---
# Adaptive Testing for LLM-Based Applications: A Diversity-based Approach 

**Title (ZH)**: 基于多样性的自适应测试方法：面向LLM的應用 

**Authors**: Juyeon Yoon, Robert Feldt, Shin Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2501.13480)  

**Abstract**: The recent surge of building software systems powered by Large Language Models (LLMs) has led to the development of various testing frameworks, primarily focused on treating prompt templates as the unit of testing. Despite the significant costs associated with test input execution and output assessment, the curation of optimized test suites is yet overlooked in these tools, which calls for tailored test selection or prioritization strategies. In this paper, we show that diversity-based testing techniques, such as Adaptive Random Testing (ART) with appropriate string distance metrics, can be effectively applied to the testing of prompt templates. Our proposed adaptive testing approach adjusts the conventional ART process to this context by selecting new test inputs based on scores derived from existing test suite and their labelling results. Our results, obtained using various implementations that explore several string-based distances, confirm that our approach enables the discovery of failures with reduced testing budgets and promotes the generation of more varied outputs. 

**Abstract (ZH)**: 近年来，受大规模语言模型（LLMs）驱动的软件系统建设热潮，催生了各种测试框架，主要侧重于将提示模板视为测试单元。尽管测试输入执行和输出评估的成本较高，但这些工具在优化测试套件的维护方面仍被忽视，这需要定制化的测试选择或优先级策略。在本文中，我们展示了一种基于多样性的测试技术，如适当使用的字符串距离度量的自适应随机测试（ART），可以有效地应用于提示模板的测试。我们提出的一种自适应测试方法，通过利用现有测试套件及其标签结果的评分来选择新的测试输入，调整了传统的ART流程。通过多种实施的研究探索了几种字符串距离方法，我们的结果证实了这种方法能够在降低测试预算的情况下发现更多失败，并促进生成更多样化的输出。 

---
# Adaptive Few-Shot Learning (AFSL): Tackling Data Scarcity with Stability, Robustness, and Versatility 

**Title (ZH)**: 自适应少样本学习（AFSL）：通过稳定性和鲁棒性应对数据稀缺性并实现多功能性 

**Authors**: Rishabh Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2501.13479)  

**Abstract**: Few-shot learning (FSL) enables machine learning models to generalize effectively with minimal labeled data, making it crucial for data-scarce domains such as healthcare, robotics, and natural language processing. Despite its potential, FSL faces challenges including sensitivity to initialization, difficulty in adapting to diverse domains, and vulnerability to noisy datasets. To address these issues, this paper introduces Adaptive Few-Shot Learning (AFSL), a framework that integrates advancements in meta-learning, domain alignment, noise resilience, and multi-modal integration. AFSL consists of four key modules: a Dynamic Stability Module for performance consistency, a Contextual Domain Alignment Module for domain adaptation, a Noise-Adaptive Resilience Module for handling noisy data, and a Multi-Modal Fusion Module for integrating diverse modalities. This work also explores strategies such as task-aware data augmentation, semi-supervised learning, and explainable AI techniques to enhance the applicability and robustness of FSL. AFSL provides scalable, reliable, and impactful solutions for real-world, high-stakes domains. 

**Abstract (ZH)**: 少样本学习（Few-shot learning, FSL）能够用最少的标注数据有效地推广机器学习模型，这对于数据稀缺的应用领域（如医疗健康、机器人技术和自然语言处理）至关重要。尽管FSL具有巨大潜力，但该技术仍面临诸多挑战，包括对初始化的敏感性、难以适应多种领域以及对噪声数据的易受影响性。为解决这些问题，本文提出了自适应少样本学习（Adaptive Few-shot Learning, AFSL）框架，该框架集成了元学习、领域对齐、噪声鲁棒性和多模态集成的最新进展。AFSL框架包括四个关键模块：动态稳定性模块（Dynamic Stability Module），用于确保性能一致性；上下文领域对齐模块（Contextual Domain Alignment Module），用于领域适应；噪声自适应鲁棒性模块（Noise-Adaptive Resilience Module），用于处理噪声数据；多模态融合模块（Multi-Modal Fusion Module），用于集成多种模态。本文还探讨了任务感知的数据增强、半监督学习和可解释人工智能技术等策略，以增强FSL的适用性和鲁棒性。AFSL提供了一种可扩展、可靠且具有影响力的解决方案，适用于实际、高风险领域。 

---
# Streaming Video Understanding and Multi-round Interaction with Memory-enhanced Knowledge 

**Title (ZH)**: Streaming视频理解与记忆增强知识的多轮交互 

**Authors**: Haomiao Xiong, Zongxin Yang, Jiazuo Yu, Yunzhi Zhuge, Lu Zhang, Jiawen Zhu, Huchuan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2501.13468)  

**Abstract**: Recent advances in Large Language Models (LLMs) have enabled the development of Video-LLMs, advancing multimodal learning by bridging video data with language tasks. However, current video understanding models struggle with processing long video sequences, supporting multi-turn dialogues, and adapting to real-world dynamic scenarios. To address these issues, we propose StreamChat, a training-free framework for streaming video reasoning and conversational interaction. $\StreamChat$ leverages a novel hierarchical memory system to efficiently process and compress video features over extended sequences, enabling real-time, multi-turn dialogue. Our framework incorporates a parallel system scheduling strategy that enhances processing speed and reduces latency, ensuring robust performance in real-world applications. Furthermore, we introduce StreamBench, a versatile benchmark that evaluates streaming video understanding across diverse media types and interactive scenarios, including multi-turn interactions and complex reasoning tasks. Extensive evaluations on StreamBench and other public benchmarks demonstrate that StreamChat significantly outperforms existing state-of-the-art models in terms of accuracy and response times, confirming its effectiveness for streaming video understanding. Code is available at StreamChat: this https URL. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）的发展推动了视频大型语言模型（Video-LLMs）的出现，通过将视频数据与语言任务相结合，促进了多模态学习的发展。然而，当前的视频理解模型在处理长视频序列、支持多轮对话以及适应真实世界的动态场景方面存在困难。为解决这些问题，我们提出了一种无需训练的框架——StreamChat，用于流式视频推理和对话交互。$\StreamChat$ 利用了一种新颖的分层记忆系统，可以在长时间序列中有效处理和压缩视频特征，从而实现实时多轮对话。我们的框架采用并行系统调度策略，提高了处理速度并降低了延迟，确保了在实际应用中的稳健性能。此外，我们引入了StreamBench，这是一种多功能基准，用于评估不同媒体类型和交互场景下的流式视频理解，包括多轮交互和复杂推理任务。在StreamBench和其他公共基准上的广泛评估表明，StreamChat在准确性和响应时间方面显著优于现有的最先进模型，验证了其在流式视频理解方面的有效性。代码可在以下链接获取：StreamChat: [这里提供链接] 

---
# Zero-Shot Trajectory Planning for Signal Temporal Logic Tasks 

**Title (ZH)**: 零样本轨迹规划用于信号时序逻辑任务 

**Authors**: Ruijia Liu, Ancheng Hou, Xiao Yu, Xiang Yin  

**Link**: [PDF](https://arxiv.org/pdf/2501.13457)  

**Abstract**: Signal Temporal Logic (STL) is a powerful specification language for describing complex temporal behaviors of continuous signals, making it well-suited for high-level robotic task descriptions. However, generating executable plans for STL tasks is challenging, as it requires consideration of the coupling between the task specification and the system dynamics. Existing approaches either follow a model-based setting that explicitly requires knowledge of the system dynamics or adopt a task-oriented data-driven approach to learn plans for specific tasks. In this work, we investigate the problem of generating executable STL plans for systems whose dynamics are unknown a priori. We propose a new planning framework that uses only task-agnostic data during the offline training stage, enabling zero-shot generalization to new STL tasks. Our framework is hierarchical, involving: (i) decomposing the STL task into a set of progress and time constraints, (ii) searching for time-aware waypoints guided by task-agnostic data, and (iii) generating trajectories using a pre-trained safe diffusion model. Simulation results demonstrate the effectiveness of our method indeed in achieving zero-shot generalization to various STL tasks. 

**Abstract (ZH)**: 基于信号时序逻辑（STL）是一种强大的规范语言，适用于描述连续信号的复杂时间行为，因此非常适合高层机器人任务的描述。然而，生成执行STL任务的计划具有挑战性，因为这需要考虑任务规范与系统动力学之间的耦合。现有方法要么采用基于模型的设置，明确要求了解系统动力学，要么采用任务导向的数据驱动方法来学习特定任务的计划。在本文中，我们研究了在系统动力学事先未知的情况下生成可执行STL计划的问题。我们提出了一种新的规划框架，在离线训练阶段仅使用任务无关的数据，从而实现对新STL任务的零样本泛化。该框架是分层的，包括以下步骤：（i）将STL任务分解为一组进度和时间约束；（ii）根据任务无关的数据搜索时间感知的航点；（iii）使用预训练的安全扩散模型生成轨迹。仿真实验结果表明，我们的方法确实能够实现对各种STL任务的零样本泛化。 

---
# KAA: Kolmogorov-Arnold Attention for Enhancing Attentive Graph Neural Networks 

**Title (ZH)**: KAA：柯尔莫哥洛夫-阿诺尔德注意力机制在增强注意型图神经网络中的应用 

**Authors**: Taoran Fang, Tianhong Gao, Chunping Wang, Yihao Shang, Wei Chow, Lei Chen, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2501.13456)  

**Abstract**: Graph neural networks (GNNs) with attention mechanisms, often referred to as attentive GNNs, have emerged as a prominent paradigm in advanced GNN models in recent years. However, our understanding of the critical process of scoring neighbor nodes remains limited, leading to the underperformance of many existing attentive GNNs. In this paper, we unify the scoring functions of current attentive GNNs and propose Kolmogorov-Arnold Attention (KAA), which integrates the Kolmogorov-Arnold Network (KAN) architecture into the scoring process. KAA enhances the performance of scoring functions across the board and can be applied to nearly all existing attentive GNNs. To compare the expressive power of KAA with other scoring functions, we introduce Maximum Ranking Distance (MRD) to quantitatively estimate their upper bounds in ranking errors for node importance. Our analysis reveals that, under limited parameters and constraints on width and depth, both linear transformation-based and MLP-based scoring functions exhibit finite expressive power. In contrast, our proposed KAA, even with a single-layer KAN parameterized by zero-order B-spline functions, demonstrates nearly infinite expressive power. Extensive experiments on both node-level and graph-level tasks using various backbone models show that KAA-enhanced scoring functions consistently outperform their original counterparts, achieving performance improvements of over 20% in some cases. 

**Abstract (ZH)**: 以下是翻译成中文后的版本，符合学术规范：

图形神经网络（GNNs）结合注意机制的变体，常被称为注意GNNs，近年来已成为高级GNN模型中的重要范式。然而，我们对评分邻居节点的关键过程的理解仍有限，导致许多现有的注意GNNs表现不佳。本文中，我们统一了当前注意GNNs的评分函数，并提出了柯尔莫哥洛夫-阿诺尔德注意机制（KAA），将柯尔莫哥洛夫-阿诺尔德网络（KAN）架构融入评分过程。KAA在整个评分函数中提升了性能，并可应用于几乎所有现有的注意GNNs。为了将KAA与其他评分函数的表达能力进行对比，我们引入了最大排名距离（MRD），以定量化地估计它们在节点重要性排名中的误差上限。我们的分析表明，即使在参数和宽度、深度等约束有限的情况下，基于线性变换和基于MLP的评分函数也表现出有限的表达能力。相比之下，我们提出的KAA，即使仅使用参数化为零阶B样条函数的一层KAN，也展示了近乎无限的表达能力。通过使用多种骨干模型在节点级别和图形级别任务上的广泛实验表明，KAA增强的评分函数始终优于原先版本，某些情况下性能提升了超过20%。 

---
# BMG-Q: Localized Bipartite Match Graph Attention Q-Learning for Ride-Pooling Order Dispatch 

**Title (ZH)**: BMG-Q：局部双部分匹配图注意力Q学习算法在拼车订单调度中的应用 

**Authors**: Yulong Hu, Siyuan Feng, Sen Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.13448)  

**Abstract**: This paper introduces Localized Bipartite Match Graph Attention Q-Learning (BMG-Q), a novel Multi-Agent Reinforcement Learning (MARL) algorithm framework tailored for ride-pooling order dispatch. BMG-Q advances ride-pooling decision-making process with the localized bipartite match graph underlying the Markov Decision Process, enabling the development of novel Graph Attention Double Deep Q Network (GATDDQN) as the MARL backbone to capture the dynamic interactions among ride-pooling vehicles in fleet. Our approach enriches the state information for each agent with GATDDQN by leveraging a localized bipartite interdependence graph and enables a centralized global coordinator to optimize order matching and agent behavior using Integer Linear Programming (ILP). Enhanced by gradient clipping and localized graph sampling, our GATDDQN improves scalability and robustness. Furthermore, the inclusion of a posterior score function in the ILP captures the online exploration-exploitation trade-off and reduces the potential overestimation bias of agents, thereby elevating the quality of the derived solutions. Through extensive experiments and validation, BMG-Q has demonstrated superior performance in both training and operations for thousands of vehicle agents, outperforming benchmark reinforcement learning frameworks by around 10% in accumulative rewards and showing a significant reduction in overestimation bias by over 50%. Additionally, it maintains robustness amidst task variations and fleet size changes, establishing BMG-Q as an effective, scalable, and robust framework for advancing ride-pooling order dispatch operations. 

**Abstract (ZH)**: 本文介绍了针对打车拼车订单分派的局部二分匹配图注意Q学习算法（BMG-Q），这是一种新颖的多智能体强化学习（MARL）算法框架。BMG-Q 通过Markov决策过程（MDP）背后的局部二分匹配图推进了打车拼车决策过程，使我们能够开发出基于图注意的双深度QLearning网络（GATDDQN）作为MARL的核心，以捕捉车队中拼车车辆之间的动态互动。我们的方法通过利用局部二分相互依赖图丰富每个智能体的状态信息，并通过整数线性规划（ILP）集中全局协调器优化订单匹配和智能体行为。通过梯度裁剪和局部图采样的增强，我们的GATDDQN提高了可扩展性和鲁棒性。此外，ILP中的后验得分函数捕捉了在线探索与利用之间的权衡，并减少了智能体的潜在过估计偏差，从而提高了所获解的质量。通过广泛实验和验证，BMG-Q 在训练和操作中均表现出了优越性能，相较于基准强化学习框架，在积累奖励上高出约10%，且过估计偏差降低超过50%。此外，它在任务变化和车队规模变化时仍保持了鲁棒性，从而确立了BMG-Q作为推进打车拼车订单分派操作的有效、可扩展且鲁棒框架的地位。 

---
# One-cycle Structured Pruning with Stability Driven Structure Search 

**Title (ZH)**: 基于稳定驱动结构搜索的一周期结构剪枝方法 

**Authors**: Deepak Ghimire, Dayoung Kil, Seonghwan Jeong, Jaesik Park, Seong-heum Kim  

**Link**: [PDF](https://arxiv.org/pdf/2501.13439)  

**Abstract**: Existing structured pruning typically involves multi-stage training procedures that often demand heavy computation. Pruning at initialization, which aims to address this limitation, reduces training costs but struggles with performance. To address these challenges, we propose an efficient framework for one-cycle structured pruning without compromising model performance. In this approach, we integrate pre-training, pruning, and fine-tuning into a single training cycle, referred to as the `one cycle approach'. The core idea is to search for the optimal sub-network during the early stages of network training, guided by norm-based group saliency criteria and structured sparsity regularization. We introduce a novel pruning indicator that determines the stable pruning epoch by assessing the similarity between evolving pruning sub-networks across consecutive training epochs. Also, group sparsity regularization helps to accelerate the pruning process and results in speeding up the entire process. Extensive experiments on datasets, including CIFAR-10/100, and ImageNet, using VGGNet, ResNet, MobileNet, and ViT architectures, demonstrate that our method achieves state-of-the-art accuracy while being one of the most efficient pruning frameworks in terms of training time. The source code will be made publicly available. 

**Abstract (ZH)**: 现有的结构化剪枝通常涉及多阶段训练过程，这往往需要大量的计算资源。初始化剪枝旨在解决这一限制，虽然能降低训练成本，但性能表现较差。为应对这些挑战，我们提出了一种高效的单周期结构化剪枝框架，该框架在不牺牲模型性能的前提下实现结构化剪枝。在该方法中，我们将预训练、剪枝和微调整合到一个单一的训练周期中，称为“单周期方法”。核心思想是在网络训练的早期阶段搜索最优子网络，并通过基于范数的组显著性标准和结构化稀疏正则化来引导这一过程。我们引入了一个新的剪枝指示器，通过评估连续训练周期中演化的剪枝子网络之间的相似性来确定稳定的剪枝周期。此外，组稀疏正则化有助于加速剪枝过程，并最终加速整个过程。在包括CIFAR-10/100和ImageNet数据集上的实验中，使用VGGNet、ResNet、MobileNet和ViT架构表明，我们的方法在训练时间方面是最高效的剪枝框架之一，同时取得了最先进的准确度。源代码将公开发布。 

---
# Softplus Attention with Re-weighting Boosts Length Extrapolation in Large Language Models 

**Title (ZH)**: 软截至注意力与权重重置增强大规模语言模型的长度外推能力 

**Authors**: Bo Gao, Michael W. Spratling  

**Link**: [PDF](https://arxiv.org/pdf/2501.13428)  

**Abstract**: Large language models have achieved remarkable success in recent years, primarily due to the implementation of self-attention mechanisms. However, traditional Softmax attention suffers from numerical instability and reduced performance as the length of inference tokens increases. This paper addresses these issues by decomposing the Softmax operation into a non-linear transformation and the $l_1$-norm. We identify the latter as essential for maintaining model performance. By replacing the non-linear transformation with the Softplus activation function and introducing a dynamic length scale factor for different token lengths based on invariance entropy, we create a novel attention mechanism with performance better than conventional Softmax attention across various inference lengths. To further improve the length extrapolation ability of the proposed attention mechanism, we introduce a re-weighting mechanism that amplifies significant attention weights while diminishing weaker ones, enabling the model to concentrate more effectively on relevant tokens. When combined with our proposed attention mechanism, this approach demonstrates significant promise in managing longer sequences, maintaining nearly constant validation loss even at 16$\times$ the training token length while ensuring numerical stability. Our code is available at: this https URL. 

**Abstract (ZH)**: 近年来，大规模语言模型取得了显著成功，主要归功于自注意力机制的实现。然而，传统的Softmax注意力在推断令牌长度增加时会遭受数值不稳定性且性能下降。本文通过将Softmax操作分解为非线性变换和$l_1$范数，解决了这些问题。我们发现$l_1$范数对于维持模型性能至关重要。通过将非线性变换替换为Softplus激活函数，并引入基于不变熵的动态长度尺度因子，我们提出了一种新的注意力机制，该机制在各种推断长度上的性能优于传统的Softmax注意力。为提高所提出的注意力机制在长度外推上的能力，我们引入了一种重新加权机制，该机制增强了重要的注意力权重同时降低了较弱的权重，使模型更能集中关注相关的令牌。将该重新加权机制与我们提出的注意力机制结合使用，表明在这种机制下，可以有效管理更长的序列，在推断长度达到训练令牌长度的16倍时，验证损失几乎保持恒定，并确保数值稳定性。我们的代码可在以下链接获取：this https URL。 

---
# Rethinking the Sample Relations for Few-Shot Classification 

**Title (ZH)**: 重新思考少样本分类中的样本关系 

**Authors**: Guowei Yin, Sheng Huang, Luwen Huangfu, Yi Zhang, Xiaohong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.13418)  

**Abstract**: Feature quality is paramount for classification performance, particularly in few-shot scenarios. Contrastive learning, a widely adopted technique for enhancing feature quality, leverages sample relations to extract intrinsic features that capture semantic information and has achieved remarkable success in Few-Shot Learning (FSL). Nevertheless, current few-shot contrastive learning approaches often overlook the semantic similarity discrepancies at different granularities when employing the same modeling approach for different sample relations, which limits the potential of few-shot contrastive learning. In this paper, we introduce a straightforward yet effective contrastive learning approach, Multi-Grained Relation Contrastive Learning (MGRCL), as a pre-training feature learning model to boost few-shot learning by meticulously modeling sample relations at different granularities. MGRCL categorizes sample relations into three types: intra-sample relation of the same sample under different transformations, intra-class relation of homogenous samples, and inter-class relation of inhomogeneous samples. In MGRCL, we design Transformation Consistency Learning (TCL) to ensure the rigorous semantic consistency of a sample under different transformations by aligning predictions of input pairs. Furthermore, to preserve discriminative information, we employ Class Contrastive Learning (CCL) to ensure that a sample is always closer to its homogenous samples than its inhomogeneous ones, as homogenous samples share similar semantic content while inhomogeneous samples have different semantic content. Our method is assessed across four popular FSL benchmarks, showing that such a simple pre-training feature learning method surpasses a majority of leading FSL methods. Moreover, our method can be incorporated into other FSL methods as the pre-trained model and help them obtain significant performance gains. 

**Abstract (ZH)**: 特征质量对于分类性能至关重要，尤其是在少量样本场景中。对比学习是一种广泛应用的技术，用于提高特征质量，通过利用样本关系来提取包含语义信息的内在特征，在少量样本学习（Few-Shot Learning, FSL）中取得了显著的成功。然而，当前少量样本对比学习方法在对不同样本关系采用相同建模方法时，往往忽略了不同粒度下的语义相似性差异，这限制了少量样本对比学习的潜力。在本文中，我们提出了一种简单而有效的对比学习方法——多粒度关系对比学习（Multi-Grained Relation Contrastive Learning, MGRCL），作为一种预训练特征学习模型，通过精细建模不同粒度下的样本关系来增强少量样本学习。MGRCL 将样本关系分为三种类型：同一样本在不同变换下的样本内关系、同类别中同质样本之间的类别内关系，以及不同类别中异质样本之间的类别间关系。在 MGRCL 中，我们设计了变换一致性学习（Transformation Consistency Learning, TCL），通过对输入对的预测进行对齐，确保样本在不同变换下的严格语义一致性。此外，为了保留鉴别信息，我们采用了类别对比学习（Class Contrastive Learning, CCL），确保样本总是比异质样本更接近其同质样本，因为同质样本具有相似的语义内容，而异质样本具有不同的语义内容。我们的方法在四个流行的 FSL 基准上进行了评估，结果显示，这种简单的预训练特征学习方法超越了大多数领先 FSL 方法。此外，我们的方法可以作为预训练模型集成到其他 FSL 方法中，帮助它们实现显著的性能提升。 

---
# M3PT: A Transformer for Multimodal, Multi-Party Social Signal Prediction with Person-aware Blockwise Attention 

**Title (ZH)**: M3PT：一种基于人员意识分块注意机制的多模态多党社会信号预测变换器 

**Authors**: Yiming Tang, Abrar Anwar, Jesse Thomason  

**Link**: [PDF](https://arxiv.org/pdf/2501.13416)  

**Abstract**: Understanding social signals in multi-party conversations is important for human-robot interaction and artificial social intelligence. Multi-party interactions include social signals like body pose, head pose, speech, and context-specific activities like acquiring and taking bites of food when dining. Incorporating all the multimodal signals in a multi-party interaction is difficult, and past work tends to build task-specific models for predicting social signals. In this work, we address the challenge of predicting multimodal social signals in multi-party settings in a single model. We introduce M3PT, a causal transformer architecture with modality and temporal blockwise attention masking which allows for the simultaneous processing of multiple social cues across multiple participants and their temporal interactions. This approach better captures social dynamics over time by considering longer horizons of social signals between individuals. We train and evaluate our unified model on the Human-Human Commensality Dataset (HHCD), and demonstrate that using multiple modalities improves bite timing and speaking status prediction. Source code: this https URL 

**Abstract (ZH)**: 理解多 Vaults 情景中的社会信号对于人类-机器人交互和人工智能社会智能具有重要意义。多 Vaults 交互包括像身体姿态、头部姿态、言语以及就餐时获取和咀嚼食物等具体上下文活动中的社会信号。在多 Vaults 交互中整合所有多模态信号是具有一定挑战性的，之前的大部分工作都倾向于为预测社会信号构建特定任务的模型。在本项工作中，我们克服了在多 Vaults 设置中单个模型预测多模态社会信号的挑战。我们提出了 M3PT，一种因果转换器架构，具有模态和时间块状注意掩蔽机制，这使得可以同时处理来自多个参与者及其时间交互的社会提示。这种方法通过考虑个体之间更长时段的社会信号动态，更好地捕捉了社会动力学。我们使用 Human-Human Commensality Dataset (HHCD) 训练和评估了我们的统一模型，并证明使用多种模态可以提高咬食时机和说话状态的预测效果。源代码：[此处链接]

注意："Vaults" 在原文中应为 "Parties"，这里根据语境进行了修正。"Commensality" 指的是共享食物的行为或场合，"Dataset" 是数据集的意思。原文链接替换为具体链接。 

---
# Load and Renewable Energy Forecasting Using Deep Learning for Grid Stability 

**Title (ZH)**: 使用深度学习进行负荷与可再生能源预测以确保电网稳定性 

**Authors**: Kamal Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2501.13412)  

**Abstract**: As the energy landscape changes quickly, grid operators face several challenges, especially when integrating renewable energy sources with the grid. The most important challenge is to balance supply and demand because the solar and wind energy are highly unpredictable. When dealing with such uncertainty, trustworthy short-term load and renewable energy forecasting can help stabilize the grid, maximize energy storage, and guarantee the effective use of renewable resources. Physical models and statistical techniques were the previous approaches employed for this kind of forecasting tasks. In forecasting renewable energy, machine learning and deep learning techniques have recently demonstrated encouraging results. More specifically, the deep learning techniques like CNN and LSTM and the conventional machine learning techniques like regression that are mostly utilized for load and renewable energy forecasting tasks. In this article, we will focus mainly on CNN and LSTM-based forecasting methods. 

**Abstract (ZH)**: 随着能源格局的快速变化，电网运营商面临着诸多挑战，尤其是在将可再生能源接入电网的过程中。最重要的挑战是平衡供需，因为太阳能和风能等可再生能源具有高度的不确定性。在应对这种不确定性时，可靠的短期负荷和可再生能源预测可以帮助稳定电网、最大化储能效率，并确保可再生能源的有效利用。以前，物理模型和统计技术被用于这类预测任务。在可再生能源预测方面，近年来机器学习和深度学习技术已显示出令人鼓舞的结果。具体而言，常用的深度学习技术如卷积神经网络（CNN）和长短期记忆网络（LSTM），以及主要用于负荷和可再生能源预测的传统机器学习技术如回归分析。本文将重点讨论基于CNN和LSTM的预测方法。 

---
# YOLOv8 to YOLO11: A Comprehensive Architecture In-depth Comparative Review 

**Title (ZH)**: YOLOv8 至 YOLO11：全面架构深入对比综述 

**Authors**: Priyanto Hidayatullah, Nurjannah Syakrani, Muhammad Rizqi Sholahuddin, Trisna Gelar, Refdinal Tubagus  

**Link**: [PDF](https://arxiv.org/pdf/2501.13400)  

**Abstract**: In the field of deep learning-based computer vision, YOLO is revolutionary. With respect to deep learning models, YOLO is also the one that is evolving the most rapidly. Unfortunately, not every YOLO model possesses scholarly publications. Moreover, there exists a YOLO model that lacks a publicly accessible official architectural diagram. Naturally, this engenders challenges, such as complicating the understanding of how the model operates in practice. Furthermore, the review articles that are presently available do not delve into the specifics of each model. The objective of this study is to present a comprehensive and in-depth architecture comparison of the four most recent YOLO models, specifically YOLOv8 through YOLO11, thereby enabling readers to quickly grasp not only how each model functions, but also the distinctions between them. To analyze each YOLO version's architecture, we meticulously examined the relevant academic papers, documentation, and scrutinized the source code. The analysis reveals that while each version of YOLO has improvements in architecture and feature extraction, certain blocks remain unchanged. The lack of scholarly publications and official diagrams presents challenges for understanding the model's functionality and future enhancement. Future developers are encouraged to provide these resources. 

**Abstract (ZH)**: 在基于深度学习的计算机视觉领域，YOLO（You Only Look Once）具有革命性的影响。在深度学习模型中，YOLO也是发展速度最快的模型之一。遗憾的是，并非每一个YOLO模型都有正式的学术论文发表。此外，存在一个YOLO模型没有公开的官方架构图。这自然带来了诸多挑战，例如使得理解模型的实际操作变得更加复杂。当前的综述文章也没有详细介绍每个模型的细节。本研究的目标是全面而深入地比较四个最新的YOLO模型（YOLOv8至YOLOv11）的架构，从而使读者能够快速了解每个模型的功能及其之间的差异。为了分析每个YOLO版本的架构，我们仔细研究了相关的学术论文、文档，并检查了源代码。分析结果显示，虽然每个YOLO版本在架构和特征提取方面都有改进，但某些模块保持不变。缺乏学术论文和官方架构图使得理解模型的功能及其未来改进成为挑战。未来的研究人员和开发者被鼓励提供这些资源。 

---
# Concurrent Learning with Aggregated States via Randomized Least Squares Value Iteration 

**Title (ZH)**: 通过随机化最小二乘值迭代进行并发学习与聚合状态估计 

**Authors**: Yan Chen, Qinxun Bai, Yiteng Zhang, Shi Dong, Maria Dimakopoulou, Qi Sun, Zhengyuan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2501.13394)  

**Abstract**: Designing learning agents that explore efficiently in a complex environment has been widely recognized as a fundamental challenge in reinforcement learning. While a number of works have demonstrated the effectiveness of techniques based on randomized value functions on a single agent, it remains unclear, from a theoretical point of view, whether injecting randomization can help a society of agents {\it concurently} explore an environment. The theoretical results %that we established in this work tender an affirmative answer to this question. We adapt the concurrent learning framework to \textit{randomized least-squares value iteration} (RLSVI) with \textit{aggregated state representation}. We demonstrate polynomial worst-case regret bounds in both finite- and infinite-horizon environments. In both setups the per-agent regret decreases at an optimal rate of $\Theta\left(\frac{1}{\sqrt{N}}\right)$, highlighting the advantage of concurent learning. Our algorithm exhibits significantly lower space complexity compared to \cite{russo2019worst} and \cite{agrawal2021improved}. We reduce the space complexity by a factor of $K$ while incurring only a $\sqrt{K}$ increase in the worst-case regret bound, compared to \citep{agrawal2021improved,russo2019worst}. Additionally, we conduct numerical experiments to demonstrate our theoretical findings. 

**Abstract (ZH)**: 在复杂环境中高效探索的学习代理的设计一直被认为是强化学习中的一个基本挑战。虽然一些工作已经证明基于随机化价值函数的技术在单一代理中是有效的，但从理论角度来看，是否可以通过向多代理系统中注入随机化来同时探索环境仍不清楚。在本文中，我们建立的理论结果对这一问题给予了肯定的回答。我们将并发学习框架应用于带聚合状态表示的随机化最小二乘价值迭代（RLSVI）。我们展示了在有限和无限时间 horizons 环境下最差情况下的多项式后悔界限。在两种设置中，每个代理的后悔界限以最优速率 $\Theta\left(\frac{1}{\sqrt{N}}\right)$ 减少，突显了并发学习的优势。与文献 \cite{russo2019worst} 和 \cite{agrawal2021improved} 相比，我们的算法具有显著较低的空间复杂度。我们将空间复杂度减少了 $K$ 倍，同时仅使最差情况下的后悔界限增加了 $\sqrt{K}$ 倍。此外，我们还进行了数值实验以验证我们的理论发现。 

---
# Generative Data Augmentation Challenge: Zero-Shot Speech Synthesis for Personalized Speech Enhancement 

**Title (ZH)**: 生成式数据增强挑战：零样本语音合成在个性化语音增强中的应用 

**Authors**: Jae-Sung Bae, Anastasia Kuznetsova, Dinesh Manocha, John Hershey, Trausti Kristjansson, Minje Kim  

**Link**: [PDF](https://arxiv.org/pdf/2501.13372)  

**Abstract**: This paper presents a new challenge that calls for zero-shot text-to-speech (TTS) systems to augment speech data for the downstream task, personalized speech enhancement (PSE), as part of the Generative Data Augmentation workshop at ICASSP 2025. Collecting high-quality personalized data is challenging due to privacy concerns and technical difficulties in recording audio from the test scene. To address these issues, synthetic data generation using generative models has gained significant attention. In this challenge, participants are tasked first with building zero-shot TTS systems to augment personalized data. Subsequently, PSE systems are asked to be trained with this augmented personalized dataset. Through this challenge, we aim to investigate how the quality of augmented data generated by zero-shot TTS models affects PSE model performance. We also provide baseline experiments using open-source zero-shot TTS models to encourage participation and benchmark advancements. Our baseline code implementation and checkpoints are available online. 

**Abstract (ZH)**: 本文提出了一项新的挑战，要求零样本文本到语音（TTS）系统在下游任务个性化语音增强（PSE）中增加语音数据，作为ICASSP 2025 生成数据增强研讨会的一部分。由于隐私问题以及在测试场景中录制音频的技术难题，高质量个性化数据的收集具有挑战性。为解决这些问题，使用生成模型生成合成数据已引起广泛关注。在此次挑战中，参与者首先需要构建零样本TTS系统以增强个性化数据。之后，要求使用此增强后的个性化数据集训练PSE系统。通过这一挑战，我们旨在研究由零样本TTS模型生成的增强数据质量如何影响PSE模型的性能。我们还提供了基于开源零样本TTS模型的基本实验以鼓励参与并评估进展。我们的基线代码实现与检查点已在线提供。 

---
# A review on development of eco-friendly filters in Nepal for use in cigarettes and masks and Air Pollution Analysis with Machine Learning and SHAP Interpretability 

**Title (ZH)**: 关于尼泊尔可降解滤嘴和口罩滤材发展的综述及基于机器学习和SHAP可解释性的空气污染分析 

**Authors**: Bishwash Paneru, Biplov Paneru, Tanka Mukhiya, Khem Narayan Poudyal  

**Link**: [PDF](https://arxiv.org/pdf/2501.13369)  

**Abstract**: In Nepal, air pollution is a serious public health concern, especially in cities like Kathmandu where particulate matter (PM2.5 and PM10) has a major influence on respiratory health and air quality. The Air Quality Index (AQI) is predicted in this work using a Random Forest Regressor, and the model's predictions are interpreted using SHAP (SHapley Additive exPlanations) analysis. With the lowest Testing RMSE (0.23) and flawless R2 scores (1.00), CatBoost performs better than other models, demonstrating its greater accuracy and generalization which is cross validated using a nested cross validation approach. NowCast Concentration and Raw Concentration are the most important elements influencing AQI values, according to SHAP research, which shows that the machine learning results are highly accurate. Their significance as major contributors to air pollution is highlighted by the fact that high values of these characteristics significantly raise the AQI. This study investigates the Hydrogen-Alpha (HA) biodegradable filter as a novel way to reduce the related health hazards. With removal efficiency of more than 98% for PM2.5 and 99.24% for PM10, the HA filter offers exceptional defense against dangerous airborne particles. These devices, which are biodegradable face masks and cigarette filters, address the environmental issues associated with traditional filters' non-biodegradable trash while also lowering exposure to air contaminants. 

**Abstract (ZH)**: 在尼泊尔，空气污染是一个严重的公共卫生问题，特别是在如加德满都这样的城市中，颗粒物（PM2.5 和 PM10）对呼吸健康和空气质量的影响巨大。本文使用随机森林回归器预测空气质量指数（AQI），并通过 SHAP（SHapley Additive exPlanations）分析解释模型预测结果。CatBoost 模型在测试集上的最低根均方误差（0.23）和完美的 R2 分数（1.00）表明其在交叉验证方法（嵌套交叉验证）验证下具有更高的准确性和泛化能力，优于其他模型。SHAP 分析显示，现况浓度和原始浓度是影响 AQI 值的最重要因素，这表明机器学习的结果非常准确。这些特征的高值显著提高了 AQI，突显了它们作为空气污染主要贡献者的重要性。本文探讨了一种新型的氢-alpha（HA）生物降解过滤器方法，以减少相关的健康危害。该过滤器对 PM2.5 的去除效率超过 98%，对 PM10 的去除效率为 99.24%，提供了对抗有害气溶胶的出色防护。这些设备作为可生物降解的口罩和卷烟过滤器，不仅解决了传统过滤器产生的不可生物降解垃圾所引起的一系列环境问题，还降低了对空气污染物的暴露。 

---
# Enhanced Extractor-Selector Framework and Symmetrization Weighted Binary Cross-Entropy for Edge Detections 

**Title (ZH)**: 增强的提取-选择框架和对称加权二进制交叉熵方法用于边缘检测 

**Authors**: Hao Shu  

**Link**: [PDF](https://arxiv.org/pdf/2501.13365)  

**Abstract**: Recent advancements have demonstrated the effectiveness of the extractor-selector (E-S) framework in edge detection (ED) tasks, which achieves state-of-the-art (SOTA) performance in both quantitative metrics and perceptual quality. However, this method still falls short of fully exploiting the potential of feature extractors, as selectors only operate on highly compressed feature maps that lack diversity and suffer from substantial information loss. Additionally, while union training can improve perceptual quality, the highest evaluation scores are typically obtained without it, creating a trade-off between quantitative accuracy and perceptual fidelity. To address these limitations, we propose an enhanced E-S architecture, which utilizes richer, less-loss feature representations and incorporates auxiliary features during the selection process, thereby improving the effectiveness of the feature selection mechanism. Additionally, we introduce a novel loss function, the Symmetrization Weight Binary Cross-Entropy (SWBCE), which simultaneously emphasizes both the recall of edge pixels and the suppression of erroneous edge predictions, thereby enhancing the predictions both in the perceptual quality and the prediction accuracy. The effectiveness and superiority of our approaches over baseline models, the standard E-S framework, and the standard Weight Binary Cross-Entropy (WBCE) loss function are demonstrated by extensive experiments. For example, our enhanced E-S architecture trained with SWBCE loss function achieves average improvements of 8.25$\%$, 8.01$\%$, and 33.25$\%$ in ODS, OIS, and AP, measured on BIPED2 compared with the baseline models, significantly outperforming the standard E-S method. The results set new benchmarks for ED tasks, and highlight the potential of the methods in beyond. 

**Abstract (ZH)**: 近年来的研究表明，提取-选择（Extractor-Selector，E-S）框架在边缘检测（Edge Detection，ED）任务中非常有效，能够在定量指标和感知质量方面达到最佳表现（State-of-the-Art，SOTA）。然而，这种方法仍未能充分利用特征提取器的潜力，因为选择器仅操作于高度压缩且缺乏多样性的特征图，从而不可避免地导致大量信息丢失。此外，虽然联合训练可以提高感知质量，但通常在不使用联合训练的情况下可以获得最高的评估分数，这在定量准确性与感知保真度之间造成了权衡。为了克服这些限制，我们提出了一种增强的E-S架构，该架构利用更丰富的、信息损失较少的特征表示，并在选择过程中融合辅助特征，从而增强了特征选择机制的有效性。此外，我们引入了一种新的损失函数，即对称权重二进制交叉熵（Symmetrization Weight Binary Cross-Entropy，SWBCE），该损失函数同时强调边缘像素的召回率和错误边缘预测的抑制，从而在感知质量和预测准确性两个方面提升预测性能。通过大量的实验，证明了我们方法的有效性和优越性，相较于基线模型、标准E-S框架和标准加权二进制交叉熵（Weight Binary Cross-Entropy，WBCE）损失函数，我们的方法和增强的E-S架构在SWBCE损失函数下取得了显著的改进，在BIPED2数据集上的ODS、OIS和AP分别平均提高了8.25%、8.01%和33.25%，显著优于标准E-S方法。实验结果确立了新的ED任务基准，并突显了这些方法在边缘检测任务之外的潜力。 

---
# One Fits All: General Mobility Trajectory Modeling via Masked Conditional Diffusion 

**Title (ZH)**: 适用于所有场景：基于遮蔽条件扩散的一体化移动轨迹建模 

**Authors**: Qingyue Long, Can Rong, Huandong Wang, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.13347)  

**Abstract**: Trajectory data play a crucial role in many applications, ranging from network optimization to urban planning. Existing studies on trajectory data are task-specific, and their applicability is limited to the specific tasks on which they have been trained, such as generation, recovery, or prediction. However, the potential of a unified model has not yet been fully explored in trajectory modeling. Although various trajectory tasks differ in inputs, outputs, objectives, and conditions, they share common mobility patterns. Based on these common patterns, we can construct a general framework that enables a single model to address different tasks. However, building a trajectory task-general framework faces two critical challenges: 1) the diversity in the formats of different tasks and 2) the complexity of the conditions imposed on different tasks. In this work, we propose a general trajectory modeling framework via masked conditional diffusion (named GenMove). Specifically, we utilize mask conditions to unify diverse formats. To adapt to complex conditions associated with different tasks, we utilize historical trajectory data to obtain contextual trajectory embeddings, which include rich contexts such as spatiotemporal characteristics and user preferences. Integrating the contextual trajectory embedding into diffusion models through a classifier-free guidance approach allows the model to flexibly adjust its outputs based on different conditions. Extensive experiments on mainstream tasks demonstrate that our model significantly outperforms state-of-the-art baselines, with the highest performance improvement exceeding 13% in generation tasks. 

**Abstract (ZH)**: 轨迹数据在许多应用中发挥着重要作用，从网络优化到城市规划等。现有轨迹数据研究具有特定任务的性质，其适用性仅限于训练任务本身，如生成、恢复或预测。然而，尚未充分探索统一模型在轨迹建模中的潜力。尽管各种轨迹任务在输入、输出、目标和条件上存在差异，但它们共享常见的移动模式。基于这些共性模式，我们可以构建一种通用框架，使单一模型能够应对不同的任务。然而，构建一种适用于各种轨迹任务的框架面临两个关键挑战：1) 不同任务格式的多样性；2) 不同任务所施加条件的复杂性。我们在此工作中提出了一种基于掩码条件扩散的一般轨迹建模框架（命名为GenMove）。具体而言，我们利用掩码条件统一了各种不同的格式。为了适应与不同任务相关联的复杂条件，我们利用历史轨迹数据获取包含丰富的上下文信息（如时空特性和用户偏好）的上下文轨迹嵌入。通过无分类器引导的方法将上下文轨迹嵌入整合到扩散模型中，使模型能够根据不同的条件灵活调整其输出。在主流任务上的广泛实验表明，我们的模型显著优于现有最先进的基线，在生成任务中的性能提升最高超过13%。 

---
# Full-Stack Optimized Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation 

**Title (ZH)**: 针对推荐系统中全 生命周期 序列行为理解的全栈优化大型语言模型 

**Authors**: Rong Shan, Jiachen Zhu, Jianghao Lin, Chenxu Zhu, Bo Chen, Ruiming Tang, Yong Yu, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.13344)  

**Abstract**: In this paper, we address the lifelong sequential behavior incomprehension problem in large language models (LLMs) for recommendation, where LLMs struggle to extract useful information from long user behavior sequences, even within their context limits. To tackle this, we propose ReLLaX (Retrieval-enhanced Large Language models Plus), a framework offering optimization across data, prompt, and parameter levels. At the data level, we introduce Semantic User Behavior Retrieval (SUBR) to reduce sequence heterogeneity, making it easier for LLMs to extract key information. For prompt-level enhancement, we employ Soft Prompt Augmentation (SPA) to inject collaborative knowledge, aligning item representations with recommendation tasks and improving LLMs's exploration of item relationships. Finally, at the parameter level, we propose Component Fully-interactive LoRA (CFLoRA), which enhances LoRA's expressiveness by enabling interactions between its components, allowing better capture of sequential information. Moreover, we present new perspectives to compare current LoRA-based LLM4Rec methods, i.e. from both a composite and a decomposed view. We theoretically demonstrate that the ways they employ LoRA for recommendation are degraded versions of our CFLoRA, with different constraints on atom component interactions. Extensive experiments on three public datasets demonstrate ReLLaX's superiority over existing baselines and its ability to mitigate lifelong sequential behavior incomprehension effectively. 

**Abstract (ZH)**: 在这篇文章中，我们研究了大型语言模型（LLMs）在推荐中的终身序列行为理解问题，LLMs 在提取来自长用户行为序列的有用信息方面存在困难，即使在它们的上下文限制内也是如此。为了解决这个问题，我们提出了一种名为 ReLLaX（检索增强的大规模语言模型增强）的框架，该框架在数据、提示和参数层面提供优化。在数据层面，我们引入了语义用户行为检索（SUBR）来减少序列异质性，使大型语言模型更容易提取关键信息。在提示层面，我们使用软提示增强（SPA）注入协作知识，使项目表示与推荐任务对齐，并提高大型语言模型在探索项目关系方面的探索能力。最后，在参数层面，我们提出了组件完全交互的低秩自适应（CFLoRA），它通过使低秩自适应（LoRA）的组件之间能够相互作用来增强它的表达能力，从而更好地捕捉序列信息。此外，我们从整体和分解的角度提出了新的视角来对比当前基于LoRA的LLM4Rec方法。我们理论证明了它们使用LoRA的方式是我们的CFLoRA的退化版本，具有不同的原子组件交互约束。大规模实验在三个公开数据集上表明，ReLLaX在性能上优于现有基线，并且能够有效缓解终身序列行为理解问题。 

---
# AgentRec: Agent Recommendation Using Sentence Embeddings Aligned to Human Feedback 

**Title (ZH)**: AgentRec：基于与人类反馈对齐的句子嵌入的智能体推荐 

**Authors**: Joshua Park, Yongfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.13333)  

**Abstract**: Multi-agent systems must decide which agent is the most appropriate for a given task. We propose a novel architecture for recommending which LLM agent out of many should perform a task given a natural language prompt by extending the Sentence-BERT (SBERT) encoder model. On test data, we are able to achieve a top-1 accuracy of 92.2% with each classification taking less than 300 milliseconds. In contrast to traditional classification methods, our architecture is computationally cheap, adaptive to new classes, interpretable, and controllable with arbitrary metrics through reinforcement learning. By encoding natural language prompts into sentence embeddings, our model captures the semantic content relevant to recommending an agent. The distance between sentence embeddings that belong to the same agent is then minimized through fine-tuning and aligned to human values through reinforcement learning from human feedback. This allows the classification of natural language prompts based on their nearest neighbors by measuring the cosine similarity between embeddings. This work is made possible through the generation of a synthetic dataset for agent recommendation, which we have open-sourced to the public along with the code for AgentRec recommendation system at this https URL. 

**Abstract (ZH)**: 多智能体系统必须决定哪个智能体最适合执行给定的任务。我们提出了一种新的架构，通过扩展Sentence-BERT（SBERT）编码器模型，来推荐在面对自然语言提示时应由哪个大型语言模型（LLM）智能体执行任务。在测试数据上，我们能够实现92.2%的最高准确率，且每个分类的计算时间少于300毫秒。与传统的分类方法不同，我们的架构计算成本较低、能够适应新的类别、具有可解释性，并可以通过强化学习任意控制。通过将自然语言提示编码为句子嵌入，我们的模型能够捕捉到推荐智能体所需的语义内容。通过微调来最小化属于同一智能体的句子嵌入之间的距离，并通过从人类反馈中进行强化学习来使这些嵌入与人类价值观对齐。这使得可以通过嵌入间余弦相似度来测量自然语言提示的最近邻来进行分类。这项工作得益于我们生成的一个用于智能体推荐的合成数据集，我们已将其开源，并在该网站（上述URL）上提供了AgentRec推荐系统的代码。 

---
# Sparse identification of nonlinear dynamics and Koopman operators with Shallow Recurrent Decoder Networks 

**Title (ZH)**: 使用浅层递归解码网络进行稀疏非线性动力学和库朗特算子的识别 

**Authors**: Mars Liyao Gao, Jan P. Williams, J. Nathan Kutz  

**Link**: [PDF](https://arxiv.org/pdf/2501.13329)  

**Abstract**: Spatiotemporal modeling of real-world data poses a challenging problem due to inherent high dimensionality, measurement noise, and expensive data collection procedures. In this paper, we present Sparse Identification of Nonlinear Dynamics with SHallow REcurrent Decoder networks (SINDy-SHRED), a method to jointly solve the sensing and model identification problems with simple implementation, efficient computation, and robust performance. SINDy-SHRED uses Gated Recurrent Units (GRUs) to model the temporal sequence of sensor measurements along with a shallow decoder network to reconstruct the full spatiotemporal field from the latent state space using only a few available sensors. Our proposed algorithm introduces a SINDy-based regularization; beginning with an arbitrary latent state space, the dynamics of the latent space progressively converges to a SINDy-class functional, provided the projection remains within the set. In restricting SINDy to a linear model, the architecture produces a Koopman-SHRED model which enforces a linear latent space dynamics. We conduct a systematic experimental study including synthetic PDE data, real-world sensor measurements for sea surface temperature, and direct video data. With no explicit encoder, SINDy-SHRED and Koopman-SHRED enable efficient training with minimal hyperparameter tuning and laptop-level computing; further, it demonstrates robust generalization in a variety of applications with minimal to no hyperparameter adjustments. Finally, the interpretable SINDy and Koopman models of latent state dynamics enables accurate long-term video predictions, achieving state-of-the-art performance and outperforming all baseline methods considered, including Convolutional LSTM, PredRNN, ResNet, and SimVP. 

**Abstract (ZH)**: 时空数据的建模是一个具有挑战性的问题，主要原因在于其固有的高维度、测量噪声以及昂贵的数据采集程序。本文中，我们提出了Sparse Identification of Nonlinear Dynamics with SHallow REcurrent Decoder networks（SINDy-SHRED）方法，该方法能够联合解决传感和模型识别问题，并具有简单的实现方式、高效的计算能力和稳健的性能。SINDy-SHRED使用Gated Recurrent Units（GRUs）来建模传感器测量的时间序列，并通过浅层解码网络从潜在状态空间重构出完整的时空场，仅使用少量可用传感器即可实现这一过程。

我们提出了一种基于SINDy的正则化方法；从任意的潜在状态空间开始，随着动态的逐步收敛，潜在空间的动力学将趋向于SINDy类函数，前提是投影仍保持在该集合内。在将SINDy限制为线性模型时，该架构产生了Koopman-SHRED模型，该模型强制潜在空间的动力学为线性。我们进行了全面的实验研究，包括合成偏微分方程（PDE）数据、实际的海洋表面温度传感器测量数据以及直接的视频数据。无显式的编码器下，SINDy-SHRED和Koopman-SHRED能够以最少的超参数调优实现高效的训练，并且在多种应用中表现出稳健的泛化能力，无需或几乎无需超参数调整。最后，可解释的SINDy和Koopman模型可以实现长期视频预测，取得了最先进的性能，并优于所有被考虑的基本方法，包括卷积LSTM、PredRNN、ResNet和SimVP。 

---
# Investigation of the Privacy Concerns in AI Systems for Young Digital Citizens: A Comparative Stakeholder Analysis 

**Title (ZH)**: 对年轻数字公民的AI系统隐私关切的调查：一项比较利益相关者分析 

**Authors**: Molly Campbell, Ankur Barthwal, Sandhya Joshi, Austin Shouli, Ajay Kumar Shrestha  

**Link**: [PDF](https://arxiv.org/pdf/2501.13321)  

**Abstract**: The integration of Artificial Intelligence (AI) systems into technologies used by young digital citizens raises significant privacy concerns. This study investigates these concerns through a comparative analysis of stakeholder perspectives. A total of 252 participants were surveyed, with the analysis focusing on 110 valid responses from parents/educators and 100 from AI professionals after data cleaning. Quantitative methods, including descriptive statistics and Partial Least Squares Structural Equation Modeling, examined five validated constructs: Data Ownership and Control, Parental Data Sharing, Perceived Risks and Benefits, Transparency and Trust, and Education and Awareness. Results showed Education and Awareness significantly influenced data ownership and risk assessment, while Data Ownership and Control strongly impacted Transparency and Trust. Transparency and Trust, along with Perceived Risks and Benefits, showed minimal influence on Parental Data Sharing, suggesting other factors may play a larger role. The study underscores the need for user-centric privacy controls, tailored transparency strategies, and targeted educational initiatives. Incorporating diverse stakeholder perspectives offers actionable insights into ethical AI design and governance, balancing innovation with robust privacy protections to foster trust in a digital age. 

**Abstract (ZH)**: 将上述论文内容或标题翻译成中文，同时确保符合学术规范，如下：

将人工智能（AI）系统集成到年轻数字公民使用的技术中会引发重大的隐私 concerns。本研究通过利益相关者视角的比较分析来探讨这些 concerns。总共调查了 252 位参与者，经过数据分析处理，针对 110 份有效的家长/教育者回应和 100 份AI专业人士的回应进行了分析。定量方法，包括描述性统计和偏最小二乘结构方程模型（PLS-SEM），研究了五个验证性的构念：数据所有权与控制、家长数据共享、感知风险与收益、透明度与信任，以及教育与意识。研究结果表明，教育与意识显著影响了数据所有权和风险评估，而数据所有权与控制则强烈影响了透明度与信任。透明度与信任，以及感知风险与收益，对家长数据共享的影响较小，表明其他因素可能起到更重要的作用。本研究强调了用户中心的隐私控制、定制化的透明度策略以及有针对性的教育倡议的必要性。将多元化利益相关者的视角纳入考虑，提供了有关道德AI设计与治理的实际见解，平衡创新与稳健的隐私保护，以在数字时代培养信任。 

---
# Toward Ethical AI: A Qualitative Analysis of Stakeholder Perspectives 

**Title (ZH)**: 迈向伦理AI：利益相关者视角的定性分析 

**Authors**: Ajay Kumar Shrestha, Sandhya Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2501.13320)  

**Abstract**: As Artificial Intelligence (AI) systems become increasingly integrated into various aspects of daily life, concerns about privacy and ethical accountability are gaining prominence. This study explores stakeholder perspectives on privacy in AI systems, focusing on educators, parents, and AI professionals. Using qualitative analysis of survey responses from 227 participants, the research identifies key privacy risks, including data breaches, ethical misuse, and excessive data collection, alongside perceived benefits such as personalized services, enhanced efficiency, and educational advancements. Stakeholders emphasized the need for transparency, privacy-by-design, user empowerment, and ethical oversight to address privacy concerns effectively. The findings provide actionable insights into balancing the benefits of AI with robust privacy protections, catering to the diverse needs of stakeholders. Recommendations include implementing selective data use, fostering transparency, promoting user autonomy, and integrating ethical principles into AI development. This study contributes to the ongoing discourse on ethical AI, offering guidance for designing privacy-centric systems that align with societal values and build trust among users. By addressing privacy challenges, this research underscores the importance of developing AI technologies that are not only innovative but also ethically sound and responsive to the concerns of all stakeholders. 

**Abstract (ZH)**: 随着人工智能（AI）系统越来越多地渗透到日常生活的各个方面，人们对隐私和伦理问责制的担忧日益凸显。本研究探讨了教育工作者、家长和AI专业人士等利益相关者对AI系统中隐私的看法，着重分析了他们对于隐私风险和益处的认知。通过对227名参与者的调查问卷进行定性分析，研究揭示了主要的隐私风险，包括数据泄露、伦理滥用和过度数据收集，以及个性化服务、提高效率和教育进步等感知到的益处。利益相关者强调了透明度、隐私设计、用户赋能和伦理监督的重要性，以有效解决隐私问题。研究结果提供了平衡AI带来的利益与 robust隐私保护的可行建议，以满足不同利益相关者的需求。建议包括实施选择性数据使用、增强透明度、促进用户自主权，以及将伦理原则融入AI开发过程。本研究推动了伦理AI的持续讨论，为设计符合社会价值观且能建立用户信任的隐私导向系统提供了指导。通过解决隐私挑战，该研究强调了开发既创新又伦理合理、并能响应所有利益相关者关切的AI技术的重要性。 

---
# Watching the AI Watchdogs: A Fairness and Robustness Analysis of AI Safety Moderation Classifiers 

**Title (ZH)**: 《监督AI监督者：AI安全モデレーション分类器的公平性和鲁棒性分析》 

**Authors**: Akshit Achara, Anshuman Chhabra  

**Link**: [PDF](https://arxiv.org/pdf/2501.13302)  

**Abstract**: AI Safety Moderation (ASM) classifiers are designed to moderate content on social media platforms and to serve as guardrails that prevent Large Language Models (LLMs) from being fine-tuned on unsafe inputs. Owing to their potential for disparate impact, it is crucial to ensure that these classifiers: (1) do not unfairly classify content belonging to users from minority groups as unsafe compared to those from majority groups and (2) that their behavior remains robust and consistent across similar inputs. In this work, we thus examine the fairness and robustness of four widely-used, closed-source ASM classifiers: OpenAI Moderation API, Perspective API, Google Cloud Natural Language (GCNL) API, and Clarifai API. We assess fairness using metrics such as demographic parity and conditional statistical parity, comparing their performance against ASM models and a fair-only baseline. Additionally, we analyze robustness by testing the classifiers' sensitivity to small and natural input perturbations. Our findings reveal potential fairness and robustness gaps, highlighting the need to mitigate these issues in future versions of these models. 

**Abstract (ZH)**: AI安全审查（ASM）分类器旨在监管社交媒体平台上的内容，并充当防止大型语言模型（LLMs）在不安全输入上进行微调的护栏。由于这些分类器可能对不同用户群体产生不同的影响，因此确保它们的功能至关重要：（1）不不公平地将少数群体用户的內容分类为不安全，而对多数群体用户则不这样做；（2）其行为在面对类似输入时保持一致性和鲁棒性。因此，本研究专注于评估四种广泛使用的封闭源ASM分类器的公平性和鲁棒性：OpenAI 审查 API、Perspective API、Google Cloud 自然语言（GCNL）API 和 Clarifai API。我们使用诸如人口统计平等等指标来评估公平性，并将其性能与ASM模型和仅公平的基线进行比较。此外，我们通过测试分类器对小规模和自然输入扰动的敏感性来分析其鲁棒性。我们的研究发现表明存在潜在的公平性和鲁棒性差距，突显出在未来版本模型中减轻这些问题的需求。 

---
# RAMQA: A Unified Framework for Retrieval-Augmented Multi-Modal Question Answering 

**Title (ZH)**: RAMQA：一种统一的检索增强多模态问答框架 

**Authors**: Yang Bai, Christan Earl Grant, Daisy Zhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.13297)  

**Abstract**: Multi-modal retrieval-augmented Question Answering (MRAQA), integrating text and images, has gained significant attention in information retrieval (IR) and natural language processing (NLP). Traditional ranking methods rely on small encoder-based language models, which are incompatible with modern decoder-based generative large language models (LLMs) that have advanced various NLP tasks. To bridge this gap, we propose RAMQA, a unified framework combining learning-to-rank methods with generative permutation-enhanced ranking techniques. We first train a pointwise multi-modal ranker using LLaVA as the backbone. Then, we apply instruction tuning to train a LLaMA model for re-ranking the top-k documents using an innovative autoregressive multi-task learning approach. Our generative ranking model generates re-ranked document IDs and specific answers from document candidates in various permutations. Experiments on two MRAQA benchmarks, WebQA and MultiModalQA, show significant improvements over strong baselines, highlighting the effectiveness of our approach. Code and data are available at: this https URL 

**Abstract (ZH)**: 多模态检索增强的问答（MRAQA），结合文本和图像，已在信息检索（IR）和自然语言处理（NLP）领域引起了广泛关注。传统的排名方法依赖于小型的基于编码器的语言模型，这些模型与现代基于解码器的生成型大型语言模型（LLMs）不兼容，后者已在多种NLP任务中取得进展。为了弥补这一差距，我们提出了一种结合了学习到排名方法和生成型排列增强排名技术的统一框架RAMQA。我们首先使用LLaVA作为骨干训练一个点wise多模态排名器。然后，我们通过创新的自回归多任务学习方法对LLaMA模型进行指令微调，以对前k个文档进行重新排名。我们的生成型排名模型从文档候选集中生成重新排排名的文档ID和特定答案的各种排列。在两个MRAQA基准数据集WebQA和MultiModalQA上的实验表明，我们的方法在强基线方法上取得了显著的改进，突显了我们方法的有效性。相关代码和数据可在以下链接获取：this https URL 

---
# Toyteller: AI-powered Visual Storytelling Through Toy-Playing with Character Symbols 

**Title (ZH)**: Toyteller：通过角色符号玩偶叙事的AI驱动视觉故事讲述 

**Authors**: John Joon Young Chung, Melissa Roemmele, Max Kreminski  

**Link**: [PDF](https://arxiv.org/pdf/2501.13284)  

**Abstract**: We introduce Toyteller, an AI-powered storytelling system where users generate a mix of story text and visuals by directly manipulating character symbols like they are toy-playing. Anthropomorphized symbol motions can convey rich and nuanced social interactions; Toyteller leverages these motions (1) to let users steer story text generation and (2) as a visual output format that accompanies story text. We enabled motion-steered text generation and text-steered motion generation by mapping motions and text onto a shared semantic space so that large language models and motion generation models can use it as a translational layer. Technical evaluations showed that Toyteller outperforms a competitive baseline, GPT-4o. Our user study identified that toy-playing helps express intentions difficult to verbalize. However, only motions could not express all user intentions, suggesting combining it with other modalities like language. We discuss the design space of toy-playing interactions and implications for technical HCI research on human-AI interaction. 

**Abstract (ZH)**: 我们将介绍Toyteller，这是一个基于AI的故事讲述系统，用户可以通过直接操纵类似于玩具的字符符号来生成文字和视觉内容的混合。拟人化的符号动作可以传达丰富而细腻的社会互动；Toyteller通过这些动作（1）让用户控制故事文字的生成，并且（2）作为与故事文字伴随的视觉输出格式。我们通过将动作和文字映射到共享的语义空间，实现了基于动作的文字生成和基于文字的动作生成。技术评估表明，Toyteller在多个指标上优于竞争baseline模型GPT-4o。我们的用户研究发现，玩具玩耍有助于表达难以用语言描述的意图。然而，仅靠动作无法完全表达所有用户意图，这表明需要结合其他模态（例如语言）。我们讨论了玩具玩耍交互的设计空间及其对人类-AI交互技术HCI研究的潜在影响。 

---
# Experience with GitHub Copilot for Developer Productivity at Zoominfo 

**Title (ZH)**: Zoominfo关于GitHub Copilot对开发者生产力影响的实践经验 

**Authors**: Gal Bakal, Ali Dasdan, Yaniv Katz, Michael Kaufman, Guy Levin  

**Link**: [PDF](https://arxiv.org/pdf/2501.13282)  

**Abstract**: This paper presents a comprehensive evaluation of GitHub Copilot's deployment and impact on developer productivity at Zoominfo, a leading Go-To-Market (GTM) Intelligence Platform. We describe our systematic four-phase approach to evaluating and deploying GitHub Copilot across our engineering organization, involving over 400 developers. Our analysis combines both quantitative metrics, focusing on acceptance rates of suggestions given by GitHub Copilot and qualitative feedback given by developers through developer satisfaction surveys. The results show an average acceptance rate of 33% for suggestions and 20% for lines of code, with high developer satisfaction scores of 72%. We also discuss language-specific performance variations, limitations, and lessons learned from this medium-scale enterprise deployment. Our findings contribute to the growing body of knowledge about AI-assisted software development in enterprise settings. 

**Abstract (ZH)**: 本文全面评估了GitHub Copilot在Zoominfo公司的部署及其对开发人员生产力的影响，Zoominfo是一家领先的市场开发（Go-To-Market, GTM）情报平台。我们描述了一种系统性的四阶段方法，用于评估和部署GitHub Copilot，涉及公司工程团队中的400多名开发人员。我们的分析结合了定量指标，重点评估了GitHub Copilot提供的建议接受率，并通过开发者满意度调查收集了定性反馈。结果显示，建议的接受率为33%，代码行的接受率为20%，开发者满意度评分为72%。我们还讨论了不同编程语言下性能的差异、局限性以及从中得出的教训。这些发现对了解企业在AI辅助软件开发方面的应用提供了补充，丰富了相关领域的知识。 

---
# Let SSMs be ConvNets: State-space Modeling with Optimal Tensor Contractions 

**Title (ZH)**: 将SSMs视为卷积神经网络：基于最优张量收缩的状态空间建模 

**Authors**: Yan Ru Pei  

**Link**: [PDF](https://arxiv.org/pdf/2501.13230)  

**Abstract**: We introduce Centaurus, a class of networks composed of generalized state-space model (SSM) blocks, where the SSM operations can be treated as tensor contractions during training. The optimal order of tensor contractions can then be systematically determined for every SSM block to maximize training efficiency. This allows more flexibility in designing SSM blocks beyond the depthwise-separable configuration commonly implemented. The new design choices will take inspiration from classical convolutional blocks including group convolutions, full convolutions, and bottleneck blocks. We architect the Centaurus network with a mixture of these blocks, to balance between network size and performance, as well as memory and computational efficiency during both training and inference. We show that this heterogeneous network design outperforms its homogeneous counterparts in raw audio processing tasks including keyword spotting, speech denoising, and automatic speech recognition (ASR). For ASR, Centaurus is the first network with competitive performance that can be made fully state-space based, without using any nonlinear recurrence (LSTMs), explicit convolutions (CNNs), or (surrogate) attention mechanism. Source code is available at this http URL 

**Abstract (ZH)**: 我们介绍了一种名为Centaurus的网络架构，该架构由广义状态空间模型（SSM）模块组成，在训练过程中，SSM 操作可以被视为张量收缩。对于每一个 SSM 模块，可以系统地确定最优的张量收缩顺序以最大化训练效率。这为设计 SSM 模块提供了更多灵活性，超越了常见的深度可分离配置。新的设计选择将借鉴经典的卷积模块，包括组卷积、全卷积和瓶颈模块。我们通过混合使用这些模块来架构 Centaurus 网络，以平衡网络大小和性能，以及训练和推理过程中的内存和计算效率。我们展示了这种异构网络设计在原始音频处理任务（包括关键词定位、语音降噪和自动语音识别（ASR））中的表现优于其同构的对照组。对于ASR，Centaurus是第一个无需使用非线性递归（LSTMs）、明确卷积（CNNs）或（代理）注意力机制即可获得竞争力性能的完全基于状态空间的网络。源代码可在以下链接获取：this http URL 

---
# SRMT: Shared Memory for Multi-agent Lifelong Pathfinding 

**Title (ZH)**: SRMT: 共享内存机制在多智能体终身路径规划中的应用 

**Authors**: Alsu Sagirova, Yuri Kuratov, Mikhail Burtsev  

**Link**: [PDF](https://arxiv.org/pdf/2501.13200)  

**Abstract**: Multi-agent reinforcement learning (MARL) demonstrates significant progress in solving cooperative and competitive multi-agent problems in various environments. One of the principal challenges in MARL is the need for explicit prediction of the agents' behavior to achieve cooperation. To resolve this issue, we propose the Shared Recurrent Memory Transformer (SRMT) which extends memory transformers to multi-agent settings by pooling and globally broadcasting individual working memories, enabling agents to exchange information implicitly and coordinate their actions. We evaluate SRMT on the Partially Observable Multi-Agent Pathfinding problem in a toy Bottleneck navigation task that requires agents to pass through a narrow corridor and on a POGEMA benchmark set of tasks. In the Bottleneck task, SRMT consistently outperforms a variety of reinforcement learning baselines, especially under sparse rewards, and generalizes effectively to longer corridors than those seen during training. On POGEMA maps, including Mazes, Random, and MovingAI, SRMT is competitive with recent MARL, hybrid, and planning-based algorithms. These results suggest that incorporating shared recurrent memory into the transformer-based architectures can enhance coordination in decentralized multi-agent systems. The source code for training and evaluation is available on GitHub: this https URL. 

**Abstract (ZH)**: 多智能体强化学习（MARL）在解决各种环境中的合作与竞争多智能体问题上取得了显著进展。MARL面临的其中一个主要挑战是如何显式预测智能体的行为以实现合作。为了解决这一问题，我们提出了一种共享递归记忆变换器（SRMT），它通过聚合和全局广播个体工作记忆，将记忆变换器扩展到多智能体设置，使智能体能够隐式地交流信息并协调其行为。我们使用包含狭窄走廊的玩具瓶颈导航任务和POGEMA基准任务集对SRMT进行了评估。在瓶颈任务中，SRMT在稀疏奖励下始终优于多种强化学习基线，并且能够有效地推广到训练时未见过的更长的走廊。在POGEMA地图上，包括迷宫、随机和MovingAI地图，SRMT与近期的MARL、混合和基于规划的算法具有竞争力。这些结果表明，将共享递归记忆纳入基于变换器的架构中能够增强分布式多智能体系统中的协调能力。关于训练和评估的源代码可在GitHub上获取：https://github.com/your-repo-url。 

---
# Learning in Log-Domain: Subthreshold Analog AI Accelerator Based on Stochastic Gradient Descent 

**Title (ZH)**: 在对数域学习：基于随机梯度下降的亚阈值模拟人工智能加速器 

**Authors**: Momen K Tageldeen, Yacine Belgaid, Vivek Mohan, Zhou Wang, Emmanuel M Drakakis  

**Link**: [PDF](https://arxiv.org/pdf/2501.13181)  

**Abstract**: The rapid proliferation of AI models, coupled with growing demand for edge deployment, necessitates the development of AI hardware that is both high-performance and energy-efficient. In this paper, we propose a novel analog accelerator architecture designed for AI/ML training workloads using stochastic gradient descent with L2 regularization (SGDr). The architecture leverages log-domain circuits in subthreshold MOS and incorporates volatile memory. We establish a mathematical framework for solving SGDr in the continuous time domain and detail the mapping of SGDr learning equations to log-domain circuits. By operating in the analog domain and utilizing weak inversion, the proposed design achieves significant reductions in transistor area and power consumption compared to digital implementations. Experimental results demonstrate that the architecture closely approximates ideal behavior, with a mean square error below 0.87% and precision as low as 8 bits. Furthermore, the architecture supports a wide range of hyperparameters. This work paves the way for energy-efficient analog AI hardware with on-chip training capabilities. 

**Abstract (ZH)**: 人工智能模型的迅速发展以及边缘部署需求的不断增加，要求开发高性能且能效高的AI硬件。本文提出了一种新型的模拟加速器架构，该架构适用于使用L2正则化的随机梯度下降（SGDr）进行AI/ML训练工作负载。该架构利用亚阈值MOS中的对数域电路，并结合了挥发性内存。我们建立了连续时间域解SGDr的数学框架，并详细说明了将SGDr学习方程映射到对数域电路的方法。通过在模拟域工作并利用微弱倒置，所提出的架构在晶体管面积和功耗方面相比于数字实现有显著的减少。实验结果表明，该架构逼近理想行为，均方误差低于0.87%，精度低至8位。此外，该架构支持广泛的超参数。本文为具有片上训练能力的能效模拟AI硬件铺平了道路。 

---
# QuFeX: Quantum feature extraction module for hybrid quantum-classical deep neural networks 

**Title (ZH)**: QuFeX：用于混合量子-经典深度神经网络的量子特征提取模块 

**Authors**: Naman Jain, Amir Kalev  

**Link**: [PDF](https://arxiv.org/pdf/2501.13165)  

**Abstract**: We introduce Quantum Feature Extraction (QuFeX), a novel quantum machine learning module. The proposed module enables feature extraction in a reduced-dimensional space, significantly decreasing the number of parallel evaluations required in typical quantum convolutional neural network architectures. Its design allows seamless integration into deep classical neural networks, making it particularly suitable for hybrid quantum-classical models. As an application of QuFeX, we propose Qu-Net -- a hybrid architecture which integrates QuFeX at the bottleneck of a U-Net architecture. The latter is widely used for image segmentation tasks such as medical imaging and autonomous driving. Our numerical analysis indicates that the Qu-Net can achieve superior segmentation performance compared to a U-Net baseline. These results highlight the potential of QuFeX to enhance deep neural networks by leveraging hybrid computational paradigms, providing a path towards a robust framework for real-world applications requiring precise feature extraction. 

**Abstract (ZH)**: 我们引入了量子特征提取(QuFeX)，这是一种新颖的量子机器学习模块。该模块能够在低维空间中进行特征提取，显著减少了典型量子卷积神经网络架构中所需的并行评估数量。其设计允许无缝集成到深度经典神经网络中，使其特别适用于量子-经典混合模型。QuFeX的应用之一是提出了一种混合架构Qu-Net，该架构在U-Net架构的瓶颈处集成了QuFeX。U-Net架构广泛应用于医学成像和自动驾驶等图像分割任务。数值分析结果表明，Qu-Net在图像分割性能上优于U-Net基线。这些结果突显了QuFeX通过利用混合计算范式来增强深度神经网络的潜力，提供了一条通往适用于需要精确特征提取的现实世界应用的稳健框架的道路。 

---
# AirRadar: Inferring Nationwide Air Quality in China with Deep Neural Networks 

**Title (ZH)**: AirRadar：使用深度神经网络推断中国全国空气质量 

**Authors**: Qiongyan Wang, Yutong Xia, Siru ZHong, Weichuang Li, Yuankai Wu, Shifen Cheng, Junbo Zhang, Yu Zheng, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2501.13141)  

**Abstract**: Monitoring real-time air quality is essential for safeguarding public health and fostering social progress. However, the widespread deployment of air quality monitoring stations is constrained by their significant costs. To address this limitation, we introduce \emph{AirRadar}, a deep neural network designed to accurately infer real-time air quality in locations lacking monitoring stations by utilizing data from existing ones. By leveraging learnable mask tokens, AirRadar reconstructs air quality features in unmonitored regions. Specifically, it operates in two stages: first capturing spatial correlations and then adjusting for distribution shifts. We validate AirRadar's efficacy using a year-long dataset from 1,085 monitoring stations across China, demonstrating its superiority over multiple baselines, even with varying degrees of unobserved data. The source code can be accessed at this https URL. 

**Abstract (ZH)**: 实时监测空气质量对于保障公众健康和促进社会进步至关重要。然而，广泛部署空气质量监测站受到其高昂成本的限制。为了解决这一问题，我们提出了\emph{AirRadar}，这是一种深度神经网络，旨在利用现有监测站的数据来准确推断缺乏监测站地区的实时空气质量。通过利用可学习的掩码令牌，AirRadar在未监测区域重建空气质量特征。具体而言，它分为两个阶段进行操作：首先捕捉空间相关性，然后调整分布偏移。我们使用来自中国1,085个监测站长达一年的数据集验证了AirRadar的有效性，结果显示其在多种基准模型中表现更优，即使存在不同程度的未观测数据也是如此。源代码可通过以下链接访问：[此处填写链接]。 

---
# Forecasting of Bitcoin Prices Using Hashrate Features: Wavelet and Deep Stacking Approach 

**Title (ZH)**: 使用哈希率特征的比特币价格预测：小波与深度叠层方法 

**Authors**: Ramin Mousa, Meysam Afrookhteh, Hooman Khaloo, Amir Ali Bengari, Gholamreza Heidary  

**Link**: [PDF](https://arxiv.org/pdf/2501.13136)  

**Abstract**: Digital currencies have become popular in the last decade due to their non-dependency and decentralized nature. The price of these currencies has seen a lot of fluctuations at times, which has increased the need for prediction. As their most popular, Bitcoin(BTC) has become a research hotspot. The main challenge and trend of digital currencies, especially BTC, is price fluctuations, which require studying the basic price prediction model. This research presents a classification and regression model based on stack deep learning that uses a wavelet to remove noise to predict movements and prices of BTC at different time intervals. The proposed model based on the stacking technique uses models based on deep learning, especially neural networks and transformers, for one, seven, thirty and ninety-day forecasting. Three feature selection models, Chi2, RFE and Embedded, were also applied to the data in the pre-processing stage. The classification model achieved 63\% accuracy for predicting the next day and 64\%, 67\% and 82\% for predicting the seventh, thirty and ninety days, respectively. For daily price forecasting, the percentage error was reduced to 0.58, while the error ranged from 2.72\% to 2.85\% for seven- to ninety-day horizons. These results show that the proposed model performed better than other models in the literature. 

**Abstract (ZH)**: 近年来，由于其非依赖性和去中心化的特点，数字货币变得非常流行。这些货币的价格在某些时候经历了大量波动，这增加了对其预测的需求。随着它们的流行，比特币（BTC）成为了研究热点。数字货币，特别是BTC，的主要挑战和趋势是价格波动，这需要研究基本的价格预测模型。本研究提出了一种基于堆叠深度学习的分类和回归模型，通过小波去除噪声，以预测不同时间间隔的BTC价格和波动情况。基于堆叠技术的所提出的模型使用了深度学习模型，特别是神经网络和变压器，分别进行1天、7天、30天和90天的预测。在预处理阶段还应用了三种特征选择模型：卡方测试（Chi2）、递归特征消除（RFE）和嵌入式方法（Embedded）。分类模型在预测次日时实现了63%的准确性，分别在第7天、第30天和第90天的预测中达到了64%、67%和82%的准确性。对于日成交价格的预测，误差被减少到了0.58%，而在7天至90天的预测区间内，误差范围为2.72%-2.85%。这些结果表明，提出的模型在文献中的其他模型中表现出更优的效果。 

---
# Applications and Challenges of AI and Microscopy in Life Science Research: A Review 

**Title (ZH)**: 人工智能和显微镜在生命科学研究中的应用与挑战：一篇综述 

**Authors**: Himanshu Buckchash, Gyanendra Kumar Verma, Dilip K. Prasad  

**Link**: [PDF](https://arxiv.org/pdf/2501.13135)  

**Abstract**: The complexity of human biology and its intricate systems holds immense potential for advancing human health, disease treatment, and scientific discovery. However, traditional manual methods for studying biological interactions are often constrained by the sheer volume and complexity of biological data. Artificial Intelligence (AI), with its proven ability to analyze vast datasets, offers a transformative approach to addressing these challenges. This paper explores the intersection of AI and microscopy in life sciences, emphasizing their potential applications and associated challenges. We provide a detailed review of how various biological systems can benefit from AI, highlighting the types of data and labeling requirements unique to this domain. Particular attention is given to microscopy data, exploring the specific AI techniques required to process and interpret this information. By addressing challenges such as data heterogeneity and annotation scarcity, we outline potential solutions and emerging trends in the field. Written primarily from an AI perspective, this paper aims to serve as a valuable resource for researchers working at the intersection of AI, microscopy, and biology. It summarizes current advancements, key insights, and open problems, fostering an understanding that encourages interdisciplinary collaborations. By offering a comprehensive yet concise synthesis of the field, this paper aspires to catalyze innovation, promote cross-disciplinary engagement, and accelerate the adoption of AI in life science research. 

**Abstract (ZH)**: 人类生物学的复杂性和其交织的系统蕴含着巨大的潜力，可用于推进人类健康、疾病治疗和科学研究。然而，传统的人工研究方法常常受限于生物数据的庞大和复杂性。人工智能（AI），凭借其分析海量数据的能力，提供了一种变革性的方法来应对这些挑战。本文探讨了AI与显微镜在生命科学中的交叉应用，强调了其潜在应用及其相关挑战。我们详细回顾了各种生物系统如何受益于AI，并突出了该领域特有的数据类型和标注要求。特别关注显微镜数据，探讨了处理和解释这些信息所需的特定AI技术。通过解决数据异质性和标注稀缺性的挑战，本文概述了该领域的潜在解决方案和新兴趋势。本文主要从AI的视角出发，旨在为从事AI、显微镜和生物学交叉研究的科研人员提供有价值的资源。它总结了当前的进展、关键见解和开放问题，促进跨学科合作的理解。通过提供该领域综合而简洁的综述，本文旨在激发创新，促进跨学科互动，并加速AI在生命科学研究中的应用。 

---
# Graph Representation Learning with Diffusion Generative Models 

**Title (ZH)**: 使用扩散生成模型的图表示学习 

**Authors**: Daniel Wesego  

**Link**: [PDF](https://arxiv.org/pdf/2501.13133)  

**Abstract**: Diffusion models have established themselves as state-of-the-art generative models across various data modalities, including images and videos, due to their ability to accurately approximate complex data distributions. Unlike traditional generative approaches such as VAEs and GANs, diffusion models employ a progressive denoising process that transforms noise into meaningful data over multiple iterative steps. This gradual approach enhances their expressiveness and generation quality. Not only that, diffusion models have also been shown to extract meaningful representations from data while learning to generate samples. Despite their success, the application of diffusion models to graph-structured data remains relatively unexplored, primarily due to the discrete nature of graphs, which necessitates discrete diffusion processes distinct from the continuous methods used in other domains. In this work, we leverage the representational capabilities of diffusion models to learn meaningful embeddings for graph data. By training a discrete diffusion model within an autoencoder framework, we enable both effective autoencoding and representation learning tailored to the unique characteristics of graph-structured data. We only need the encoder at the end to extract representations. Our approach demonstrates the potential of discrete diffusion models to be used for graph representation learning. 

**Abstract (ZH)**: 扩散模型已成为各类数据模态（包括图像和视频）的一流生成模型，这得益于它们能够精确逼近复杂的数据分布。与传统的生成方法（如VAEs和GANs）不同，扩散模型通过分步去噪过程，在多个迭代步骤中将噪声逐步转化为有意义的数据，这种方式提高了它们的表达能力和生成质量。此外，扩散模型还能从数据中提取有意义的表示，并学会生成样本。尽管在生成方面取得了成功，但将扩散模型应用于图结构数据仍然相对未被充分探索，主要是由于图的离散性特点，这需要与其它领域使用的连续方法不同的离散扩散过程。在本文中，我们利用扩散模型的表示能力来学习图数据的有意义嵌入。通过在自编码器框架下训练一个离散扩散模型，我们能够实现既有效的自编码又针对图结构数据的独特特性进行表示学习。我们仅在最后使用编码器来提取表示。我们的方法展示了离散扩散模型在图表示学习中的潜在应用价值。 

---
# A Hierarchical Reinforcement Learning Framework for Multi-UAV Combat Using Leader-Follower Strategy 

**Title (ZH)**: 使用领导者-追随者策略的多无人机协同作战分层强化学习框架 

**Authors**: Jinhui Pang, Jinglin He, Noureldin Mohamed Abdelaal Ahmed Mohamed, Changqing Lin, Zhihui Zhang, Xiaoshuai Hao  

**Link**: [PDF](https://arxiv.org/pdf/2501.13132)  

**Abstract**: Multi-UAV air combat is a complex task involving multiple autonomous UAVs, an evolving field in both aerospace and artificial intelligence. This paper aims to enhance adversarial performance through collaborative strategies. Previous approaches predominantly discretize the action space into predefined actions, limiting UAV maneuverability and complex strategy implementation. Others simplify the problem to 1v1 combat, neglecting the cooperative dynamics among multiple UAVs. To address the high-dimensional challenges inherent in six-degree-of-freedom space and improve cooperation, we propose a hierarchical framework utilizing the Leader-Follower Multi-Agent Proximal Policy Optimization (LFMAPPO) strategy. Specifically, the framework is structured into three levels. The top level conducts a macro-level assessment of the environment and guides execution policy. The middle level determines the angle of the desired action. The bottom level generates precise action commands for the high-dimensional action space. Moreover, we optimize the state-value functions by assigning distinct roles with the leader-follower strategy to train the top-level policy, followers estimate the leader's utility, promoting effective cooperation among agents. Additionally, the incorporation of a target selector, aligned with the UAVs' posture, assesses the threat level of targets. Finally, simulation experiments validate the effectiveness of our proposed method. 

**Abstract (ZH)**: 多无人机空中格斗是一项涉及多个自主无人机的复杂任务，是航空航天和人工智能领域的不断发展的领域。本文旨在通过协作策略增强对抗性能。之前的许多方法主要将动作空间离散化为预定义的动作，这限制了无人机的操作灵活性和复杂策略的实施。其他方法简化了问题为一对一对抗，忽略了多无人机间的协同动态。为了解决六自由度空间中固有的高维挑战并提高协同性，我们提出了一种基于领导者-跟随者多智能体增强策略优化（Leader-Follower Multi-Agent Proximal Policy Optimization, LFMAPPO）策略的分层框架。具体而言，该框架分为三个层次。顶层进行宏观环境评估并指导执行策略；中间层确定所需动作的角度；底层生成高维动作空间的精确动作指令。此外，我们通过领导者-跟随者策略为顶层策略分配不同的角色来优化状态值函数，跟随者估算领导者的效用，促进智能体间的有效协同。同时，引入与无人机姿态相协调的目标选择器，评估目标威胁级别。最后，仿真实验验证了我们所提出方法的有效性。 

---
# Preference Curriculum: LLMs Should Always Be Pretrained on Their Preferred Data 

**Title (ZH)**: 偏好 Curriculum：大规模语言模型应在首选数据上始终进行预训练 

**Authors**: Xuemiao Zhang, Liangyu Xu, Feiyu Duan, Yongwei Zhou, Sirui Wang, Jingang Wang, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2501.13126)  

**Abstract**: Current large language models (LLMs) generally utilize a consistent data distribution throughout the entire pretraining process. However, as the model's ability improves, it intuitively should be pretrained with differentiated data. To achieve it, we propose the Perplexity Difference based Preference Curriculum learning (PDPC) framework, which always perceives and uses the data preferred by LLMs to train and boost them. Firstly, we introduce the PD metric to measure the difference in how well strong and weak models fit the samples. Samples with high PD are more challenging for weak models to learn and are more suitable to be arranged in the later stage of pretraining. Secondly, we propose the PD preference function to approximate the model and predict the data preference of the LLM at any time, so as to complete the arrangement of the entire data offline and ensure continuous training without interruption. Experimental results on 1.3B and 3B models demonstrate that our PDPC significantly surpasses baselines. Notably, the 3B model achieved more substantial gains, with an increased average accuracy of over 4.1% across various benchmarks. 

**Abstract (ZH)**: 当前的大规模语言模型（LLMs）通常在整个预训练过程中采用一致的数据分布。然而，随着模型能力的提高，直观上应该使用差异化的数据来进行预训练。为实现这一点，我们提出了基于困惑度差异的偏好 Curriculum 学习框架（PDPC），该框架始终感知并利用 LLM 更偏好使用的数据来训练和提升模型。首先，我们引入了 PD 指标来衡量强模型和弱模型在拟合样本方面差异的程度。高 PD 的样本对于弱模型来说更具挑战性，更适合安排在预训练的后期阶段。其次，我们提出了 PD 偏好函数来近似模型，并预测 LLM 在任意时间的数据偏好，从而在线下完成整个数据的排列，确保连续训练不间断。我们在1.3B和3B模型上的实验结果显示，我们的 PDPC 显著优于基线模型。值得注意的是，3B模型取得了更大的进步，各基准测试的平均准确率提高了超过4.1%。 

---
# Generating Plausible Distractors for Multiple-Choice Questions via Student Choice Prediction 

**Title (ZH)**: 通过学生选择预测生成合理的多项选择题干扰项 

**Authors**: Yooseop Lee, Suin Kim, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2501.13125)  

**Abstract**: In designing multiple-choice questions (MCQs) in education, creating plausible distractors is crucial for identifying students' misconceptions and gaps in knowledge and accurately assessing their understanding. However, prior studies on distractor generation have not paid sufficient attention to enhancing the difficulty of distractors, resulting in reduced effectiveness of MCQs. This study presents a pipeline for training a model to generate distractors that are more likely to be selected by students. First, we train a pairwise ranker to reason about students' misconceptions and assess the relative plausibility of two distractors. Using this model, we create a dataset of pairwise distractor ranks and then train a distractor generator via Direct Preference Optimization (DPO) to generate more plausible distractors. Experiments on computer science subjects (Python, DB, MLDL) demonstrate that our pairwise ranker effectively identifies students' potential misunderstandings and achieves ranking accuracy comparable to human experts. Furthermore, our distractor generator outperforms several baselines in generating plausible distractors and produces questions with a higher item discrimination index (DI). 

**Abstract (ZH)**: 在教育中设计多项选择题（MCQs）时，创造合理的诱饵选项对于识别学生的思想谬误和知识漏洞，并准确评估其理解能力至关重要。然而，先前关于诱饵生成的研究并未充分关注增强诱饵难度的问题，从而降低了MCQs的有效性。本研究提出了一种管道来训练一个生成更可能被学生选择的诱饵选项的模型。首先，我们训练了一个成对排序器来分析学生的错误观念，并评估两个诱饵选项的相对合理性。借助该模型，我们创建了一个成对诱饵选项的排序数据集，然后利用直接偏好优化（DPO）训练一个诱饵生成器，以生成更合理的诱饵选项。在计算机科学科目（Python、DB、MLDL）上的实验表明，我们的成对排序器有效识别了学生可能存在的误解，并且其排序准确度达到了与人类专家相当的水平。此外，我们的诱饵生成器在生成合理的诱饵选项方面优于多个基线模型，并产生了具有更高项目区分指数（DI）的问题。 

---
# Debate Helps Weak-to-Strong Generalization 

**Title (ZH)**: 辩论有助于从弱泛化到强泛化的提升 

**Authors**: Hao Lang, Fei Huang, Yongbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.13124)  

**Abstract**: Common methods for aligning already-capable models with desired behavior rely on the ability of humans to provide supervision. However, future superhuman models will surpass the capability of humans. Therefore, humans will only be able to weakly supervise superhuman models. This expected deficiency of human evaluation would weaken the safety of future AI systems. Scalable oversight and weak-to-strong generalization are two complementary approaches to tackle this issue. In this paper, we attempt to combine the strengths of these two approaches to further improve alignment. Specifically, we investigate ways of improving human supervision with a strong pretrained model and then supervise the strong model with enhanced weak human supervision. To make iterative empirical progress, we consider an analogy: can we use a strong model to improve weak model supervision and then use it to supervise the strong model? We empirically test it by finetuning a small weak model on ground truth labels with the additional help from a large strong model, and then finetuning the strong model on labels generated by the weak model. We find that debate can assist a weak model in extracting trustworthy information from an untrustworthy strong model, which provides leverage as context on samples when training a weak model. We also show that an ensemble of weak models helps exploit long arguments generated by strong model debaters and obtain a more robust supervision estimate. Extensive experiments on the OpenAI weak-to-strong NLP benchmarks show that the combination approach leads to better alignment, which indicates that debate has the potential to help weak-to-strong generalization. 

**Abstract (ZH)**: 现有的方法和框架用于将已有能力的模型与期望的行为进行对齐，通常依赖于人类监督的能力。然而，未来超越人类能力的超级模型将会超过人类的能力。因此，人类将只能进行弱监督。这种人类评价的不足将削弱未来AI系统的安全性。可扩展的监督和从弱到强的一般化是应对这一问题的两种互补方法。本文尝试结合这两种方法的优点，进一步提高对齐效果。具体而言，我们研究了通过强预训练模型改进人类监督的方法，然后使用增强的弱人类监督来监督强模型。为了进行迭代实验性进展，我们考虑了一个类比：我们可以使用强模型来改进弱模型的监督，然后再使用它来监督强模型吗？我们通过在大型强模型的帮助下对小型弱模型进行微调，并使用弱模型生成的标签对强模型进行微调，进行了实证测试。我们发现，辩论可以帮助弱模型从不值得信赖的强模型中提取可靠的信息，这在训练弱模型时提供了上下文作为参考。我们还展示了弱模型集合如何利用强模型辩论者生成的长论证，获得更加稳健的监督估计。在OpenAI的弱到强自然语言处理基准测试上进行的广泛实验表明，结合方法能够提高对齐效果，这表明辩论有可能有助于弱到强的一般化。 

---
# Zero-Shot Verification-guided Chain of Thoughts 

**Title (ZH)**: 零样本验证引导的推理链 

**Authors**: Jishnu Ray Chowdhury, Cornelia Caragea  

**Link**: [PDF](https://arxiv.org/pdf/2501.13122)  

**Abstract**: Previous works have demonstrated the effectiveness of Chain-of-Thought (COT) prompts and verifiers in guiding Large Language Models (LLMs) through the space of reasoning. However, most such studies either use a fine-tuned verifier or rely on manually handcrafted few-shot examples. In contrast, in this paper, we focus on LLM-based self-verification of self-generated reasoning steps via COT prompts in a completely zero-shot regime. To explore this setting, we design a new zero-shot prompt, which we call COT STEP, to aid zero-shot decomposition of reasoning steps and design two new zero-shot prompts for LLM-based verifiers. We evaluate the verifiers' ability to classify the correctness of reasoning chains and explore different ways to use verifier scores in guiding reasoning for various mathematical and commonsense reasoning tasks with different LLMs. 

**Abstract (ZH)**: 先前的研究证明了Chain-of-Thought (COT) 提示和验证器在引导大语言模型（LLMs）进行推理方面的有效性。然而，大多数此类研究要么使用微调的验证器，要么依赖于手工构建的少量示例。相比之下，在本论文中，我们关注LLM本身的自验证机制，通过COT提示在完全零样本的情况下对自我生成的推理步骤进行自验证。为了探索这一环境，我们设计了一种新的零样本提示，我们称之为COT STEP，以帮助零样本分解推理步骤，并设计了两种新的零样本提示用于基于LLM的验证器。我们评估了验证器区分推理链正确性的能力，并探索了使用验证器分数指导不同LLM的各种数学和常识推理任务的各种方法。 

---
# Episodic Memories Generation and Evaluation Benchmark for Large Language Models 

**Title (ZH)**: 大型语言模型的 episodic 记忆生成与评估基准 

**Authors**: Alexis Huet, Zied Ben Houidi, Dario Rossi  

**Link**: [PDF](https://arxiv.org/pdf/2501.13121)  

**Abstract**: Episodic memory -- the ability to recall specific events grounded in time and space -- is a cornerstone of human cognition, enabling not only coherent storytelling, but also planning and decision-making. Despite their remarkable capabilities, Large Language Models (LLMs) lack a robust mechanism for episodic memory: we argue that integrating episodic memory capabilities into LLM is essential for advancing AI towards human-like cognition, increasing their potential to reason consistently and ground their output in real-world episodic events, hence avoiding confabulations. To address this challenge, we introduce a comprehensive framework to model and evaluate LLM episodic memory capabilities. Drawing inspiration from cognitive science, we develop a structured approach to represent episodic events, encapsulating temporal and spatial contexts, involved entities, and detailed descriptions. We synthesize a unique episodic memory benchmark, free from contamination, and release open source code and datasets to assess LLM performance across various recall and episodic reasoning tasks. Our evaluation of state-of-the-art models, including GPT-4 and Claude variants, Llama 3.1, and o1-mini, reveals that even the most advanced LLMs struggle with episodic memory tasks, particularly when dealing with multiple related events or complex spatio-temporal relationships -- even in contexts as short as 10k-100k tokens. 

**Abstract (ZH)**: episodic 记忆是指回忆特定时空背景下的事件的能力，这是人类认知的基础，不仅使连贯叙述故事成为可能，还使规划和决策成为可能。虽然大语言模型（LLMs）在许多方面表现出色，但它们缺乏一种坚固的表象记忆机制：我们认为，将表象记忆能力整合到LLMs中对于推进人工智能向类人认知发展至关重要。这将增加它们在一致推理和将输出与现实世界的表象事件联系起来方面的潜力，从而避免编造。为应对这一挑战，我们提出了一个综合框架来建模和评估LLMs的表象记忆能力。受到认知科学的启发，我们开发了一种结构化方法来表示表象事件，封装时间、空间背景、涉及的实体以及详细的描述。我们合成了一种独特的表象记忆基准，不受污染，并开源代码和数据集以评估LLMs在各种回忆和表象推理任务中的表现。对当前最先进的模型（包括GPT-4和Claude变体、Llama 3.1和o1-mini）的评估表明，即使是最先进的LLMs在处理表象记忆任务时也存在挑战，特别是在处理多个相关事件或复杂的时间-空间关系时——即使是在仅10k-100k标记的上下文中也是如此。 

---
# Multilinguality in LLM-Designed Reward Functions for Restless Bandits: Effects on Task Performance and Fairness 

**Title (ZH)**: LLM设计的奖励函数中的多语言性对活跃臂问题任务性能和公平性的影响 

**Authors**: Ambreesh Parthasarathy, Chandrasekar Subramanian, Ganesh Senrayan, Shreyash Adappanavar, Aparna Taneja, Balaraman Ravindran, Milind Tambe  

**Link**: [PDF](https://arxiv.org/pdf/2501.13120)  

**Abstract**: Restless Multi-Armed Bandits (RMABs) have been successfully applied to resource allocation problems in a variety of settings, including public health. With the rapid development of powerful large language models (LLMs), they are increasingly used to design reward functions to better match human preferences. Recent work has shown that LLMs can be used to tailor automated allocation decisions to community needs using language prompts. However, this has been studied primarily for English prompts and with a focus on task performance only. This can be an issue since grassroots workers, especially in developing countries like India, prefer to work in local languages, some of which are low-resource. Further, given the nature of the problem, biases along population groups unintended by the user are also undesirable. In this work, we study the effects on both task performance and fairness when the DLM algorithm, a recent work on using LLMs to design reward functions for RMABs, is prompted with non-English language commands. Specifically, we run the model on a synthetic environment for various prompts translated into multiple languages. The prompts themselves vary in complexity. Our results show that the LLM-proposed reward functions are significantly better when prompted in English compared to other languages. We also find that the exact phrasing of the prompt impacts task performance. Further, as prompt complexity increases, performance worsens for all languages; however, it is more robust with English prompts than with lower-resource languages. On the fairness side, we find that low-resource languages and more complex prompts are both highly likely to create unfairness along unintended dimensions. 

**Abstract (ZH)**: restless 多臂老虎机（RMABs）已经在多种场景下的资源分配问题中得到了成功的应用，包括公共卫生领域。随着强大语言模型（LLMs）的快速发展，它们正越来越多地被用于设计奖励函数，以更好地匹配人类偏好。最近的研究表明，LLMs可以使用语言提示来定制自动化的资源分配决策以满足社区需求。然而，这一研究主要集中在英语提示上，并且仅关注任务性能。这可能是一个问题，因为草根工作者，尤其是像印度这样的发展中地区的工作者，更倾向于使用地方语言工作，其中一些语言是资源匮乏的语言。此外，鉴于问题的性质，由用户无意引入的人群组偏见也是不希望的。在本项工作中，我们研究了当使用 LLMs 设计奖励函数的 DLM 算法被提示非英语语言命令时，其对任务性能和公平性的影响。具体来说，我们在多种语言翻译成的多个提示中运行了该模型，并在合成环境中进行了各种提示试验。提示本身在复杂度上有所不同。我们的研究表明，与用其他语言提示相比，用英语提示时提供的 LLM 设计的奖励函数显著更好。我们还发现，提示的具体措辞对任务性能有影响。进一步地，随着提示复杂度的增加，所有语言的性能都会下降；然而，使用英语提示比使用资源匮乏的语言提示时表现更为稳定。在公平性方面，我们发现低资源语言和更复杂的提示都很可能在未预期的方面造成不公正。 

---
# MyGO Multiplex CoT: A Method for Self-Reflection in Large Language Models via Double Chain of Thought Thinking 

**Title (ZH)**: MyGO 多重共思：一种通过双重链式思考在大规模语言模型中实现自我反思的方法 

**Authors**: Shihao Ji, Zihui Song, Fucheng Zhong, Jisen Jia, Zhaobo Wu, Zheyi Cao, Tianhao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2501.13117)  

**Abstract**: Recent advancements in large language models (LLMs) have demonstrated their impressive abilities in various reasoning and decision-making tasks. However, the quality and coherence of the reasoning process can still benefit from enhanced introspection and self-reflection. In this paper, we introduce Multiplex CoT (Chain of Thought), a method that enables LLMs to simulate a form of self-review while reasoning, by initiating double Chain of Thought (CoT) thinking. Multiplex CoT leverages the power of iterative reasoning, where the model generates an initial chain of thought and subsequently critiques and refines this reasoning with a second round of thought generation. This recursive approach allows for more coherent, logical, and robust answers, improving the overall decision-making process. We demonstrate how this method can be effectively implemented using simple prompt engineering in existing LLM architectures, achieving an effect similar to that of the Learning-Refinement Model (LRM) without the need for additional training. Additionally, we present a practical guide for implementing the method in Google Colab, enabling easy integration into real-world applications. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）在各种推理和决策任务中展现出令人印象深刻的性能。然而，推理过程的质量和连贯性仍可从增强的内省和自我反思中受益。本文介绍了一种名为Multiplex CoT（链式思维）的方法，该方法允许LLMs在推理过程中模拟一种自我审查的形式，通过启动双重链式思维（CoT）思考实现这一点。Multiplex CoT利用迭代推理的力量，模型首先生成初始的链式思维，然后通过第二轮思维生成对其进行批判和修订。这种递归方法可以提高答案的连贯性、逻辑性和鲁棒性，从而改进整体的决策过程。我们展示了如何通过简单的提示工程技术在现有的LLM架构中有效地实施该方法，从而实现类似于学习精炼模型（LRM）的效果，无需额外的训练。此外，我们提供了一种实用指南，说明如何在Google Colab中实施该方法，便于将其无缝集成到实际应用中。 

---
# Dagger Behind Smile: Fool LLMs with a Happy Ending Story 

**Title (ZH)**: 微笑背后的伪装：用一个圆满结局的故事愚弄LLMs 

**Authors**: Xurui Song, Zhixin Xie, Shuo Huai, Jiayi Kong, Jun Luo  

**Link**: [PDF](https://arxiv.org/pdf/2501.13115)  

**Abstract**: The wide adoption of Large Language Models (LLMs) has attracted significant attention from \textit{jailbreak} attacks, where adversarial prompts crafted through optimization or manual design exploit LLMs to generate malicious content. However, optimization-based attacks have limited efficiency and transferability, while manual designs are either easily detectable or demand intricate interactions with LLMs. In this paper, we first point out a novel perspective for jailbreak attacks: LLMs are more responsive to \textit{positive} prompts. Based on this, we deploy Happy Ending Attack (HEA) to wrap up a malicious request in a scenario template involving a positive prompt formed mainly via a \textit{happy ending}, it thus fools LLMs into jailbreaking either immediately or at a follow-up malicious request. This has made HEA both efficient and effective, as it requires only up to two steps to fully jailbreak LLMs. Extensive experiments show that our HEA can successfully jailbreak on state-of-the-art LLMs, including GPT-4o, Llama3-70b, Gemini-pro, and achieves 88.79\% Attack Success Rate on average. We also provide potential quantitative explanations for the success of HEA. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的广泛应用引发了对“逃逸攻击”(jailbreak attacks)的广泛关注，这种攻击通过优化或手动设计的对抗提示利用LLMs生成恶意内容。然而，基于优化的攻击效率有限且缺乏可移植性，而手动设计要么容易被检测到，要么需要复杂的与LLMs的交互。在本文中，我们首先指出“逃逸攻击”的一个新颖视角：LLMs对积极正面的提示更为敏感。基于这一观点，我们部署了Happy Ending Attack（喜 ending 攻击）来将恶意请求嵌入到包含主要通过“完美结局”形成的积极正面提示的场景模板中。这样一来，在第一次请求或随后的恶意请求中，该攻击可以欺骗LLMs进行逃逸。这使HEA既高效又有效，因为它只需要两步即可实现LLMs的完全逃逸。大量的实验表明，我们的HEA能够成功地逃逸最先进的LLMs，包括GPT-4o、Llama3-70b、Gemini-pro，并且平均成功率为88.79%。我们还提供了HEA成功潜在的定量解释。 

---
