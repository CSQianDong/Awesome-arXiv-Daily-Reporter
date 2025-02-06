# On Fairness of Unified Multimodal Large Language Model for Image Generation 

**Title (ZH)**: 统一多模态大语言模型的图像生成公平性研究 

**Authors**: Ming Liu, Hao Chen, Jindong Wang, Liwen Wang, Bhiksha Raj Ramakrishnan, Wensheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.03429)  

**Abstract**: Unified multimodal large language models (U-MLLMs) have demonstrated impressive performance in visual understanding and generation in an end-to-end pipeline. Compared with generation-only models (e.g., Stable Diffusion), U-MLLMs may raise new questions about bias in their outputs, which can be affected by their unified capabilities. This gap is particularly concerning given the under-explored risk of propagating harmful stereotypes. In this paper, we benchmark the latest U-MLLMs and find that most exhibit significant demographic biases, such as gender and race bias. To better understand and mitigate this issue, we propose a locate-then-fix strategy, where we audit and show how the individual model component is affected by bias. Our analysis shows that bias originates primarily from the language model. More interestingly, we observe a "partial alignment" phenomenon in U-MLLMs, where understanding bias appears minimal, but generation bias remains substantial. Thus, we propose a novel balanced preference model to balance the demographic distribution with synthetic data. Experiments demonstrate that our approach reduces demographic bias while preserving semantic fidelity. We hope our findings underscore the need for more holistic interpretation and debiasing strategies of U-MLLMs in the future. 

**Abstract (ZH)**: 统一多模态大语言模型（U-MLLMs）已经在端到端管道中的视觉理解和生成任务中展现了卓越的性能。与仅生成模型（例如Stable Diffusion）相比，U-MLLMs可能会在其输出中产生新的偏差问题，这些问题可能受到其统一能力的影响。鉴于传播有害刻板印象的风险尚未充分探索，这一点尤其令人担忧。在本文中，我们对标了最新的U-MLLMs，并发现大多数模型在性别和种族方面表现出显著的民概况念偏差。为了更深入地理解和缓解这一问题，我们提出了一种“定位-修复”策略，其中我们审计并展示了各个模型组件如何受到偏差的影响。我们的分析表明，偏差主要源于语言模型。更有趣的是，我们发现在U-MLLMs中存在一种“部分对齐”现象，即理解偏差似乎较小，但生成偏差仍然显著。因此，我们提出了一种新颖的平衡偏好模型，该模型旨在通过合成数据平衡民概况念分布与语义保真度。实验结果表明，我们的方法能够减少民概况念偏差而不影响语义保真度。我们希望研究结果能强调对未来U-MLLMs进行全面解释和去偏见策略的需要。 

---
# Think or Step-by-Step? UnZIPping the Black Box in Zero-Shot Prompts 

**Title (ZH)**: 思考还是逐步推理？解开零样本提示黑箱之谜 

**Authors**: Nikta Gohari Sadr, Sangmitra Madhusudan, Ali Emami  

**Link**: [PDF](https://arxiv.org/pdf/2502.03418)  

**Abstract**: Zero-shot prompting techniques have significantly improved the performance of Large Language Models (LLMs). However, we lack a clear understanding of why zero-shot prompts are so effective. For example, in the prompt "Let's think step-by-step," is "think" or "step-by-step" more crucial to its success? Existing interpretability methods, such as gradient-based and attention-based approaches, are computationally intensive and restricted to open-source models. We introduce the ZIP score (Zero-shot Importance of Perturbation score), a versatile metric applicable to both open and closed-source models, based on systematic input word perturbations. Our experiments across four recent LLMs, seven widely-used prompts, and several tasks, reveal interesting patterns in word importance. For instance, while both 'step-by-step' and 'think' show high ZIP scores, which one is more influential depends on the model and task. We validate our method using controlled experiments and compare our results with human judgments, finding that proprietary models align more closely with human intuition regarding word significance. These findings enhance our understanding of LLM behavior and contribute to developing more effective zero-shot prompts and improved model analysis. 

**Abstract (ZH)**: 零样本提示技术显著提高了大型语言模型（LLMs）的性能。然而，我们对零样本提示为何如此有效缺乏清晰的理解。例如，在提示“让我们一步一步地思考”中，“思考”还是“一步一步地”更为关键？现有的可解释性方法，如梯度基和注意力基方法，计算量大且仅限于开源模型。我们引入了ZIP分数（Zero-shot Importance of Perturbation分数），这是一种适用于开源和闭源模型的通用度量标准，基于系统的输入词扰动。我们的实验跨越了四个最新的LLMs、七个广泛使用的提示以及多个任务，揭示了一些有趣的重要词模式。例如，虽然“step-by-step”和“think”都显示出高ZIP分数，但哪一个更具影响力取决于模型和任务。我们使用受控实验验证了我们的方法，并将我们的结果与人类判断进行了比较，发现专有模型在词汇重要性方面与人类直觉更为一致。这些发现增强了我们对LLM行为的理解，并为开发更有效的零样本提示和改进模型分析做出了贡献。 

---
# SPRI: Aligning Large Language Models with Context-Situated Principles 

**Title (ZH)**: SPRI：将大型语言模型与情境化原则对齐 

**Authors**: Hongli Zhan, Muneeza Azmat, Raya Horesh, Junyi Jessy Li, Mikhail Yurochkin  

**Link**: [PDF](https://arxiv.org/pdf/2502.03397)  

**Abstract**: Aligning Large Language Models to integrate and reflect human values, especially for tasks that demand intricate human oversight, is arduous since it is resource-intensive and time-consuming to depend on human expertise for context-specific guidance. Prior work has utilized predefined sets of rules or principles to steer the behavior of models (Bai et al., 2022; Sun et al., 2023). However, these principles tend to be generic, making it challenging to adapt them to each individual input query or context. In this work, we present Situated-PRInciples (SPRI), a framework requiring minimal or no human effort that is designed to automatically generate guiding principles in real-time for each input query and utilize them to align each response. We evaluate SPRI on three tasks, and show that 1) SPRI can derive principles in a complex domain-specific task that leads to on-par performance as expert-crafted ones; 2) SPRI-generated principles lead to instance-specific rubrics that outperform prior LLM-as-a-judge frameworks; 3) using SPRI to generate synthetic SFT data leads to substantial improvement on truthfulness. We release our code and model generations at this https URL. 

**Abstract (ZH)**: 将下面的论文内容或标题翻译成中文，要符合学术规范：

对大型语言模型进行调整，使其集成和反映人类价值观尤为重要，特别是在需要细致的人类监督的任务中。依赖人类专业知识为具体上下文提供指导是资源密集型和耗时的过程。先前的研究利用预定义的规则集或原则来引导模型的行为（Bai等，2022；Sun等，2023）。然而，这些原则通常是概括性的，难以适应每个单独的输入查询或上下文。在此工作中，我们提出了Situated-PRInciples （SPRI）框架，该框架设计用于自动为每个输入查询实时生成指导原则，并利用这些原则对响应进行调整。我们在这项工作中评估了SPRI，展示了以下几点：1）SPRI能够从一个复杂的领域特定任务中导出原则，其性能可与专家制作的原则相媲美；2）SPRI生成的原则导致特定实例的标准优于先前的LLM作为裁判框架；3）利用SPRI生成合成SFT数据显著提高了真实性。我们在此网址发布了我们的代码和模型生成：[请提供具体网址]。 

---
# LIMO: Less is More for Reasoning 

**Title (ZH)**: LIMO：更少即是更多，精简促进推理 

**Authors**: Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.03387)  

**Abstract**: We present a fundamental discovery that challenges our understanding of how complex reasoning emerges in large language models. While conventional wisdom suggests that sophisticated reasoning tasks demand extensive training data (>100,000 examples), we demonstrate that complex mathematical reasoning abilities can be effectively elicited with surprisingly few examples. Through comprehensive experiments, our proposed model LIMO demonstrates unprecedented performance in mathematical reasoning. With merely 817 curated training samples, LIMO achieves 57.1% accuracy on AIME and 94.8% on MATH, improving from previous SFT-based models' 6.5% and 59.2% respectively, while only using 1% of the training data required by previous approaches. LIMO demonstrates exceptional out-of-distribution generalization, achieving 40.5% absolute improvement across 10 diverse benchmarks, outperforming models trained on 100x more data, challenging the notion that SFT leads to memorization rather than generalization. Based on these results, we propose the Less-Is-More Reasoning Hypothesis (LIMO Hypothesis): In foundation models where domain knowledge has been comprehensively encoded during pre-training, sophisticated reasoning capabilities can emerge through minimal but precisely orchestrated demonstrations of cognitive processes. This hypothesis posits that the elicitation threshold for complex reasoning is determined by two key factors: (1) the completeness of the model's encoded knowledge foundation during pre-training, and (2) the effectiveness of post-training examples as "cognitive templates" that show the model how to utilize its knowledge base to solve complex reasoning tasks. To facilitate reproducibility and future research in data-efficient reasoning, we release LIMO as a comprehensive open-source suite at this https URL. 

**Abstract (ZH)**: 我们提出了一个根本性的发现，挑战了我们对大型语言模型中复杂推理是如何产生的理解。尽管传统观点认为复杂的推理任务需要大量的训练数据（>100,000 个示例），我们证明了复杂的数学推理能力可以用出人意料的少量示例有效激发。通过全面的实验，我们提出的模型 LIMO 在数学推理方面取得了前所未有的性能。仅使用 817 个精挑细选的训练样本，LIMO 在 AIME 上达到了 57.1% 的准确率，在 MATH 上达到了 94.8% 的准确率，分别比之前的 SFT 基准模型提高了 50.6% 和 35.6%，同时只使用了之前方法所需训练数据的 1%。LIMO 展示了出色的泛化能力，在 10 个不同的基准测试中取得了 40.5% 的绝对提升，超越了在 100 倍更多数据上训练的模型，挑战了 SFT 导致记忆而非泛化的观点。基于这些结果，我们提出了“少即是多推理假设”（LIMO 假说）：在基础模型中，如果在其预训练期间全面编码了领域知识，那么通过少量但精准调节的认知过程演示，复杂的推理能力可以得以产生。该假设表明，复杂推理的激发阈值由两个关键因素决定：（1）模型在预训练期间编码的知识基础的完整性；（2）后训练示例作为“认知模板”的有效性，展示了模型如何利用其知识库解决复杂的推理任务。为了促进高效推理的可再现性和未来研究，我们在此 https://github.com/阿里云Qwen/LIMO 开放源代码 LIMO 作为全面的开源套件。

注：上述翻译基于您提供的英文文本进行，其中"this https URL"部分保持英文形式，这是标准的做法。如果您需要提供具体的链接内容，请告知。 

---
# High-Fidelity Simultaneous Speech-To-Speech Translation 

**Title (ZH)**: 高保真同时同声翻译 

**Authors**: Tom Labiausse, Laurent Mazaré, Edouard Grave, Patrick Pérez, Alexandre Défossez, Neil Zeghidour  

**Link**: [PDF](https://arxiv.org/pdf/2502.03382)  

**Abstract**: We introduce Hibiki, a decoder-only model for simultaneous speech translation. Hibiki leverages a multistream language model to synchronously process source and target speech, and jointly produces text and audio tokens to perform speech-to-text and speech-to-speech translation. We furthermore address the fundamental challenge of simultaneous interpretation, which unlike its consecutive counterpart, where one waits for the end of the source utterance to start translating, adapts its flow to accumulate just enough context to produce a correct translation in real-time, chunk by chunk. To do so, we introduce a weakly-supervised method that leverages the perplexity of an off-the-shelf text translation system to identify optimal delays on a per-word basis and create aligned synthetic data. After supervised training, Hibiki performs adaptive, simultaneous speech translation with vanilla temperature sampling. On a French-English simultaneous speech translation task, Hibiki demonstrates state-of-the-art performance in translation quality, speaker fidelity and naturalness. Moreover, the simplicity of its inference process makes it compatible with batched translation and even real-time on-device deployment. We provide examples as well as models and inference code. 

**Abstract (ZH)**: 我们介绍了Hibiki，一个只解码器模型，用于同时进行语音翻译。Hibiki利用一个多流语言模型，同步处理源语音和目标语音，并联合生成文本和音频令牌，以执行语音到文本和语音到语音翻译。此外，我们还解决了同时口译这一基本挑战，不同于连续口译（需要等待源话语结束后再开始翻译），同时口译会根据需积累的上下文量，在逐块的基础上实时适应其流动速度，以生成正确的翻译。为此，我们引入了一种弱监督方法，通过利用现成的文本翻译系统困惑度来逐词识别最优延迟，并创建对齐的合成数据。经过监督训练后，Hibiki可以使用基本的温度采样方法进行适应性、实时的语音翻译。在一项法语-英语同时进行语音翻译任务中，Hibiki在翻译质量、说话人忠实度和自然度方面均表现出色。此外，其简单的推理过程使其能够兼容批量翻译，并甚至可以在设备上实时部署。我们还提供了示例、模型和推理代码。 

---
# Integrating automatic speech recognition into remote healthcare interpreting: A pilot study of its impact on interpreting quality 

**Title (ZH)**: 将自动语音识别集成到远程医疗 interpreting 中：对其 interpreting 质量影响的试点研究 

**Authors**: Shiyi Tan, Constantin Orăsan, Sabine Braun  

**Link**: [PDF](https://arxiv.org/pdf/2502.03381)  

**Abstract**: This paper reports on the results from a pilot study investigating the impact of automatic speech recognition (ASR) technology on interpreting quality in remote healthcare interpreting settings. Employing a within-subjects experiment design with four randomised conditions, this study utilises scripted medical consultations to simulate dialogue interpreting tasks. It involves four trainee interpreters with a language combination of Chinese and English. It also gathers participants' experience and perceptions of ASR support through cued retrospective reports and semi-structured interviews. Preliminary data suggest that the availability of ASR, specifically the access to full ASR transcripts and to ChatGPT-generated summaries based on ASR, effectively improved interpreting quality. Varying types of ASR output had different impacts on the distribution of interpreting error types. Participants reported similar interactive experiences with the technology, expressing their preference for full ASR transcripts. This pilot study shows encouraging results of applying ASR to dialogue-based healthcare interpreting and offers insights into the optimal ways to present ASR output to enhance interpreter experience and performance. However, it should be emphasised that the main purpose of this study was to validate the methodology and that further research with a larger sample size is necessary to confirm these findings. 

**Abstract (ZH)**: 本文报道了一项初步研究的结果，该研究旨在探索自动语音识别（ASR）技术在远程医疗口译设置中对口译质量的影响。该研究采用被试内实验设计，随机设置了四种条件，利用标准化的医疗咨询情景模拟对话口译任务。研究对象为四名训练有素的口译员，使用中英双语搭配。此外，通过触发式回顾报告和半结构化访谈收集参与者关于ASR支持的体验和观点。初步数据分析表明，ASR技术特别是全文本ASR转录和基于ASR生成的ChatGPT摘要，显著提升了口译质量。不同类型的ASR输出对口译错误类型分布的影响也各不相同。参与者报告称，他们与技术之间的互动体验大致相似，并倾向于使用完整的ASR转录文本。该初步研究显示了将ASR应用于基于对话的医疗口译中的积极成果，并提供了关于如何呈现ASR输出以增强口译员体验和表现的有效途径的重要见解。然而，必须强调的是，本研究的主要目的是验证方法论的有效性，未来需要更大样本的研究来确认这些发现在更广泛人群中的有效性。 

---
# Demystifying Long Chain-of-Thought Reasoning in LLMs 

**Title (ZH)**: 揭开大型语言模型中长链条推理的奥秘 

**Authors**: Edward Yeo, Yuxuan Tong, Morry Niu, Graham Neubig, Xiang Yue  

**Link**: [PDF](https://arxiv.org/pdf/2502.03373)  

**Abstract**: Scaling inference compute enhances reasoning in large language models (LLMs), with long chains-of-thought (CoTs) enabling strategies like backtracking and error correction. Reinforcement learning (RL) has emerged as a crucial method for developing these capabilities, yet the conditions under which long CoTs emerge remain unclear, and RL training requires careful design choices. In this study, we systematically investigate the mechanics of long CoT reasoning, identifying the key factors that enable models to generate long CoT trajectories. Through extensive supervised fine-tuning (SFT) and RL experiments, we present four main findings: (1) While SFT is not strictly necessary, it simplifies training and improves efficiency; (2) Reasoning capabilities tend to emerge with increased training compute, but their development is not guaranteed, making reward shaping crucial for stabilizing CoT length growth; (3) Scaling verifiable reward signals is critical for RL. We find that leveraging noisy, web-extracted solutions with filtering mechanisms shows strong potential, particularly for out-of-distribution (OOD) tasks such as STEM reasoning; and (4) Core abilities like error correction are inherently present in base models, but incentivizing these skills effectively for complex tasks via RL demands significant compute, and measuring their emergence requires a nuanced approach. These insights provide practical guidance for optimizing training strategies to enhance long CoT reasoning in LLMs. Our code is available at: this https URL. 

**Abstract (ZH)**: 扩展推理计算可以增强大型语言模型（LLMs）的推理能力，而具有长链思考（长CoT）的策略可以支持回溯和错误纠正等方法。强化学习（RL）已成为开发这些能力的关键方法，然而长CoT如何出现的具体条件仍然不清楚，且RL训练需要精心的设计选择。本研究系统地探讨了长CoT推理的机制，确定了使模型能够生成长CoT轨迹的关键因素。通过广泛的监督微调（SFT）和RL实验，我们提出了四项主要发现：（1）虽然SFT不是严格必要的，但它简化了训练并提高了效率；（2）推理能力倾向于随着训练计算量的增加而出现，但其发展并非必然，因此合理的奖赏塑造对于稳定CoT长度增长至关重要；（3）扩展可验证的奖赏信号对RL至关重要。我们发现利用过滤机制下的噪声、从网络提取的解决方案显示出强大的潜力，尤其是在STEM推理等分布外（OOD）任务中；（4）核心能力如错误纠正在基础模型中固有存在，但通过RL有效地激励这些技能以应对复杂任务需要大量计算资源，而且衡量其出现的方式也需精细化。

这些见解为优化训练策略以增强LLMs中的长CoT推理提供了实用指导。我们的代码可在以下链接获取：this https URL。 

---
# Minerva: A Programmable Memory Test Benchmark for Language Models 

**Title (ZH)**: Minerva：一种用于语言模型的可编程内存测试基准 

**Authors**: Menglin Xia, Victor Ruehle, Saravan Rajmohan, Reza Shokri  

**Link**: [PDF](https://arxiv.org/pdf/2502.03358)  

**Abstract**: How effectively can LLM-based AI assistants utilize their memory (context) to perform various tasks? Traditional data benchmarks, which are often manually crafted, suffer from several limitations: they are static, susceptible to overfitting, difficult to interpret, and lack actionable insights--failing to pinpoint the specific capabilities a model lacks when it does not pass a test. In this paper, we present a framework for automatically generating a comprehensive set of tests to evaluate models' abilities to use their memory effectively. Our framework extends the range of capability tests beyond the commonly explored (passkey, key-value, needle in the haystack) search, a dominant focus in the literature. Specifically, we evaluate models on atomic tasks such as searching, recalling, editing, matching, comparing information in context memory, and performing basic operations when inputs are structured into distinct blocks, simulating real-world data. Additionally, we design composite tests to investigate the models' ability to maintain state while operating on memory. Our benchmark enables an interpretable, detailed assessment of memory capabilities of LLMs. 

**Abstract (ZH)**: 基于LLM的AI助手能够有效地利用其记忆（上下文）来执行各种任务吗？传统的数据基准通常是由人工构建的，存在若干局限性：它们是静态的、容易过拟合、难以解释，并且缺乏可操作的洞察——无法准确指出一个模型在测试中未通过时缺少的具体能力。在本文中，我们提出了一种自动生成全面测试集的框架，用于评估模型有效利用其记忆的能力。我们的框架将能力测试的范围扩展到超越文献中通常探索的（如密钥、键值对、大海捞针）搜索任务。具体而言，我们评估模型在原子任务（如搜索、回忆、编辑、根据上下文记忆匹配信息、以及结构化输入时执行基本操作）上的表现，模拟实际数据。此外，我们设计了复合测试来研究模型在操作记忆时维持状态的能力。我们的基准测试能够实现对LLM记忆能力的可解释和详细的评估。 

---
# ECM: A Unified Electronic Circuit Model for Explaining the Emergence of In-Context Learning and Chain-of-Thought in Large Language Model 

**Title (ZH)**: ECM：解释大型语言模型上下文学习和链式思维涌现的统一电子电路模型 

**Authors**: Qiguang Chen, Libo Qin, Jinhao Liu, Dengyun Peng, Jiaqi Wang, Mengkang Hu, Zhi Chen, Wanxiang Che, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.03325)  

**Abstract**: Recent advancements in large language models (LLMs) have led to significant successes across various applications, where the most noticeable is to a series of emerging capabilities, particularly in the areas of In-Context Learning (ICL) and Chain-of-Thought (CoT). To better understand and control model performance, many studies have begun investigating the underlying causes of these phenomena and their impact on task outcomes. However, existing explanatory frameworks predominantly focus on isolating and explaining ICL and CoT independently, leading to an incomplete understanding of their combined influence on model performance. To address this gap, we propose the Electronic Circuit Model (ECM), which provides a foundation for developing scalable, learnable policies and improving the management of AI-generated content. Specifically, ECM conceptualizes model behavior as an electronic circuit: ICL is represented as semantic magnetic field to providing an additional voltage following Faraday's Law, while CoT is modeled as series resistors to constrain the model output performance following Ohm's Law. Experimental results demonstrate that the ECM effectively predicts and explains LLM performance across a variety of prompting strategies. Furthermore, we apply ECM to advanced reasoning strategy optimization on a series of tasks, such as the International Olympiad in Informatics (IOI) and the International Mathematical Olympiad (IMO), achieving competitive performance that surpasses nearly 80% of top human competitors. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）的进步已经在各种应用中取得了显著的成功，其中最显著的是在上下文学习（ICL）和链式思维（CoT）等一系列新兴能力方面的进展。为了更好地理解并控制模型性能，许多研究已经开始探讨这些现象的内在原因及其对任务结果的影响。然而，现有的解释框架大多集中在独立地隔离和解释ICL和CoT，导致对它们联合影响模型性能的理解不够全面。为了解决这一问题，我们提出了电子电路模型（ECM），该模型为开发可扩展的学习策略和改进AI生成内容的管理提供了基础。具体而言，ECM 将模型行为类比为电子电路：ICL 被描绘为语义磁场所提供的额外电压，遵循法拉第定律，而CoT 则被建模为串联电阻，以根据欧姆定律限制模型输出性能。实验结果表明，ECM 有效地预测和解释了各种提示策略下的LLM性能。此外，我们应用ECM 对国际信息学奥林匹克竞赛（IOI）和国际数学奥林匹克竞赛（IMO）等任务的高级推理策略进行优化，实现了能超越近80%顶尖人类竞争对手的竞争力水平。 

---
# Out-of-Distribution Detection using Synthetic Data Generation 

**Title (ZH)**: 使用合成数据生成进行域外检测 

**Authors**: Momin Abbas, Muneeza Azmat, Raya Horesh, Mikhail Yurochkin  

**Link**: [PDF](https://arxiv.org/pdf/2502.03323)  

**Abstract**: Distinguishing in- and out-of-distribution (OOD) inputs is crucial for reliable deployment of classification systems. However, OOD data is typically unavailable or difficult to collect, posing a significant challenge for accurate OOD detection. In this work, we present a method that harnesses the generative capabilities of Large Language Models (LLMs) to create high-quality synthetic OOD proxies, eliminating the dependency on any external OOD data source. We study the efficacy of our method on classical text classification tasks such as toxicity detection and sentiment classification as well as classification tasks arising in LLM development and deployment, such as training a reward model for RLHF and detecting misaligned generations. Extensive experiments on nine InD-OOD dataset pairs and various model sizes show that our approach dramatically lowers false positive rates (achieving a perfect zero in some cases) while maintaining high accuracy on in-distribution tasks, outperforming baseline methods by a significant margin. 

**Abstract (ZH)**: 可靠部署分类系统的关键在于区分分布内（In-Distribution, InD）和分布外（Out-of-Distribution, OOD）输入。然而，OOD数据通常难以获取或难以收集，这为准确的OOD检测带来了重大挑战。在本文中，我们提出了一种方法，利用大型语言模型（LLM）的生成能力创建高质量的合成OOD代理，从而消除对外部OOD数据源的依赖。我们在传统的文本分类任务（如毒性检测和情感分类）以及LLM开发和部署中产生的分类任务（如为RLHF训练奖励模型和检测对齐偏差生成）上研究了该方法的有效性。在九对InD-OOD数据集和各种模型规模的广泛实验中，我们的方法显著降低了假阳性率（在某些情况下实现完美的零假阳性），同时在分布内任务上保持了高精度，超过了基线方法的显著幅度。 

---
# MeDiSumQA: Patient-Oriented Question-Answer Generation from Discharge Letters 

**Title (ZH)**: MeDiSumQA：从出院病历中生成患者导向的问题与答案摘要 

**Authors**: Amin Dada, Osman Alperen Koras, Marie Bauer, Amanda Butler, Kaleb E. Smith, Jens Kleesiek, Julian Friedrich  

**Link**: [PDF](https://arxiv.org/pdf/2502.03298)  

**Abstract**: While increasing patients' access to medical documents improves medical care, this benefit is limited by varying health literacy levels and complex medical terminology. Large language models (LLMs) offer solutions by simplifying medical information. However, evaluating LLMs for safe and patient-friendly text generation is difficult due to the lack of standardized evaluation resources. To fill this gap, we developed MeDiSumQA. MeDiSumQA is a dataset created from MIMIC-IV discharge summaries through an automated pipeline combining LLM-based question-answer generation with manual quality checks. We use this dataset to evaluate various LLMs on patient-oriented question-answering. Our findings reveal that general-purpose LLMs frequently surpass biomedical-adapted models, while automated metrics correlate with human judgment. By releasing MeDiSumQA on PhysioNet, we aim to advance the development of LLMs to enhance patient understanding and ultimately improve care outcomes. 

**Abstract (ZH)**: 尽管增加患者对医疗文件的访问能够改善医疗服务，这一好处受到不同健康素养水平和复杂医学术语的限制。大规模语言模型（LLMs）通过简化医疗信息提供了解决方案。然而，由于缺乏标准化的评估资源，对LLMs进行安全性和患者友好的文本生成评估颇具挑战性。为填补这一空白，我们开发了MeDiSumQA。MeDiSumQA是由MIMIC-IV出院总结通过一个自动化的管道生成的数据集，该管道结合了基于LLM的问题-答案生成和人工质量检查。我们使用该数据集评估各种LLMs在以患者为中心的问题回答方面的性能。我们的研究发现，通用语言模型通常优于医学适应型模型，而自动评估指标与人类判断相关。通过在PhysioNet上公开MeDiSumQA，我们旨在促进LLMs的发展，以增强患者的理解并最终改善护理结果。 

---
# ALPET: Active Few-shot Learning for Citation Worthiness Detection in Low-Resource Wikipedia Languages 

**Title (ZH)**: ALPET：低资源维基语种引用可信度检测的主动少量样本学习方法 

**Authors**: Aida Halitaj, Arkaitz Zubiaga  

**Link**: [PDF](https://arxiv.org/pdf/2502.03292)  

**Abstract**: Citation Worthiness Detection (CWD) consists in determining which sentences, within an article or collection, should be backed up with a citation to validate the information it provides. This study, introduces ALPET, a framework combining Active Learning (AL) and Pattern-Exploiting Training (PET), to enhance CWD for languages with limited data resources. Applied to Catalan, Basque, and Albanian Wikipedia datasets, ALPET outperforms the existing CCW baseline while reducing the amount of labeled data in some cases above 80\%. ALPET's performance plateaus after 300 labeled samples, showing it suitability for low-resource scenarios where large, labeled datasets are not common. While specific active learning query strategies, like those employing K-Means clustering, can offer advantages, their effectiveness is not universal and often yields marginal gains over random sampling, particularly with smaller datasets. This suggests that random sampling, despite its simplicity, remains a strong baseline for CWD in constraint resource environments. Overall, ALPET's ability to achieve high performance with fewer labeled samples makes it a promising tool for enhancing the verifiability of online content in low-resource language settings. 

**Abstract (ZH)**: 引用价值检测（Citation Worthiness Detection, CWD）是指确定文章或集合中的哪些句子需要通过引用来验证所提供的信息。本研究引入了一种结合主动学习（Active Learning, AL）和模式挖掘训练（Pattern-Exploiting Training, PET）的框架（ALPET），以提高对于数据资源有限的语言的CWD性能。ALPET在加泰罗尼亚语、巴斯克语和阿尔巴尼亚语维基百科数据集上的应用表明，它在某些情况下能比现有的CCW基线模型提高性能，同时减少了需要标注的数据量，甚至在某些情况下高出80%以上。ALPET的性能在300个标注样本后趋于稳定，这表明它适用于大型、标注数据集不常见的低资源环境。虽然一些特定的主动学习查询策略，如使用K-Means聚类，可能提供优势，但其有效性并非普遍适用，且在小数据集的情况下往往仅带来微小的改进，这是通过随机抽样实现的。这表明，尽管简单，随机抽样仍然是在受限资源环境中进行CWD的强大基线方法。总体而言，ALPET能够在较少的标注样本下实现高性能，使其成为提高低资源语言环境中在线内容可验证性的有前途的工具。 

---
# Token Assorted: Mixing Latent and Text Tokens for Improved Language Model Reasoning 

**Title (ZH)**: Token Assorted：结合潜在词元和文本词元以提高语言模型推理能力 

**Authors**: DiJia Su, Hanlin Zhu, Yingchen Xu, Jiantao Jiao, Yuandong Tian, Qinqing Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.03275)  

**Abstract**: Large Language Models (LLMs) excel at reasoning and planning when trained on chainof-thought (CoT) data, where the step-by-step thought process is explicitly outlined by text tokens. However, this results in lengthy inputs where many words support textual coherence rather than core reasoning information, and processing these inputs consumes substantial computation resources. In this work, we propose a hybrid representation of the reasoning process, where we partially abstract away the initial reasoning steps using latent discrete tokens generated by VQ-VAE, significantly reducing the length of reasoning traces. We explore the use of latent trace abstractions in two scenarios: 1) training the model from scratch for the Keys-Finding Maze problem, 2) fine-tuning LLMs on this hybrid data with an extended vocabulary including unseen latent tokens, for both logical and mathematical reasoning problems. To facilitate effective learning, we introduce a simple training procedure that randomly mixes latent and text tokens, which enables fast adaptation to new latent tokens. Our approach consistently outperforms the baselines methods in various benchmarks. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在训练于链式思考（CoT）数据集上时，能够很好地进行推理和规划，其中推理过程通过文本标记明确列出。然而，这导致输入变得相当长，其中许多单词增加了文本连贯性而非核心推理信息，处理这些输入消耗了大量的计算资源。在本研究中，我们提出了一种混合表示推理过程的方法，通过部分使用由VQ-VAE生成的潜在离散标记来抽象初始推理步骤，显著减少了推理追踪的长度。我们探讨了潜在追踪抽象在两种场景中的应用：1) 从头开始训练模型解决“寻找钥匙迷宫”问题；2) 在这种混合数据上微调LLMs，包括扩展词汇表，包含看不见的潜在标记，以解决逻辑和数学推理问题。为促进有效的学习，我们引入了一种简单的训练程序，随机混合潜在标记和文本标记，这使得模型能够快速适应新的潜在标记。我们的方法在各种基准测试中始终优于基准方法。 

---
# Efficient extraction of medication information from clinical notes: an evaluation in two languages 

**Title (ZH)**: 从临床记录中高效提取药物信息：两种语言下的评估 

**Authors**: Thibaut Fabacher, Erik-André Sauleau, Emmanuelle Arcay, Bineta Faye, Maxime Alter, Archia Chahard, Nathan Miraillet, Adrien Coulet, Aurélie Névéol  

**Link**: [PDF](https://arxiv.org/pdf/2502.03257)  

**Abstract**: Objective: To evaluate the accuracy, computational cost and portability of a new Natural Language Processing (NLP) method for extracting medication information from clinical narratives. Materials and Methods: We propose an original transformer-based architecture for the extraction of entities and their relations pertaining to patients' medication regimen. First, we used this approach to train and evaluate a model on French clinical notes, using a newly annotated corpus from Hôpitaux Universitaires de Strasbourg. Second, the portability of the approach was assessed by conducting an evaluation on clinical documents in English from the 2018 n2c2 shared task. Information extraction accuracy and computational cost were assessed by comparison with an available method using transformers. Results: The proposed architecture achieves on the task of relation extraction itself performance that are competitive with the state-of-the-art on both French and English (F-measures 0.82 and 0.96 vs 0.81 and 0.95), but reduce the computational cost by 10. End-to-end (Named Entity recognition and Relation Extraction) F1 performance is 0.69 and 0.82 for French and English corpus. Discussion: While an existing system developed for English notes was deployed in a French hospital setting with reasonable effort, we found that an alternative architecture offered end-to-end drug information extraction with comparable extraction performance and lower computational impact for both French and English clinical text processing, respectively. Conclusion: The proposed architecture can be used to extract medication information from clinical text with high performance and low computational cost and consequently suits with usually limited hospital IT resources 

**Abstract (ZH)**: 目标：评估一种新的自然语言处理（NLP）方法的准确性、计算成本和可移植性，该方法用于从临床记录中提取药物信息。
材料与方法：我们提出了一种基于变换器的原创架构，用于提取患者用药方案相关实体及其关系。首先，我们使用这种方法在法国临床记录上训练和评估了一个模型，并利用斯特拉斯堡大学医院新标注的语料库进行训练和评估。第二，我们通过在2018年n2c2共享任务中使用的英语临床文件对方法的可移植性进行了评估。通过与使用变换器的现有方法进行比较，评估了信息提取的准确性及计算成本。结果：所提出的架构在关系提取任务上的性能在法语和英语上的F值分别为0.82和0.96，达到了与最先进的方法相当的水平（0.81和0.95），同时将计算成本降低了10%。端到端（命名实体识别与关系提取）F1性能分别为法语和英语语料库的0.69和0.82。讨论：尽管现有的系统是为英语笔记设计的，并在一台法国医院环境下得以实施，但我们的研究表明，一种替代架构在同一语料库中实现了端到端的药物信息提取，提取性能可与现有系统媲美，且计算成本较低。结论：所提出的架构可以高精度且低计算成本地从临床文本中提取药物信息，因而适合资源有限的医院IT环境。 

---
# How do Humans and Language Models Reason About Creativity? A Comparative Analysis 

**Title (ZH)**: 人类与语言模型在推理创造力方面有何差异？一种比较分析 

**Authors**: Antonio Laverghetta Jr., Tuhin Chakrabarty, Tom Hope, Jimmy Pronchick, Krupa Bhawsar, Roger E. Beaty  

**Link**: [PDF](https://arxiv.org/pdf/2502.03253)  

**Abstract**: Creativity assessment in science and engineering is increasingly based on both human and AI judgment, but the cognitive processes and biases behind these evaluations remain poorly understood. We conducted two experiments examining how including example solutions with ratings impact creativity evaluation, using a finegrained annotation protocol where raters were tasked with explaining their originality scores and rating for the facets of remoteness (whether the response is "far" from everyday ideas), uncommonness (whether the response is rare), and cleverness. In Study 1, we analyzed creativity ratings from 72 experts with formal science or engineering training, comparing those who received example solutions with ratings (example) to those who did not (no example). Computational text analysis revealed that, compared to experts with examples, no-example experts used more comparative language (e.g., "better/worse") and emphasized solution uncommonness, suggesting they may have relied more on memory retrieval for comparisons. In Study 2, parallel analyses with state-of-the-art LLMs revealed that models prioritized uncommonness and remoteness of ideas when rating originality, suggesting an evaluative process rooted around the semantic similarity of ideas. In the example condition, while LLM accuracy in predicting the true originality scores improved, the correlations of remoteness, uncommonness, and cleverness with originality also increased substantially - to upwards of 0.99 - suggesting a homogenization in the LLMs evaluation of the individual facets. These findings highlight important implications for how humans and AI reason about creativity and suggest diverging preferences for what different populations prioritize when rating. 

**Abstract (ZH)**: 科学和工程领域的创造力评估越来越多地依赖于人类和人工智能的判断，但这些评估背后的认知过程和偏见仍知之甚少。我们进行了两项实验，旨在探讨提供例题及其评分如何影响创造力评估，使用了一种精细标注协议，要求评阅人在解释其原创性评分的同时，对响应的远程性（即响应是否“远离”日常生活中的想法）、稀有性（即响应是否罕见）以及巧妙性进行评分。在研究1中，我们分析了72名具有正式科学或工程训练的专家的创造力评分，比较了获得评分例题（例题组）与未获得例题（无例题组）的专家之间的情况。计算文本分析表明，与获得例题的专家相比，无例题的专家使用了更多的比较语言（例如，“更好/更差”），更加强调解决方案的稀有性，表明他们可能更多地依赖记忆检索来进行比较。在研究2中，使用尖端语言模型进行的平行分析表明，这些模型在评估原创性时更倾向于关注想法的独特性和远程性，表明评估过程围绕着想法的语义相似性进行。在有例题的条件下，虽然语言模型预测真实原创性评分的准确性有所提高，但远程性、稀有性和巧妙性与原创性的相关性也显著增加——达到0.99以上，这表明语言模型在评估这些方面时表现出了同质性。这些发现强调了人类和人工智能在处理创造力时的重要含义，并暗示不同群体在评定时偏好的差异。 

---
# A scale of conceptual orality and literacy: Automatic text categorization in the tradition of "N\"ahe und Distanz" 

**Title (ZH)**: 一种概念性口头与文字尺度：基于“近与远”传统的自动文本分类 

**Authors**: Volker Emmrich  

**Link**: [PDF](https://arxiv.org/pdf/2502.03252)  

**Abstract**: Koch and Oesterreicher's model of "Nähe und Distanz" (Nähe = immediacy, conceptual orality; Distanz = distance, conceptual literacy) is constantly used in German linguistics. However, there is no statistical foundation for use in corpus linguistic analyzes, while it is increasingly moving into empirical corpus linguistics. Theoretically, it is stipulated, among other things, that written texts can be rated on a scale of conceptual orality and literacy by linguistic features. This article establishes such a scale based on PCA and combines it with automatic analysis. Two corpora of New High German serve as examples. When evaluating established features, a central finding is that features of conceptual orality and literacy must be distinguished in order to rank texts in a differentiated manner. The scale is also discussed with a view to its use in corpus compilation and as a guide for analyzes in larger corpora. With a theory-driven starting point and as a "tailored" dimension, the approach compared to Biber's Dimension 1 is particularly suitable for these supporting, controlling tasks. 

**Abstract (ZH)**: 科赫和奥斯特雷希的“近与远”模型（Nähe = 立体性或概念口述；Distanz = 距离或概念书写能力）在德国语言学中一直被广泛应用。然而，该模型缺乏统计数据支持其在语料库语言学分析中的应用，而这一领域正逐渐转向实证语料库语言学。理论上，规定了书面文本可以基于语言特征被评定为概念口述或概念书写的程度。本文基于主成分分析（PCA）建立了一个这样的尺度，并将其与自动分析相结合。以两种新的高德语言语料库为例，评估已确立的特征时的一个主要发现是，必须区分概念口述和概念书写的特点，以对文本进行有区别的评价。此外，还讨论了该尺度在语料库编制和在大语料库分析中的指导作用。与以理论为导向的起点和“定制化”的维度相比，该方法特别适用于这些支持性和控制性任务，类似于比伯的维度1。 

---
# Mitigating Language Bias in Cross-Lingual Job Retrieval: A Recruitment Platform Perspective 

**Title (ZH)**: 跨语言职位检索中语言偏见的缓解：从招聘平台的角度 

**Authors**: Napat Laosaengpha, Thanit Tativannarat, Attapol Rutherford, Ekapol Chuangsuwanich  

**Link**: [PDF](https://arxiv.org/pdf/2502.03220)  

**Abstract**: Understanding the textual components of resumes and job postings is critical for improving job-matching accuracy and optimizing job search systems in online recruitment platforms. However, existing works primarily focus on analyzing individual components within this information, requiring multiple specialized tools to analyze each aspect. Such disjointed methods could potentially hinder overall generalizability in recruitment-related text processing. Therefore, we propose a unified sentence encoder that utilized multi-task dual-encoder framework for jointly learning multiple component into the unified sentence encoder. The results show that our method outperforms other state-of-the-art models, despite its smaller model size. Moreover, we propose a novel metric, Language Bias Kullback-Leibler Divergence (LBKL), to evaluate language bias in the encoder, demonstrating significant bias reduction and superior cross-lingual performance. 

**Abstract (ZH)**: 理解简历和职位描述中的文本组成部分对于提高求职匹配准确性和优化在线招聘平台的求职系统至关重要。然而，现有的研究主要侧重于单独分析这些信息中的各个组成部分，这需要使用多个专业工具来分析各个方面。这种离散的方法可能在招聘相关的文本处理中限制整体的通用性。因此，我们提出了一种统一的句子编码器，利用多任务双编码器框架，联合学习多个组成部分。实验结果表明，尽管我们的模型规模较小，但其性能仍优于其他最先进的模型。此外，我们提出了一种新的度量标准——语言偏差Kullback-Leibler散度（LBKL）——来评估编码器中的语言偏差，结果显示这种评价方法能够显著减少偏差并提高跨语言性能。 

---
# iVISPAR -- An Interactive Visual-Spatial Reasoning Benchmark for VLMs 

**Title (ZH)**: iVISPAR -- 一种面向VLMs的交互式视觉-空间推理基准测试 

**Authors**: Julius Mayer, Mohamad Ballout, Serwan Jassim, Farbod Nosrat Nezami, Elia Bruni  

**Link**: [PDF](https://arxiv.org/pdf/2502.03214)  

**Abstract**: Vision-Language Models (VLMs) are known to struggle with spatial reasoning and visual alignment. To help overcome these limitations, we introduce iVISPAR, an interactive multi-modal benchmark designed to evaluate the spatial reasoning capabilities of VLMs acting as agents. iVISPAR is based on a variant of the sliding tile puzzle-a classic problem that demands logical planning, spatial awareness, and multi-step reasoning. The benchmark supports visual 2D, 3D, and text-based input modalities, enabling comprehensive assessments of VLMs' planning and reasoning skills. We evaluate a broad suite of state-of-the-art open-source and closed-source VLMs, comparing their performance while also providing optimal path solutions and a human baseline to assess the task's complexity and feasibility for humans. Results indicate that while some VLMs perform well on simple spatial tasks, they encounter difficulties with more complex configurations and problem properties. Notably, while VLMs generally perform better in 2D vision compared to 3D or text-based representations, they consistently fall short of human performance, illustrating the persistent challenge of visual alignment. This highlights critical gaps in current VLM capabilities, highlighting their limitations in achieving human-level cognition. 

**Abstract (ZH)**: 视觉语言模型（VLMs）已知在空间推理和视觉对齐方面存在困难。为克服这些局限，我们引入了iVISPAR，这是一个交互式的多模态基准，旨在评估VLM作为代理时的空间推理能力。iVISPAR基于滑动拼图问题的一种变体——这是一个经典的逻辑规划、空间意识和多步推理需求的问题。该基准支持视觉2D、3D和基于文本的输入模态，使得对VLM的规划和推理能力进行全面评估成为可能。我们评估了一系列最新的开源和封闭源VLM，对比了它们的表现，并提供了最优路径解决方案和人类基线，以评估任务的复杂性和对人类的可行性。结果显示，尽管有些VLM在简单空间任务上表现良好，但在更复杂的空间配置和问题属性上遇到困难。值得注意的是，虽然VLM在2D视觉上的表现普遍优于3D或基于文本的表示，但在所有情况下，它们都未能达到人类的性能，这突显了视觉对齐的持续挑战。这一结果表明，当前VLM在实现人类级认知方面存在关键差距和限制。

（译者注：学术翻译应尽可能精确传达原文含义，同时符合中文表达习惯。上述翻译在保持原文意思的基础上，进行了适当的技术性措辞调整，以适应学术表述。） 

---
# Improve Decoding Factuality by Token-wise Cross Layer Entropy of Large Language Models 

**Title (ZH)**: 通过词-token层面跨层熵改进大型语言模型的解码事实性 

**Authors**: Jialiang Wu, Yi Shen, Sijia Liu, Yi Tang, Sen Song, Xiaoyi Wang, Longjun Cai  

**Link**: [PDF](https://arxiv.org/pdf/2502.03199)  

**Abstract**: Despite their impressive capacities, Large language models (LLMs) often struggle with the hallucination issue of generating inaccurate or fabricated content even when they possess correct knowledge. In this paper, we extend the exploration of the correlation between hidden-state prediction changes and output factuality into a deeper, token-wise level. Based on the insights , we propose cross-layer Entropy eNhanced Decoding (END), a decoding method that mitigates hallucinations without requiring extra training. END leverages inner probability changes across layers to individually quantify the factual knowledge required for each candidate token, and adjusts the final predicting distribution to prioritize tokens with higher factuality. Experiments on both hallucination and QA benchmarks demonstrate that END significantly enhances the truthfulness and informativeness of generated content while maintaining robust QA accuracy. Moreover, our work provides a deeper perspective on understanding the correlations between inherent knowledge and output factuality. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）具有出色的能力，但在拥有正确知识的情况下，它们往往难以避免生成不准确或虚构内容的幻觉问题。在本文中，我们进一步将隐藏状态预测变化与输出事实性的相关性探索扩展到更深层次的、以令牌为基础的层面。基于这些洞察，我们提出了一种名为跨层熵增强解码（END）的解码方法，该方法能够在不需要额外训练的情况下减轻幻觉问题。END 利用跨层内概率变化，分别量化每个候选令牌所需的事实知识，并调整最终的预测分布，以优先考虑具有更高事实性的令牌。在幻觉和问答基准测试中的实验表明，END 显著提高了生成内容的真实性与信息量，同时保持了稳健的问答准确性。此外，我们的工作为理解固有知识与输出事实性之间的关系提供了更深入的视角。 

---
# Euska\~nolDS: A Naturally Sourced Corpus for Basque-Spanish Code-Switching 

**Title (ZH)**: EuskannolDS：一种自然来源的巴斯克-西班牙语代码切换语料库 

**Authors**: Maite Heredia, Jeremy Barnes, Aitor Soroa  

**Link**: [PDF](https://arxiv.org/pdf/2502.03188)  

**Abstract**: Code-switching (CS) remains a significant challenge in Natural Language Processing (NLP), mainly due a lack of relevant data. In the context of the contact between the Basque and Spanish languages in the north of the Iberian Peninsula, CS frequently occurs in both formal and informal spontaneous interactions. However, resources to analyse this phenomenon and support the development and evaluation of models capable of understanding and generating code-switched language for this language pair are almost non-existent. We introduce a first approach to develop a naturally sourced corpus for Basque-Spanish code-switching. Our methodology consists of identifying CS texts from previously available corpora using language identification models, which are then manually validated to obtain a reliable subset of CS instances. We present the properties of our corpus and make it available under the name EuskañolDS. 

**Abstract (ZH)**: 代码转换（CS）仍然是自然语言处理（NLP）领域的一个重大挑战，主要是由于缺乏相关数据。在伊比利亚半岛北部巴斯克语与西班牙语接触的背景下，CS在正式和非正式的自发交流中经常发生。然而，用于分析这一现象以及支持为此语言对开发和评估能够理解和生成代码混合语言的模型的资源几乎是不存在的。我们介绍了一种开发巴斯克语-西班牙语代码转换自然来源语料库的第一种方法。我们的方法包括使用语言识别模型从现有语料库中识别CS文本，然后手动验证以获得可靠的CS实例集。我们呈现了该语料库的特性，并将其命名为EuskañolDS，以供公众使用。 

---
# Scalable In-Context Learning on Tabular Data via Retrieval-Augmented Large Language Models 

**Title (ZH)**: 通过检索增强大型语言模型实现表格数据的大规模上下文学习 

**Authors**: Xumeng Wen, Shun Zheng, Zhen Xu, Yiming Sun, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2502.03147)  

**Abstract**: Recent studies have shown that large language models (LLMs), when customized with post-training on tabular data, can acquire general tabular in-context learning (TabICL) capabilities. These models are able to transfer effectively across diverse data schemas and different task domains. However, existing LLM-based TabICL approaches are constrained to few-shot scenarios due to the sequence length limitations of LLMs, as tabular instances represented in plain text consume substantial tokens. To address this limitation and enable scalable TabICL for any data size, we propose retrieval-augmented LLMs tailored to tabular data. Our approach incorporates a customized retrieval module, combined with retrieval-guided instruction-tuning for LLMs. This enables LLMs to effectively leverage larger datasets, achieving significantly improved performance across 69 widely recognized datasets and demonstrating promising scaling behavior. Extensive comparisons with state-of-the-art tabular models reveal that, while LLM-based TabICL still lags behind well-tuned numeric models in overall performance, it uncovers powerful algorithms under limited contexts, enhances ensemble diversity, and excels on specific datasets. These unique properties underscore the potential of language as a universal and accessible interface for scalable tabular data learning. 

**Abstract (ZH)**: 近年来的研究表明，通过在表格数据上进行后训练，大型语言模型（LLMs）能够获得一般的表格上下文学习（TabICL）能力。这些模型能够有效地跨越多种数据模式和不同的任务领域进行迁移。然而，现有的基于LLM的TabICL方法由于序列长度的限制，大多局限于少样本场景，因为以文本形式表示的表格实例会消耗大量的令牌。为了解决这一限制，并能够对任意大小的数据进行扩展的TabICL，我们提出了一种针对表格数据的检索增强LLMs方法。该方法结合了一个定制的检索模块，并通过检索指导的指令微调来对LLMs进行优化。这使得LLMs能够充分利用更大规模的数据集，实现了在69个广泛认可的基准数据集上的显著性能提升，并表现出良好的扩展行为。与最先进的表格模型的广泛比较表明，虽然基于LLM的TabICL在整体性能上仍落后于仔细调优的数值模型，但它在有限上下文中揭示了强大的算法，增强了模型多样性，并在特定数据集上表现出色。这些独特的特性突显了语言作为面向大规模表格数据学习的通用且可访问界面的潜力。 

---
# Teaching Large Language Models Number-Focused Headline Generation With Key Element Rationales 

**Title (ZH)**: 用关键元素理由指导大规模语言模型生成以数字为重点的新闻标题 

**Authors**: Zhen Qian, Xiuzhen Zhang, Xiaofei Xu, Feng Xia  

**Link**: [PDF](https://arxiv.org/pdf/2502.03129)  

**Abstract**: Number-focused headline generation is a summarization task requiring both high textual quality and precise numerical accuracy, which poses a unique challenge for Large Language Models (LLMs). Existing studies in the literature focus only on either textual quality or numerical reasoning and thus are inadequate to address this challenge. In this paper, we propose a novel chain-of-thought framework for using rationales comprising key elements of the Topic, Entities, and Numerical reasoning (TEN) in news articles to enhance the capability for LLMs to generate topic-aligned high-quality texts with precise numerical accuracy. Specifically, a teacher LLM is employed to generate TEN rationales as supervision data, which are then used to teach and fine-tune a student LLM. Our approach teaches the student LLM automatic generation of rationales with enhanced capability for numerical reasoning and topic-aligned numerical headline generation. Experiments show that our approach achieves superior performance in both textual quality and numerical accuracy. 

**Abstract (ZH)**: 以下是对给定内容的中文翻译，符合学术规范：

基于数字的标题生成是一项要求极高文本质量和精确数字准确性的摘要任务，这为大型语言模型（LLMs）带来了独特的挑战。现有文献中的研究仅侧重于文本质量或数值推理中的一个方面，因此无法有效应对这一挑战。本文提出了一种新颖的链式思考框架，该框架利用新闻文章中与主题（Topic）、实体（Entities）和数值推理（Numerical reasoning）相关的理由（TEN），增强大型语言模型生成与主题一致、高质量且精确数值的文本的能力。具体而言，使用具有较强数值推理能力的教师语言模型生成TEN理由作为监督数据，然后利用这些数据训练和微调学生模型。我们的方法能够使学生模型自动生成理由，并增强其数值推理能力和主题一致的数值标题生成能力。实验结果表明，我们的方法在文本质量和数值准确性方面均取得了优越表现。 

---
# Policies and Evaluation for Online Meeting Summarization 

**Title (ZH)**: 在线会议总结的政策与评估方法 

**Authors**: Felix Schneider, Marco Turchi, Alex Waibel  

**Link**: [PDF](https://arxiv.org/pdf/2502.03111)  

**Abstract**: With more and more meetings moving to a digital domain, meeting summarization has recently gained interest in both academic and commercial research. However, prior academic research focuses on meeting summarization as an offline task, performed after the meeting concludes. In this paper, we perform the first systematic study of online meeting summarization. For this purpose, we propose several policies for conducting online summarization. We discuss the unique challenges of this task compared to the offline setting and define novel metrics to evaluate latency and partial summary quality. The experiments on the AutoMin dataset show that 1) online models can produce strong summaries, 2) our metrics allow a detailed analysis of different systems' quality-latency trade-off, also taking into account intermediate outputs and 3) adaptive policies perform better than fixed scheduled ones. These findings provide a starting point for the wider research community to explore this important task. 

**Abstract (ZH)**: 随着越来越多的会议转向数字化领域，会议总结近年来在学术界和商业研究中引起了广泛关注。然而，之前的学术研究主要集中在会议总结作为一种离线任务，即在会议结束后进行。本文首次对该领域的在线会议总结进行了系统性的研究。为此，我们提出了几种在线总结政策。我们讨论了与离线环境相比，该任务的独特挑战，并定义了新的指标来评估延迟和部分总结的质量。对AutoMin数据集的实验表明：1）在线模型可以生成强大的总结；2）我们的指标允许对不同系统在质量-延迟权衡方面的详细分析，同时也考虑了中间输出；3）自适应策略比固定时间表策略表现更好。这些发现为更广泛的学术研究界提供了探索这一重要任务的起点。 

---
# Structured Token Retention and Computational Memory Paths in Large Language Models 

**Title (ZH)**: 大规模语言模型中的结构化标记保留与计算记忆路径 

**Authors**: Jonathan Delena, Augustin Moreau, Dominic Ravensdale, Frederick Chatterton  

**Link**: [PDF](https://arxiv.org/pdf/2502.03102)  

**Abstract**: Memory retention mechanisms play a central role in determining the efficiency of computational architectures designed for processing extended sequences. Conventional methods for token management often impose fixed retention thresholds or rely on uniform attention weight distributions, leading to inefficient memory utilization and premature information loss in extended sequence modeling. Structured Token Retention (STR) introduces a probabilistic selection framework that dynamically adjusts token persistence based on contextual significance, ensuring that computational resources are allocated to semantically relevant elements. Computational Memory Paths (CMP) extend this framework through hierarchical memory allocation, refining retention efficiency through structured reallocation of token embeddings. Comparative assessments against baseline models demonstrate that STR and CMP improve token survival rates across long input sequences while reducing cumulative error propagation across processing layers. Experimental results further indicate reductions in computational overhead, improving inference speed without degrading contextual coherence. Token distribution analyses reveal that structured memory allocation prevents excessive redundancy in attention weight calculations, optimizing information retrieval efficiency in large-scale generative architectures. The integration of STR and CMP into an open-source model illustrates the adaptability of structured memory retention methodologies, highlighting their applicability in generative text processing, long-context comprehension, and scalable sequence modeling. 

**Abstract (ZH)**: 记忆保留机制在确定处理延伸序列的计算架构的效率方面发挥着核心作用。传统的 token 管理方法通常会施加固定的保留阈值或依赖于均匀的注意力权重分布，导致在处理延伸序列建模时出现低效的记忆利用和信息过早丢失的问题。结构化 Token 保留 (STR) 引入了一种概率性选择框架，该框架根据上下文的重要性动态调整 Token 的持久性，确保计算资源被分配到语义相关元素上。计算记忆路径 (CMP) 通过层次化记忆分配，进一步增强了保留效率，通过结构化的 Token 嵌入重新分配优化了保留效率。基线模型的对比评估表明，STR 和 CMP 能够在整个输入序列中提高 Token 的生存率，同时减少处理层间的累积误差传播。实验结果还表明，这可以降低计算开销，提高推理速度而不损害上下文一致性。Token 分布分析揭示了结构化记忆分配可以防止注意力权重计算中的过度冗余，从而优化了大规模生成架构的信息检索效率。将 STR 和 CMP 集成到开源模型中展示了结构化记忆保留方法的适应性，突显了它们在生成文本处理、长上下文理解以及可扩展序列建模方面的应用潜力。 

---
# IAO Prompting: Making Knowledge Flow Explicit in LLMs through Structured Reasoning Templates 

**Title (ZH)**: IAO提示：通过结构化推理模板使知识流动在大型语言模型中变得明确 

**Authors**: Aissatou Diallo, Antonis Bikakis, Luke Dickens, Anthony Hunter, Rob Miller  

**Link**: [PDF](https://arxiv.org/pdf/2502.03080)  

**Abstract**: While Large Language Models (LLMs) demonstrate impressive reasoning capabilities, understanding and validating their knowledge utilization remains challenging. Chain-of-thought (CoT) prompting partially addresses this by revealing intermediate reasoning steps, but the knowledge flow and application remain implicit. We introduce IAO (Input-Action-Output) prompting, a structured template-based method that explicitly models how LLMs access and apply their knowledge during complex reasoning tasks. IAO decomposes problems into sequential steps, each clearly identifying the input knowledge being used, the action being performed, and the resulting output. This structured decomposition enables us to trace knowledge flow, verify factual consistency, and identify potential knowledge gaps or misapplications. Through experiments across diverse reasoning tasks, we demonstrate that IAO not only improves zero-shot performance but also provides transparency in how LLMs leverage their stored knowledge. Human evaluation confirms that this structured approach enhances our ability to verify knowledge utilization and detect potential hallucinations or reasoning errors. Our findings provide insights into both knowledge representation within LLMs and methods for more reliable knowledge application. 

**Abstract (ZH)**: 虽然大型语言模型（LLMs）展示了令人印象深刻的推理能力，但理解和验证其知识利用仍然具有挑战性。思维链（CoT）提示部分解决了这一问题，通过揭示中间的推理步骤来展示模型的思维过程，但知识流和应用仍然较为隐含。我们引入了一种名为IAO（输入-动作-输出）提示的方法，这是一种结构化模板方法，明确地建模了LLMs在复杂推理任务中是如何获取和应用知识的。IAO将问题分解为一系列顺序步骤，每个步骤都明确标识出所使用的输入知识、所执行的动作以及由此产生的结果。这种结构化分解使我们能够追踪知识流、验证事实一致性，并识别潜在的知识空白或误用。通过跨各种推理任务的实验，我们证明IAO不仅提高了零样本性能，还提升了我们理解LLMs如何利用其存储知识的透明度。人类评估证实，这种结构化方法增强了我们验证知识利用能力和检测潜在幻觉或推理错误的能力。我们的研究结果为了解LLMs中的知识表示以及更可靠的知识应用方法提供了见解。 

---
# DOLFIN -- Document-Level Financial test set for Machine Translation 

**Title (ZH)**: DOLFIN —— 金融文档级别机器翻译数据集 

**Authors**: Mariam Nakhlé, Marco Dinarelli, Raheel Qader, Emmanuelle Esperança-Rodier, Hervé Blanchon  

**Link**: [PDF](https://arxiv.org/pdf/2502.03053)  

**Abstract**: Despite the strong research interest in document-level Machine Translation (MT), the test sets dedicated to this task are still scarce. The existing test sets mainly cover topics from the general domain and fall short on specialised domains, such as legal and financial. Also, in spite of their document-level aspect, they still follow a sentence-level logic that does not allow for including certain linguistic phenomena such as information reorganisation. In this work, we aim to fill this gap by proposing a novel test set: DOLFIN. The dataset is built from specialised financial documents, and it makes a step towards true document-level MT by abandoning the paradigm of perfectly aligned sentences, presenting data in units of sections rather than sentences. The test set consists of an average of 1950 aligned sections for five language pairs. We present a detailed data collection pipeline that can serve as inspiration for aligning new document-level datasets. We demonstrate the usefulness and quality of this test set by evaluating a number of models. Our results show that the test set is able to discriminate between context-sensitive and context-agnostic models and shows the weaknesses when models fail to accurately translate financial texts. The test set is made public for the community. 

**Abstract (ZH)**: 尽管在文档级别机器翻译（MT）方面的研究兴趣浓厚，但专门为这一任务设计的测试集仍然稀缺。现有的测试集主要涵盖了通用领域的主题，而对于法律和金融等专业领域则有所不足。尽管这些测试集具有文档级别的特点，但它们仍然遵循基于句子级别的逻辑，这并不允许包含某些语言现象，如信息重组。在此项工作中，我们旨在通过提出一个全新的测试集——DOLFIN 来填补这一空白。该数据集是从专业金融文件中构建的，并朝着真正的文档级别 MT 前进了一步，放弃了完美对齐句子的范式，而是以章节为单位呈现数据。测试集包含五个语言对平均约 1950 个对齐的章节。我们提供了一个详细的数据收集管道，可以作为构建新文档级别数据集对齐的参考。我们通过评估多种模型，展示了该测试集的有效性和质量。实验结果表明，该测试集能够区分上下文敏感模型和上下文无关模型，并展示了当模型在准确翻译金融文本方面出现问题时的不足之处。该测试集将向公众开放，供社区使用。 

---
# Knowledge Distillation from Large Language Models for Household Energy Modeling 

**Title (ZH)**: 来自大型语言模型的知识蒸馏在家庭能源建模中的应用 

**Authors**: Mohannad Takrouri, Nicolás M. Cuadrado, Martin Takáč  

**Link**: [PDF](https://arxiv.org/pdf/2502.03034)  

**Abstract**: Machine learning (ML) is increasingly vital for smart-grid research, yet restricted access to realistic, diverse data - often due to privacy concerns - slows progress and fuels doubts within the energy sector about adopting ML-based strategies. We propose integrating Large Language Models (LLMs) in energy modeling to generate realistic, culturally sensitive, and behavior-specific data for household energy usage across diverse geographies. In this study, we employ and compare five different LLMs to systematically produce family structures, weather patterns, and daily consumption profiles for households in six distinct countries. A four-stage methodology synthesizes contextual daily data, including culturally nuanced activities, realistic weather ranges, HVAC operations, and distinct `energy signatures' that capture unique consumption footprints. Additionally, we explore an alternative strategy where external weather datasets can be directly integrated, bypassing intermediate weather modeling stages while ensuring physically consistent data inputs. The resulting dataset provides insights into how cultural, climatic, and behavioral factors converge to shape carbon emissions, offering a cost-effective avenue for scenario-based energy optimization. This approach underscores how prompt engineering, combined with knowledge distillation, can advance sustainable energy research and climate mitigation efforts. Source code is available at this https URL . 

**Abstract (ZH)**: 机器学习（ML）在智能电网研究中的重要性日益增加，但由于隐私问题导致获取现实且多样的数据受限，这阻碍了研究进展，也加剧了能源领域对基于ML策略的怀疑。我们提出在能源建模中整合大型语言模型（LLMs），以生成适用于不同地理区域的家庭能源使用数据，这些数据具有现实性、文化敏感性和行为特定性。在本研究中，我们使用和比较了五种不同的LLMs，系统地生成了六个国家的家庭结构、天气模式和日常消费概况。四种阶段的方法论综合了包括文化细微差异的活动、现实的天气范围、中央空调和供暖系统（HVAC）的操作以及独特“能源签名”，这些特征捕捉了独特的消费足迹。此外，我们探讨了一种替代策略，即直接整合外部天气数据集，可以绕过中间的天气建模阶段，同时确保物理上一致的数据输入。由此产生的数据集提供了文化、气候和行为因素如何共同塑造碳排放的洞察，提供了一种基于情景的能源优化的经济有效途径。这种方法强调了结合提示工程和知识蒸馏如何推动可持续能源研究和气候缓解努力的进步。相关源代码请参见此链接：[链接地址]。 

---
# MedBioLM: Optimizing Medical and Biological QA with Fine-Tuned Large Language Models and Retrieval-Augmented Generation 

**Title (ZH)**: MedBioLM：通过微调大规模语言模型和检索增强生成技术优化医学和生物学问答 

**Authors**: Seonok Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.03004)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities across natural language processing tasks. However, their application to specialized domains such as medicine and biology requires further optimization to ensure factual accuracy, reliability, and contextual depth. We introduce MedBioLM, a domain-adapted biomedical question-answering model designed to enhance both short-form and long-form queries. By integrating fine-tuning and retrieval-augmented generation (RAG), MedBioLM dynamically incorporates domain-specific knowledge, improving reasoning abilities and factual accuracy. To evaluate its effectiveness, we fine-tuned the model on diverse biomedical QA datasets, covering structured multiple-choice assessments and complex clinical reasoning tasks. Fine-tuning significantly improves accuracy on benchmark datasets, while RAG enhances factual consistency. These results highlight the potential of domain-optimized LLMs in advancing biomedical research, medical education, and clinical decision support. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在自然语言处理任务中展现出了令人印象深刻的性能。然而，将其应用于医学和生物学等专门领域需要进一步优化，以确保事实的准确性、可靠性和情境深度。我们介绍了MedBioLM，这是一种专门设计的生物医学问答模型，旨在提高短形式和长形式查询的能力。通过集成微调和检索增强生成（RAG）技术，MedBioLM动态地融入了领域特定的知识，从而提升了推理能力和事实准确性。为了评估其有效性，我们在多种生物医学问答数据集上进行了微调，涵盖了结构化的多项选择评估和复杂的临床推理任务。微调在基准数据集上的准确率显著提高，而RAG则增强了事实的一致性。这些结果突显了优化领域的大规模语言模型在促进生物医学研究、医学教育和临床决策支持方面的潜力。 

---
# Training an LLM-as-a-Judge Model: Pipeline, Insights, and Practical Lessons 

**Title (ZH)**: 训练作为裁判的大型语言模型：流程、见解与实践经验 

**Authors**: Renjun Hu, Yi Cheng, Libin Meng, Jiaxin Xia, Yi Zong, Xing Shi, Wei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2502.02988)  

**Abstract**: The rapid advancement of large language models (LLMs) has opened new possibilities for their adoption as evaluative judges. This paper introduces Themis, a fine-tuned LLM judge that delivers sophisticated context-aware evaluations. We provide a comprehensive overview of the development pipeline for Themis, highlighting its scenario-dependent evaluation prompts and two novel methods for controlled instruction generation. These designs enable Themis to effectively distill evaluative skills from teacher models, while retaining flexibility for continuous development. We introduce two human-labeled benchmarks for meta-evaluation, demonstrating that Themis can achieve high alignment with human preferences in an economical manner. Additionally, we explore insights into the LLM-as-a-judge paradigm, revealing nuances in performance and the varied effects of reference answers. Notably, we observe that pure knowledge distillation from strong LLMs, though common, does not guarantee performance improvement through scaling. We propose a mitigation strategy based on instruction-following difficulty. Furthermore, we provide practical guidelines covering data balancing, prompt customization, multi-objective training, and metric aggregation. We aim for our method and findings, along with the fine-tuning data, benchmarks, and model checkpoints, to support future research and development in this area. 

**Abstract (ZH)**: 大型语言模型（LLMs）的迅速发展为将其作为评估法官的应用开启了新的可能性。本文介绍了Themis，这是一种细调的LLM法官，能够提供复杂的上下文感知评估。我们详细介绍了Themis的开发流程，强调了其场景依赖的评估提示，并介绍了两种新的控制指令生成方法。这些设计使得Themis能够有效地从教师模型中提炼评估技能，同时保留持续开发的灵活性。我们介绍了两个元评估的人工标注基准，展示了Themis能够在经济有效的方式下实现对人类偏好的高度一致。此外，我们探讨了LLM作为法官的范式，揭示了其性能中的复杂性以及参考答案的多样效用。值得注意的是，我们观察到，尽管从强大的LLM中提取纯粹的知识是一种常见做法，但通过扩展并不能保证性能的提升。我们提出了基于指令跟随难度的缓解策略。此外，我们还提供了关于数据平衡、提示定制、多目标训练和指标聚合的实用指南。我们希望我们的方法和发现，包括细调数据、基准和模型检查点，能够支持该领域未来的研究和发展。 

---
# Position: Editing Large Language Models Poses Serious Safety Risks 

**Title (ZH)**: 位置：编辑大型语言模型存在严重的安全风险 

**Authors**: Paul Youssef, Zhixue Zhao, Daniel Braun, Jörg Schlötterer, Christin Seifert  

**Link**: [PDF](https://arxiv.org/pdf/2502.02958)  

**Abstract**: Large Language Models (LLMs) contain large amounts of facts about the world. These facts can become outdated over time, which has led to the development of knowledge editing methods (KEs) that can change specific facts in LLMs with limited side effects. This position paper argues that editing LLMs poses serious safety risks that have been largely overlooked. First, we note the fact that KEs are widely available, computationally inexpensive, highly performant, and stealthy makes them an attractive tool for malicious actors. Second, we discuss malicious use cases of KEs, showing how KEs can be easily adapted for a variety of malicious purposes. Third, we highlight vulnerabilities in the AI ecosystem that allow unrestricted uploading and downloading of updated models without verification. Fourth, we argue that a lack of social and institutional awareness exacerbates this risk, and discuss the implications for different stakeholders. We call on the community to (i) research tamper-resistant models and countermeasures against malicious model editing, and (ii) actively engage in securing the AI ecosystem. 

**Abstract (ZH)**: 大型语言模型（LLMs）包含了大量关于世界的事实。这些事实可能会随着时间的推移变得过时，这导致了知识编辑方法（KEs）的发展，这些方法可以在LLMs中改变特定的事实，并具有有限的副作用。本文观点认为，编辑LLMs带来了严重安全隐患，这一问题并未得到充分关注。首先，我们指出KEs广泛可用、计算成本低廉、性能强大且隐蔽的特点，使它们成为恶意行为者青睐的工具。第二，我们讨论了KEs的恶意使用案例，展示了如何轻松将其改编用于多种恶意目的。第三，我们突出了AI生态系统中的漏洞，这些漏洞允许未经验证就上传和下载更新后的模型。第四，我们指出缺乏社会和机构意识进一步加剧了这一风险，并讨论了不同利益相关者的潜在影响。我们呼吁社区（i）研究防篡改模型和对抗恶意模型编辑的对策，并（ii）积极参与保护AI生态系统的安全。 

---
# ReachAgent: Enhancing Mobile Agent via Page Reaching and Operation 

**Title (ZH)**: ReachAgent：通过页面到达和操作增强移动代理 

**Authors**: Qinzhuo Wu, Wei Liu, Jian Luan, Bin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.02955)  

**Abstract**: Recently, mobile AI agents have gained increasing attention. Given a task, mobile AI agents can interact with mobile devices in multiple steps and finally form a GUI flow that solves the task. However, existing agents tend to focus on most task-relevant elements at each step, leading to local optimal solutions and ignoring the overall GUI flow. To address this issue, we constructed a training dataset called MobileReach, which breaks the task into page reaching and operation subtasks. Furthermore, we propose ReachAgent, a two-stage framework that focuses on improving its task-completion abilities. It utilizes the page reaching and page operation subtasks, along with reward-based preference GUI flows, to further enhance the agent. Experimental results show that ReachAgent significantly improves the IoU Acc and Text Acc by 7.12% and 7.69% on the step-level and 4.72% and 4.63% on the task-level compared to the SOTA agent. Our data and code will be released upon acceptance. 

**Abstract (ZH)**: 近年来，移动AI代理逐渐受到了广泛关注。给定一个任务，移动AI代理可以在多步骤中与移动设备相互作用，并最终形成一个解决该任务的GUI流程。然而，现有的代理往往在每个步骤中专注于与任务最相关的小元素，导致局部最优解，而忽略了整体的GUI流程。为了解决这一问题，我们构建了一个名为MobileReach的数据集，将任务拆分为页面到达和页面操作子任务。此外，我们提出了ReachAgent，这是一种两阶段框架，旨在提高其任务完成能力。该框架利用页面到达和页面操作子任务，以及基于奖励的偏好GUI流程，进一步提升代理性能。实验结果表明，与当前最先进的代理相比，ReachAgent在步骤层面显著提高了IoU Acc和Text Acc，分别提高了7.12%和7.69%，在任务层面分别提高了4.72%和4.63%。我们的数据和代码将在接收后公开。 

---
# LLM-KT: Aligning Large Language Models with Knowledge Tracing using a Plug-and-Play Instruction 

**Title (ZH)**: LLM-KT：通过可插拔指令对大型语言模型与知识追踪进行对齐 

**Authors**: Ziwei Wang, Jie Zhou, Qin Chen, Min Zhang, Bo Jiang, Aimin Zhou, Qinchun Bai, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2502.02945)  

**Abstract**: The knowledge tracing (KT) problem is an extremely important topic in personalized education, which aims to predict whether students can correctly answer the next question based on their past question-answer records. Prior work on this task mainly focused on learning the sequence of behaviors based on the IDs or textual information. However, these studies usually fail to capture students' sufficient behavioral patterns without reasoning with rich world knowledge about questions. In this paper, we propose a large language models (LLMs)-based framework for KT, named \texttt{\textbf{LLM-KT}}, to integrate the strengths of LLMs and traditional sequence interaction models. For task-level alignment, we design Plug-and-Play instruction to align LLMs with KT, leveraging LLMs' rich knowledge and powerful reasoning capacity. For modality-level alignment, we design the plug-in context and sequence to integrate multiple modalities learned by traditional methods. To capture the long context of history records, we present a plug-in context to flexibly insert the compressed context embedding into LLMs using question-specific and concept-specific tokens. Furthermore, we introduce a plug-in sequence to enhance LLMs with sequence interaction behavior representation learned by traditional sequence models using a sequence adapter. Extensive experiments show that \texttt{\textbf{LLM-KT}} obtains state-of-the-art performance on four typical datasets by comparing it with approximately 20 strong baselines. 

**Abstract (ZH)**: 知识追踪（KT）问题是个性化教育中的一个极其重要的研究主题，旨在基于学生以往的问题答题记录预测他们是否能正确回答下一个问题。在此之前，对该任务的研究主要集中在基于问题ID或文本信息学习行为序列上。然而，这些研究通常未能捕捉到学生的行为模式，尤其是在缺乏关于问题的丰富背景知识的情况下。本文提出了一种基于大语言模型（LLMs）的知识追踪框架，命名为\texttt{\textbf{LLM-KT}}，以整合LLMs和传统序列交互模型的优势。为了任务级别对齐，我们设计了一种可插拔指令，利用LLMs丰富的知识和强大的推理能力将LLMs与知识追踪进行对齐。为了模态级别对齐，我们设计了插件上下文和序列以整合传统方法学习的多种模态信息。为了捕捉历史记录中的长上下文，我们提出了一个插件上下文，利用问题特定和概念特定的标记将压缩的上下文嵌入灵活地插入到LLMs中。此外，我们引入了一个插件序列，通过序列适配器增强了LLMs，使其具备传统序列模型学习到的序列交互行为表示。广泛的实验表明，\texttt{\textbf{LLM-KT}}在四个典型数据集上超过了约20种强大的基线方法，获得了最先进的性能。 

---
# LLaVAC: Fine-tuning LLaVA as a Multimodal Sentiment Classifier 

**Title (ZH)**: LLaVAC：将LLaVA微调为多模态情感分类器 

**Authors**: T. Chay-intr, Y. Chen, K. Viriyayudhakorn, T. Theeramunkong  

**Link**: [PDF](https://arxiv.org/pdf/2502.02938)  

**Abstract**: We present LLaVAC, a method for constructing a classifier for multimodal sentiment analysis. This method leverages fine-tuning of the Large Language and Vision Assistant (LLaVA) to predict sentiment labels across both image and text modalities. Our approach involves designing a structured prompt that incorporates both unimodal and multimodal labels to fine-tune LLaVA, enabling it to perform sentiment classification effectively. Experiments on the MVSA-Single dataset demonstrate that LLaVAC outperforms existing methods in multimodal sentiment analysis across three data processing procedures. The implementation of LLaVAC is publicly available at this https URL. 

**Abstract (ZH)**: 我们提出了LLaVAC方法，用于构建一个多模态情感分析分类器。该方法利用了对大型语言和视觉助手（LLaVA）进行微调，以在图像和文本模态中预测情感标签。我们的方法包括设计一个结构化的提示，该提示结合了单模态和多模态标签来微调LLaVA，从而使其能够有效地进行情感分类。在MVSA-Single数据集上的实验表明，LLaVAC在三种数据处理程序下的多模态情感分析中均优于现有方法。LLaVAC的实现已公开发布在以下链接：[](https://example-url.com)（注意：URL需要替换为实际的公开链接地址）。 

---
# What is in a name? Mitigating Name Bias in Text Embeddings via Anonymization 

**Title (ZH)**: 名字背后有什么？通过匿名化减轻文本嵌入中的名称偏差 

**Authors**: Sahil Manchanda, Pannaga Shivaswamy  

**Link**: [PDF](https://arxiv.org/pdf/2502.02903)  

**Abstract**: Text-embedding models often exhibit biases arising from the data on which they are trained. In this paper, we examine a hitherto unexplored bias in text-embeddings: bias arising from the presence of $\textit{names}$ such as persons, locations, organizations etc. in the text. Our study shows how the presence of $\textit{name-bias}$ in text-embedding models can potentially lead to erroneous conclusions in assessment of thematic this http URL-embeddings can mistakenly indicate similarity between texts based on names in the text, even when their actual semantic content has no similarity or indicate dissimilarity simply because of the names in the text even when the texts match semantically. We first demonstrate the presence of name bias in different text-embedding models and then propose $\textit{text-anonymization}$ during inference which involves removing references to names, while preserving the core theme of the text. The efficacy of the anonymization approach is demonstrated on two downstream NLP tasks, achieving significant performance gains. Our simple and training-optimization-free approach offers a practical and easily implementable solution to mitigate name bias. 

**Abstract (ZH)**: 文本嵌入模型通常会表现出由训练数据中产生的偏差。本文我们探讨了一种迄今未被研究的文本嵌入偏差：由于文本中包含人员、地名、组织等名字而导致的偏差。我们的研究显示了文本嵌入模型中名字偏差可能导致对主题评估的错误结论。这种名字偏差可能会导致文本嵌入模型误判两段文本的相似性，尤其是当它们的内容实际上并不相似或相反仅因为文本中的名字相似。我们首先展示了不同文本嵌入模型中名字偏差的存在，然后提出了一种在推理过程中使用文本匿名化的方法，该方法涉及去除对名字的提及，同时保留文本的核心主题。我们通过两个下游自然语言处理任务来展示匿名化方法的有效性，实现了显著的性能提升。我们提出的方法简单且无需训练优化，提供了一种实用且易于实施的减轻名字偏差的方法。 

---
# A Benchmark for the Detection of Metalinguistic Disagreements between LLMs and Knowledge Graphs 

**Title (ZH)**: LLM与知识图谱之间元语言分歧检测基准 

**Authors**: Bradley P. Allen, Paul T. Groth  

**Link**: [PDF](https://arxiv.org/pdf/2502.02896)  

**Abstract**: Evaluating large language models (LLMs) for tasks like fact extraction in support of knowledge graph construction frequently involves computing accuracy metrics using a ground truth benchmark based on a knowledge graph (KG). These evaluations assume that errors represent factual disagreements. However, human discourse frequently features metalinguistic disagreement, where agents differ not on facts but on the meaning of the language used to express them. Given the complexity of natural language processing and generation using LLMs, we ask: do metalinguistic disagreements occur between LLMs and KGs? Based on an investigation using the T-REx knowledge alignment dataset, we hypothesize that metalinguistic disagreement does in fact occur between LLMs and KGs, with potential relevance for the practice of knowledge graph engineering. We propose a benchmark for evaluating the detection of factual and metalinguistic disagreements between LLMs and KGs. An initial proof of concept of such a benchmark is available on Github. 

**Abstract (ZH)**: 在知识图谱（KG）构建支持下的事实抽取等任务中，评估大规模语言模型（LLMs）通常涉及使用基于知识图谱的真实基准来计算准确性指标。这些评估假设错误代表了事实上的分歧。然而，人类对话中经常存在元语言分歧，这些分歧并非基于事实本身，而是基于表达这些事实的语言含义的分歧。鉴于使用LLMs进行自然语言处理和生成的复杂性，我们提出一个问题：LLMs与KGs之间是否也会存在元语言分歧？基于对T-REx知识对齐数据集的调查，我们假设LLMs与KGs之间确实存在元语言分歧，并且这可能对知识图谱工程实践具有重要意义。我们提出了一种用于评估LLMs与KGs之间事实和元语言分歧检测的基准。关于该基准的初步概念已在Github上提供。 

---
# Lowering the Barrier of Machine Learning: Achieving Zero Manual Labeling in Review Classification Using LLMs 

**Title (ZH)**: 降低机器学习的门槛：利用大语言模型实现评论分类中的零人工标注 

**Authors**: Yejian Zhang, Shingo Takada  

**Link**: [PDF](https://arxiv.org/pdf/2502.02893)  

**Abstract**: With the internet's evolution, consumers increasingly rely on online reviews for service or product choices, necessitating that businesses analyze extensive customer feedback to enhance their offerings. While machine learning-based sentiment classification shows promise in this realm, its technical complexity often bars small businesses and individuals from leveraging such advancements, which may end up making the competitive gap between small and large businesses even bigger in terms of improving customer satisfaction. This paper introduces an approach that integrates large language models (LLMs), specifically Generative Pre-trained Transformer (GPT) and Bidirectional Encoder Representations from Transformers (BERT)-based models, making it accessible to a wider audience. Our experiments across various datasets confirm that our approach retains high classification accuracy without the need for manual labeling, expert knowledge in tuning and data annotation, or substantial computational power. By significantly lowering the barriers to applying sentiment classification techniques, our methodology enhances competitiveness and paves the way for making machine learning technology accessible to a broader audience. 

**Abstract (ZH)**: 随着互联网的发展，消费者越来越多地依赖在线评价来选择服务或产品，这促使企业需要分析大量的客户反馈以提升其产品和服务。虽然基于机器学习的情绪分类在这一领域展现出巨大的潜力，但由于技术复杂性，小型企业和个人往往无法利用这些进展。这可能会加剧小型企业和大型企业在提升客户满意度方面的竞争差距。本文提出了一种方法，该方法整合了大型语言模型（LLMs），具体包括生成型预训练Transformer（GPT）和双向编码表示Transformer（BERT）模型，使其能够更加广泛地应用。我们对多个数据集的实验结果表明，该方法在保持高分类准确性的前提下，无需人工标注、专家调优和数据注释，也无需大量计算资源。通过显著降低应用情绪分类技术的门槛，我们的方法提高了竞争力，并为使机器学习技术更广泛地应用铺平了道路。 

---
# Achieving Operational Universality through a Turing Complete Chemputer 

**Title (ZH)**: 通过图灵完备的化学计算机实现操作通用性 

**Authors**: Daniel Gahler, Dean Thomas, Slawomir Lach, Leroy Cronin  

**Link**: [PDF](https://arxiv.org/pdf/2502.02872)  

**Abstract**: The most fundamental abstraction underlying all modern computers is the Turing Machine, that is if any modern computer can simulate a Turing Machine, an equivalence which is called Turing completeness, it is theoretically possible to achieve any task that can be algorithmically described by executing a series of discrete unit operations. In chemistry, the ability to program chemical processes is demanding because it is hard to ensure that the process can be understood at a high level of abstraction, and then reduced to practice. Herein we exploit the concept of Turing completeness applied to robotic platforms for chemistry that can be used to synthesise complex molecules through unit operations that execute chemical processes using a chemically-aware programming language, XDL. We leverage the concept of computability by computers to synthesizability of chemical compounds by automated synthesis machines. The results of an interactive demonstration of Turing completeness using the colour gamut and conditional logic are presented and examples of chemical use-cases are discussed. Over 16.7 million combinations of Red, Green, Blue (RGB) colour space were binned into 5 discrete values and measured over 10 regions of interest (ROIs), affording 78 million possible states per step and served as a proxy for conceptual, chemical space exploration. This formal description establishes a formal framework in future chemical programming languages to ensure complex logic operations are expressed and executed correctly, with the possibility of error correction, in the automated and autonomous pursuit of increasingly complex molecules. 

**Abstract (ZH)**: 所有现代计算机最根本的抽象是图灵机。如果任何现代计算机能够模拟图灵机，并且这种等价性被称为图灵完备性，那么理论上就可以通过执行一系列离散单元操作来实现任何可算法描述的任务。在化学领域，确保能够编程化学过程的需求尤为迫切，因为很难在高层次上理解这些过程，然后将其转化为实践。在此，我们将图灵完备性应用于化学机器人平台，这些平台能够通过使用化学感知编程语言XDL来执行化学过程，从而合成复杂分子。我们利用计算机的计算能力来实现化学化合物的自动合成。展示了使用颜色色域和条件逻辑的交互演示结果，并讨论了化学应用案例。通过将1670多万种红、绿、蓝（RGB）颜色空间组合归类为5个离散值，测量10个感兴趣的区域（ROIs），每个步骤提供了7.8亿种可能状态，作为概念化学空间探索的代理。此种正式描述为未来的化学编程语言提供了一个正式框架，确保复杂的逻辑操作能够被正确地表达和执行，具有错误校正的可能性，在自动和自主追求日益复杂的分子过程中变得越来越重要。 

---
# Position: Multimodal Large Language Models Can Significantly Advance Scientific Reasoning 

**Title (ZH)**: 位置：多模态大型语言模型可以显著推进科学推理 

**Authors**: Yibo Yan, Shen Wang, Jiahao Huo, Jingheng Ye, Zhendong Chu, Xuming Hu, Philip S. Yu, Carla Gomes, Bart Selman, Qingsong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2502.02871)  

**Abstract**: Scientific reasoning, the process through which humans apply logic, evidence, and critical thinking to explore and interpret scientific phenomena, is essential in advancing knowledge reasoning across diverse fields. However, despite significant progress, current scientific reasoning models still struggle with generalization across domains and often fall short of multimodal perception. Multimodal Large Language Models (MLLMs), which integrate text, images, and other modalities, present an exciting opportunity to overcome these limitations and enhance scientific reasoning. Therefore, this position paper argues that MLLMs can significantly advance scientific reasoning across disciplines such as mathematics, physics, chemistry, and biology. First, we propose a four-stage research roadmap of scientific reasoning capabilities, and highlight the current state of MLLM applications in scientific reasoning, noting their ability to integrate and reason over diverse data types. Second, we summarize the key challenges that remain obstacles to achieving MLLM's full potential. To address these challenges, we propose actionable insights and suggestions for the future. Overall, our work offers a novel perspective on MLLM integration with scientific reasoning, providing the LLM community with a valuable vision for achieving Artificial General Intelligence (AGI). 

**Abstract (ZH)**: 科学推理是人类运用逻辑、证据和批判性思维探索和解释科学现象的过程，对于跨学科知识推理的发展至关重要。尽管已取得显著进展，现有科学推理模型仍然在领域间的泛化以及多模态感知方面存在局限。多模态大型语言模型（MLLMs），通过集成文本、图像和其他模态信息，为克服这些局限和提升科学推理提供了令人兴奋的机遇。因此，本文立场认为MLLMs可以在数学、物理学、化学和生物学等学科中显著推进科学推理。首先，我们提出了科学推理能力的四阶段研究路线图，并强调了当前MLLM在科学推理中的应用状态，指出其在整合和处理多种数据类型方面的优势。其次，我们总结了仍然阻碍MLLM充分发挥潜力的关键挑战，并提出了解决这些挑战的具体建议。总体而言，我们的工作为MLLM与科学推理集成提供了一个新颖视角，为大语言模型（LLM）社区提供了实现通用人工智能（AGI）的宝贵愿景。 

---
# CAMI: A Counselor Agent Supporting Motivational Interviewing through State Inference and Topic Exploration 

**Title (ZH)**: CAMI：一种通过状态推断和主题探索支持动机访谈的咨询代理 

**Authors**: Yizhe Yang, Palakorn Achananuparp, Heyan Huang, Jing Jiang, Kit Phey Leng, Nicholas Gabriel Lim, Cameron Tan Shi Ern, Ee-peng Lim  

**Link**: [PDF](https://arxiv.org/pdf/2502.02807)  

**Abstract**: Conversational counselor agents have become essential tools for addressing the rising demand for scalable and accessible mental health support. This paper introduces CAMI, a novel automated counselor agent grounded in Motivational Interviewing (MI) -- a client-centered counseling approach designed to address ambivalence and facilitate behavior change. CAMI employs a novel STAR framework, consisting of client's state inference, motivation topic exploration, and response generation modules, leveraging large language models (LLMs). These components work together to evoke change talk, aligning with MI principles and improving counseling outcomes for clients from diverse backgrounds. We evaluate CAMI's performance through both automated and manual evaluations, utilizing simulated clients to assess MI skill competency, client's state inference accuracy, topic exploration proficiency, and overall counseling success. Results show that CAMI not only outperforms several state-of-the-art methods but also shows more realistic counselor-like behavior. Additionally, our ablation study underscores the critical roles of state inference and topic exploration in achieving this performance. 

**Abstract (ZH)**: 会话咨询代理已成为应对不断增长的可扩展和易获取心理健康支持需求的重要工具。本文介绍了CAMI，这是一种基于动机访谈（MI）的新型自动化咨询代理——动机访谈是一种以客户为中心的咨询方法，旨在解决客户的犹疑和促进行为改变。CAMI采用了一种创新的STAR框架，包括客户端状态推理、动机话题探索和响应生成模块，利用大规模语言模型（LLM）。这些组件共同作用以诱发改变对话，符合动机访谈的原则，并改善来自不同背景客户的咨询服务效果。我们通过自动评估和人工评估来评估CAMI的性能，利用模拟客户评估其动机访谈技能、客户状态推理准确性、话题探索能力以及整体咨询成效。研究结果显示，CAMI不仅优于多种最先进的方法，还表现出更接近人类咨询师的行为。此外，我们的消融研究进一步强调了状态推理和话题探索在实现这一性能中的关键作用。 

---
# Consistent Client Simulation for Motivational Interviewing-based Counseling 

**Title (ZH)**: 基于动机访谈的辅导中一致的客户端模拟研究 

**Authors**: Yizhe Yang, Palakorn Achananuparp, Heyan Huang, Jing Jiang, John Pinto, Jenny Giam, Kit Phey Leng, Nicholas Gabriel Lim, Cameron Tan Shi Ern, Ee-peng Lim  

**Link**: [PDF](https://arxiv.org/pdf/2502.02802)  

**Abstract**: Simulating human clients in mental health counseling is crucial for training and evaluating counselors (both human or simulated) in a scalable manner. Nevertheless, past research on client simulation did not focus on complex conversation tasks such as mental health counseling. In these tasks, the challenge is to ensure that the client's actions (i.e., interactions with the counselor) are consistent with with its stipulated profiles and negative behavior settings. In this paper, we propose a novel framework that supports consistent client simulation for mental health counseling. Our framework tracks the mental state of a simulated client, controls its state transitions, and generates for each state behaviors consistent with the client's motivation, beliefs, preferred plan to change, and receptivity. By varying the client profile and receptivity, we demonstrate that consistent simulated clients for different counseling scenarios can be effectively created. Both our automatic and expert evaluations on the generated counseling sessions also show that our client simulation method achieves higher consistency than previous methods. 

**Abstract (ZH)**: 模拟心理健康咨询中的人类用户对于以可扩展的方式训练和评估辅导员（无论是真人还是模拟人）至关重要。然而，以往关于客户模拟的研究并没有关注像心理健康咨询这样复杂的对话任务。在这些任务中，挑战在于确保模拟客户的行动（即与辅导员的互动）与其规定的人格特征和不良行为设定一致。在本文中，我们提出了一种新的框架，用于支持心理健康咨询中的一致客户模拟。我们的框架跟踪模拟客户的心理状态，控制其状态转换，并为每个状态生成与客户动机、信念、偏爱的改变计划以及接受性一致的行为。通过改变客户的人格特征和接受性，我们证明可以有效地创建适用于不同咨询场景的一致模拟客户。我们的自动评估和专家对生成的咨询会话的评估表明，我们的客户模拟方法在一致性方面优于以往的方法。 

---
# Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation 

**Title (ZH)**: 推测性预填充：通过轻量级且无训练的 token 重要性估计加速 TTFT 

**Authors**: Jingyu Liu, Beidi Chen, Ce Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.02789)  

**Abstract**: Improving time-to-first-token (TTFT) is an essentially important objective in modern large language model (LLM) inference engines. Because optimizing TTFT directly results in higher maximal QPS and meets the requirements of many critical applications. However, boosting TTFT is notoriously challenging since it is purely compute-bounded and the performance bottleneck shifts from the self-attention to the MLP part. We present SpecPrefill, a training free framework that accelerates the inference TTFT for both long and medium context queries based on the following insight: LLMs are generalized enough to still preserve the quality given only a carefully chosen subset of prompt tokens. At its core, SpecPrefill leverages a lightweight model to speculate locally important tokens based on the context. These tokens, along with the necessary positional information, are then sent to the main model for processing. We evaluate SpecPrefill with a diverse set of tasks, followed by a comprehensive benchmarking of performance improvement both in a real end-to-end setting and ablation studies. SpecPrefill manages to serve Llama-3.1-405B-Instruct-FP8 with up to $7\times$ maximal end-to-end QPS on real downstream tasks and $7.66\times$ TTFT improvement during benchmarking. 

**Abstract (ZH)**: 提高第一个标记时间（TTFT）是现代大型语言模型（LLM）推理引擎中的一个基本重要目标。因为直接优化TTFT可以直接提高最大QPS（每秒查询数），并满足许多关键应用的需求。然而，提升TTFT异常具有挑战性，因为它是完全由计算能力限制的，性能瓶颈从自我注意部分转移到MLP部分。我们提出了一种名为SpecPrefill的训练免费框架，该框架基于以下洞察：即使仅基于精心选择的部分提示标记，LLM仍然能够保持质量。其核心在于，SpecPrefill利用一个轻量级模型根据上下文推测出局部重要的标记。这些标记连同必要的位置信息，随后被发送到主模型进行处理。我们使用一系列多样化的任务对SpecPrefill进行了评估，并在实际端到端设置和消融研究中进行了综合性能基准测试。SpecPrefill能够在实际下游任务中为Llama-3.1-405B-Instruct-FP8提供高达7倍的最大端到端QPS，并在基准测试中实现7.66倍的TTFT提速。 

---
# SimMark: A Robust Sentence-Level Similarity-Based Watermarking Algorithm for Large Language Models 

**Title (ZH)**: SimMark：一种基于句级相似性的鲁棒水印算法用于大型语言模型 

**Authors**: Amirhossein Dabiriaghdam, Lele Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.02787)  

**Abstract**: The rapid proliferation of large language models (LLMs) has created an urgent need for reliable methods to detect whether a text is generated by such models. In this paper, we propose SimMark, a posthoc watermarking algorithm that makes LLMs' outputs traceable without requiring access to the model's internal logits, enabling compatibility with a wide range of LLMs, including API-only models. By leveraging the similarity of semantic sentence embeddings and rejection sampling to impose detectable statistical patterns imperceptible to humans, and employing a soft counting mechanism, SimMark achieves robustness against paraphrasing attacks. Experimental results demonstrate that SimMark sets a new benchmark for robust watermarking of LLM-generated content, surpassing prior sentence-level watermarking techniques in robustness, sampling efficiency, and applicability across diverse domains, all while preserving the text quality. 

**Abstract (ZH)**: 大型语言模型（LLMs）的快速 proliferate 已经迫切需要可靠的方法来检测文本是否由这些模型生成。本文我们提出 SimMark，一种后处理水印算法，可以在不访问模型内部 logits 的情况下使 LLM 的输出变得可追溯，从而与各种广泛使用的 LLM 兼容，包括仅通过 API 的模型。通过利用语义句子嵌入的相似性和拒绝采样的方法来施加可被检测但对人类不可感知的统计模式，并采用软计数机制，SimMark 能够抵抗改写攻击。实验结果表明，SimMark 在鲁棒性、采样效率和跨多种领域的适用性方面为 LLM 生成内容的鲁棒水印设定了新的标准，同时优于先前的句子级水印技术，并且不会损害文本质量。 

---
# SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model 

**Title (ZH)**: SmolLM2：从小到大的转变——数据为中心的小型语言模型训练 

**Authors**: Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martín Blázquez, Guilherme Penedo, Lewis Tunstall, Andrés Marafioti, Hynek Kydlíček, Agustín Piqueres Lajarín, Vaibhav Srivastav, Joshua Lochner, Caleb Fahlgren, Xuan-Son Nguyen, Clémentine Fourrier, Ben Burtenshaw, Hugo Larcher, Haojun Zhao, Cyril Zakka, Mathieu Morlon, Colin Raffel, Leandro von Werra, Thomas Wolf  

**Link**: [PDF](https://arxiv.org/pdf/2502.02737)  

**Abstract**: While large language models have facilitated breakthroughs in many applications of artificial intelligence, their inherent largeness makes them computationally expensive and challenging to deploy in resource-constrained settings. In this paper, we document the development of SmolLM2, a state-of-the-art "small" (1.7 billion parameter) language model (LM). To attain strong performance, we overtrain SmolLM2 on ~11 trillion tokens of data using a multi-stage training process that mixes web text with specialized math, code, and instruction-following data. We additionally introduce new specialized datasets (FineMath, Stack-Edu, and SmolTalk) at stages where we found existing datasets to be problematically small or low-quality. To inform our design decisions, we perform both small-scale ablations as well as a manual refinement process that updates the dataset mixing rates at each stage based on the performance at the previous stage. Ultimately, we demonstrate that SmolLM2 outperforms other recent small LMs including Qwen2.5-1.5B and Llama3.2-1B. To facilitate future research on LM development as well as applications of small LMs, we release both SmolLM2 as well as all of the datasets we prepared in the course of this project. 

**Abstract (ZH)**: 尽管大型语言模型在许多人工智能应用中促进了突破性进展，但它们固有的庞大性使得它们在资源受限的环境中计算成本高昂且难以部署。本文我们记录了SmolLM2的发展，SmolLM2是一款最先进的“小”规模（17亿参数）语言模型（LM）。为了获得强大的性能，我们使用多阶段训练过程对SmolLM2进行了过训练，该过程将网络文本与专门的数学、代码和指令遵循数据混合在一起，使用了大约11万亿个数据标记。此外，在我们发现现有数据集在一些阶段过小或质量较低时，我们引入了新的专门数据集（FineMath、Stack-Edu和SmolTalk）。为了指导我们的设计决策，我们进行了小型的消融实验，还进行了一种手动调整过程，根据上一阶段的表现调整每个阶段的数据集混合比例。最终，我们证明了SmolLM2在与其他最近的小型语言模型（如Qwen2.5-1.5B和Llama3.2-1B）的比较中表现更优。为了促进未来对语言模型开发的研究以及小型语言模型的应用，我们不仅发布了SmolLM2，还发布了本项目过程中准备的所有数据集。 

---
# Cross-Lingual Transfer for Low-Resource Natural Language Processing 

**Title (ZH)**: 低资源自然语言处理中的跨语言转移方法 

**Authors**: Iker García-Ferrero  

**Link**: [PDF](https://arxiv.org/pdf/2502.02722)  

**Abstract**: Natural Language Processing (NLP) has seen remarkable advances in recent years, particularly with the emergence of Large Language Models that have achieved unprecedented performance across many tasks. However, these developments have mainly benefited a small number of high-resource languages such as English. The majority of languages still face significant challenges due to the scarcity of training data and computational resources. To address this issue, this thesis focuses on cross-lingual transfer learning, a research area aimed at leveraging data and models from high-resource languages to improve NLP performance for low-resource languages. Specifically, we focus on Sequence Labeling tasks such as Named Entity Recognition, Opinion Target Extraction, and Argument Mining.
The research is structured around three main objectives: (1) advancing data-based cross-lingual transfer learning methods through improved translation and annotation projection techniques, (2) developing enhanced model-based transfer learning approaches utilizing state-of-the-art multilingual models, and (3) applying these methods to real-world problems while creating open-source resources that facilitate future research in low-resource NLP.
More specifically, this thesis presents a new method to improve data-based transfer with T-Projection, a state-of-the-art annotation projection method that leverages text-to-text multilingual models and machine translation systems. T-Projection significantly outperforms previous annotation projection methods by a wide margin. For model-based transfer, we introduce a constrained decoding algorithm that enhances cross-lingual Sequence Labeling in zero-shot settings using text-to-text models. Finally, we develop Medical mT5, the first multilingual text-to-text medical model, demonstrating the practical impact of our research on real-world applications. 

**Abstract (ZH)**: 自然语言处理（NLP）近年来取得了显著的进步，尤其是在大型语言模型的出现下，这些模型在许多任务上都取得了前所未有的性能。然而，这些进展主要惠及了诸如英语等高资源语言。大多数语言仍然面临着严重挑战，原因在于缺少训练数据和计算资源。为了解决这一问题，本论文专注于跨语言迁移学习这一研究领域，旨在利用高资源语言的数据和模型来提高低资源语言的NLP性能。具体而言，我们重点关注序列标注任务，如命名实体识别、意见目标提取和论据挖掘。

研究围绕三个主要目标进行：（1）通过改进翻译和注释投影技术来推进基于数据的跨语言迁移学习方法；（2）利用最先进的多语言模型开发增强的基于模型的迁移学习方法；（3）将这些方法应用于实际问题，并创建开源资源，以促进低资源NLP领域的未来研究。

具体而言，本论文提出了一种新方法来改进基于数据的迁移学习，即T-Projection，这是一种最先进的注释投影方法，利用了文本到文本多语言模型和机器翻译系统。T-Projection在多种应用场景中显著优于之前的注释投影方法。在基于模型的迁移学习方面，我们引入了一种约束解码算法，该算法在零样本设置下利用文本到文本模型增强了跨语言序列标注。最后，我们开发了Medical mT5，这是第一个多语言文本到文本医学模型，展示了我们研究在实际应用中的具体影响。 

---
# Developing multilingual speech synthesis system for Ojibwe, Mi'kmaq, and Maliseet 

**Title (ZH)**: 开发奥吉贝韦、密克马克和马利希特多语言语音合成系统 

**Authors**: Shenran Wang, Changbing Yang, Mike Parkhill, Chad Quinn, Christopher Hammerly, Jian Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2502.02703)  

**Abstract**: We present lightweight flow matching multilingual text-to-speech (TTS) systems for Ojibwe, Mi'kmaq, and Maliseet, three Indigenous languages in North America. Our results show that training a multilingual TTS model on three typologically similar languages can improve the performance over monolingual models, especially when data are scarce. Attention-free architectures are highly competitive with self-attention architecture with higher memory efficiency. Our research not only advances technical development for the revitalization of low-resource languages but also highlights the cultural gap in human evaluation protocols, calling for a more community-centered approach to human evaluation. 

**Abstract (ZH)**: 我们介绍了适用于奥吉布威语、米克马克语和梅利塞特语三种北美原住民语言的轻量级流匹配多语言文本到语音（TTS）系统。我们的研究结果显示，通过在三种类型学相似的语言上训练多语言TTS模型，可以在数据稀缺的情况下提高模型性能，尤其是在单一语言模型之上。无注意力架构在记忆效率方面与自我注意力架构具有竞争力。我们的研究不仅推动了低资源语言 revitalization 的技术发展，还突显了人类评估协议中的文化差距，呼吁采取更以社区为中心的人类评估方法。 

---
# How Inclusively do LMs Perceive Social and Moral Norms? 

**Title (ZH)**: 语言模型如何全面地感知社会和道德规范？ 

**Authors**: Michael Galarnyk, Agam Shah, Dipanwita Guhathakurta, Poojitha Nandigam, Sudheer Chava  

**Link**: [PDF](https://arxiv.org/pdf/2502.02696)  

**Abstract**: This paper discusses and contains offensive content. Language models (LMs) are used in decision-making systems and as interactive assistants. However, how well do these models making judgements align with the diversity of human values, particularly regarding social and moral norms? In this work, we investigate how inclusively LMs perceive norms across demographic groups (e.g., gender, age, and income). We prompt 11 LMs on rules-of-thumb (RoTs) and compare their outputs with the existing responses of 100 human annotators. We introduce the Absolute Distance Alignment Metric (ADA-Met) to quantify alignment on ordinal questions. We find notable disparities in LM responses, with younger, higher-income groups showing closer alignment, raising concerns about the representation of marginalized perspectives. Our findings highlight the importance of further efforts to make LMs more inclusive of diverse human values. The code and prompts are available on GitHub under the CC BY-NC 4.0 license. 

**Abstract (ZH)**: 本论文讨论并包含了一些敏感内容。语言模型（LMs）被应用于决策系统和交互式助手中。然而，这些模型在进行判断时是否与人类价值观的多样性一致，尤其是在社会和道德规范方面，其程度如何？在这项工作中，我们研究了LMs在不同人群（如性别、年龄和收入）中的包容性感知规范。我们对11种LMs提出了一些经验法则（RoTs，并将其输出与100名人工注释员的现有回答进行了比较。我们引入了绝对距离一致性度量（ADA-Met）来量化序数问题的一致性。我们的研究发现，不同群体之间LMs的回答存在显著差异，年轻、高收入群体的答案更为接近，这引起了人们对边缘化视角代表性不足的担忧。研究结果强调了进一步努力使LMs更加包容多样人类价值观的重要性。该代码和提示可在GitHub上获取，采用CC BY-NC 4.0许可。 

---
# Transformers Boost the Performance of Decision Trees on Tabular Data across Sample Sizes 

**Title (ZH)**: Transformer模型在不同样本规模下提升了表格数据上决策树的性能 

**Authors**: Mayuka Jayawardhana, Renbo Tu, Samuel Dooley, Valeriia Cherepanova, Andrew Gordon Wilson, Frank Hutter, Colin White, Tom Goldstein, Micah Goldblum  

**Link**: [PDF](https://arxiv.org/pdf/2502.02672)  

**Abstract**: Large language models (LLMs) perform remarkably well on tabular datasets in zero- and few-shot settings, since they can extract meaning from natural language column headers that describe features and labels. Similarly, TabPFN, a recent non-LLM transformer pretrained on numerous tables for in-context learning, has demonstrated excellent performance for dataset sizes up to a thousand samples. In contrast, gradient-boosted decision trees (GBDTs) are typically trained from scratch on each dataset without benefiting from pretraining data and must learn the relationships between columns from their entries alone since they lack natural language understanding. LLMs and TabPFN excel on small tabular datasets where a strong prior is essential, yet they are not competitive with GBDTs on medium or large datasets, since their context lengths are limited. In this paper, we propose a simple and lightweight approach for fusing large language models and TabPFN with gradient-boosted decision trees, which allows scalable GBDTs to benefit from the natural language capabilities and pretraining of transformers. We name our fusion methods LLM-Boost and PFN-Boost, respectively. While matching or surpassing the performance of the transformer at sufficiently small dataset sizes and GBDTs at sufficiently large sizes, LLM-Boost and PFN-Boost outperform both standalone components on a wide range of dataset sizes in between. We demonstrate state-of-the-art performance against numerous baselines and ensembling algorithms. We find that PFN-Boost achieves the best average performance among all methods we test for all but very small dataset sizes. We release our code at this http URL . 

**Abstract (ZH)**: 以下是经过学术规范翻译后的论文内容或标题：

大规模语言模型（LLMs）在零样本和少样本设置中对表格数据集表现出色，因为它们可以从描述特征和标签的自然语言列头中提取意义。类似地，TabPFN 是一种近期的非 LLM 转换器，在预训练了大量的表格数据后，对于样本量多达一千个的数据集展现了优秀的性能。相比之下，梯度提升决策树（GBDTs）通常需要从头开始训练每个数据集，不能从预训练数据中受益，并且由于缺乏自然语言理解能力，只能通过学习表格项之间的关系来学习列之间的关系。LLMs 和 TabPFN 在小的表格数据集上表现出色，这些数据集需要强烈先验知识，但当数据集规模中等或较大时，它们在与 GBDTs 的竞争中并不具备优势，因为它们的上下文长度是有限的。在本文中，我们提出了一种简单而轻量级的方法，用于将大型语言模型、TabPFN 与梯度提升决策树融合，从而使可扩展的 GBDTs 能够利用转换器的自然语言能力和预训练。我们分别将这两种融合方法命名为 LLM-Boost 和 PFN-Boost。在足够小的数据集规模下，LLM-Boost 和 PFN-Boost 的性能与转换器相当或超过转换器；在足够大的数据集规模下，其性能与 GBDTs 相当或超过 GBDTs。在各种规模的数据集上，LLM-Boost 和 PFN-Boost 在大多数情况下均优于各自的独立组件。我们与多个基准和集成算法进行了对比实验，并展示了最先进的性能表现。我们发现，在除非常小的数据集规模之外的所有测试方法中，PFN-Boost 在所有方法中表现最佳。我们已将我们的代码发布在以下网址：[链接]。 

---
# A Training-Free Length Extrapolation Approach for LLMs: Greedy Attention Logit Interpolation (GALI) 

**Title (ZH)**: 无需训练的长度外推方法：贪婪注意力分数插值（GALI） 

**Authors**: Yan Li, Tianyi Zhang, Zechuan Li, Soyeon Caren Han  

**Link**: [PDF](https://arxiv.org/pdf/2502.02659)  

**Abstract**: Transformer-based Large Language Models (LLMs) struggle to process inputs exceeding their training context window, with performance degrading due to positional out-of-distribution (O.O.D.) that disrupt attention computations. Existing solutions, fine-tuning and training-free methods, are limited by computational inefficiency, attention logit outliers or loss of local positional information. To address this, we propose Greedy Attention Logit Interpolation (GALI), a training-free length extrapolation method that maximizes the utilization of pretrained positional intervals while avoiding attention logit outliers through attention logit interpolation. The result demonstrates that GALI consistently outperforms state-of-the-art training-free methods. Our findings reveal that LLMs interpret positional intervals unevenly within their training context window, suggesting that extrapolating within a smaller positional interval range yields superior results-even for short-context tasks. GALI represents a significant step toward resolving the positional O.O.D. challenge, enabling more reliable long-text understanding in LLMs. Our implementation of GALI, along with the experiments from our paper, is open-sourced at this https URL. 

**Abstract (ZH)**: 基于Transformer的大型语言模型（LLMs）在处理超出其训练上下文窗口的输入时表现不佳，因为长距离的输入会导致位置上的“离分布状况”（Positional Out-of-Distribution, POOD），从而破坏注意力计算。现有的解决方案，如微调和无需训练的方法，都受到计算效率低下、注意力分数异常或局部位置信息丢失的限制。为了解决这个问题，我们提出了一种无需训练的长度外推方法——贪婪注意力分数插值（GALI, Greedy Attention Logit Interpolation），该方法通过注意力分数插值最大化预训练位置区间的同时避免了注意力分数异常。实验结果表明，GALI 在性能上持续优于现有的无需训练的方法。我们的研究发现，LLMs在训练上下文窗口内的位置区间上解释不均匀，这表明在较小的位置区间范围内进行外推可以取得更好的效果，即使对于短上下文任务也是如此。GALI 表示朝着解决位置POOD挑战的一项重要进步，使得LLMs在处理长文本时更为可靠。我们所开发的GALI 实现及论文中的实验已开源在此网址：[请填写具体网址]。 

---
# Do Large Language Model Benchmarks Test Reliability? 

**Title (ZH)**: 大型语言模型基准测试能否衡量可靠性？ 

**Authors**: Joshua Vendrow, Edward Vendrow, Sara Beery, Aleksander Madry  

**Link**: [PDF](https://arxiv.org/pdf/2502.03461)  

**Abstract**: When deploying large language models (LLMs), it is important to ensure that these models are not only capable, but also reliable. Many benchmarks have been created to track LLMs' growing capabilities, however there has been no similar focus on measuring their reliability. To understand the potential ramifications of this gap, we investigate how well current benchmarks quantify model reliability. We find that pervasive label errors can compromise these evaluations, obscuring lingering model failures and hiding unreliable behavior.
Motivated by this gap in the evaluation of reliability, we then propose the concept of so-called platinum benchmarks, i.e., benchmarks carefully curated to minimize label errors and ambiguity. As a first attempt at constructing such benchmarks, we revise examples from fifteen existing popular benchmarks. We evaluate a wide range of models on these platinum benchmarks and find that, indeed, frontier LLMs still exhibit failures on simple tasks such as elementary-level math word problems. Analyzing these failures further reveals previously unidentified patterns of problems on which frontier models consistently struggle. We provide code at this https URL 

**Abstract (ZH)**: 在部署大型语言模型（LLMs）时，确保这些模型不仅功能强大，而且可靠也非常关键。虽然已经创建了许多基准来跟踪LLMs的能力增长，但迄今为止没有类似的焦点放在衡量其可靠性的方面。为了理解这一差距的潜在影响，我们调查了当前基准在衡量模型可靠性方面的有效性。我们发现，普遍的标签错误可能会破坏这些评估，使得持续存在的模型失败和不可靠行为变得隐匿。

鉴于在可靠性评估方面存在的这一差距，我们随后提出了所谓的铂金基准的概念，即精心筛选并编程的基准，旨在最小化标签错误和模糊性。我们首先尝试构建此类基准，对十五个现有流行基准中的一些示例进行了修订。我们在这些铂金基准上评估了多种模型，并发现前沿的LLMs仍然在诸如基础数学文字问题等简单任务上表现出失败。进一步分析这些失败揭示了前沿模型在某些问题上一贯存在的未识别困难模式。代码可在以下链接获得：[提供代码的链接] 

---
# Adapt-Pruner: Adaptive Structural Pruning for Efficient Small Language Model Training 

**Title (ZH)**: Adapt-Pruner: 适应性结构剪枝以实现高效的小型语言模型训练 

**Authors**: Boyao Wang, Rui Pan, Shizhe Diao, Xingyuan Pan, Jipeng Zhang, Renjie Pi, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.03460)  

**Abstract**: Small language models (SLMs) have attracted considerable attention from both academia and industry due to their broad range of applications in edge devices. To obtain SLMs with strong performance, conventional approaches either pre-train the models from scratch, which incurs substantial computational costs, or compress/prune existing large language models (LLMs), which results in performance drops and falls short in comparison to pre-training. In this paper, we investigate the family of acceleration methods that involve both structured pruning and model training. We found 1) layer-wise adaptive pruning (Adapt-Pruner) is extremely effective in LLMs and yields significant improvements over existing pruning techniques, 2) adaptive pruning equipped with further training leads to models comparable to those pre-training from scratch, 3) incremental pruning brings non-trivial performance gain by interleaving pruning with training and only removing a small portion of neurons ($\sim$5%) at a time. Experimental results on LLaMA-3.1-8B demonstrate that Adapt-Pruner outperforms conventional pruning methods, such as LLM-Pruner, FLAP, and SliceGPT, by an average of 1%-7% in accuracy on commonsense benchmarks. Additionally, Adapt-Pruner restores the performance of MobileLLM-125M to 600M on the MMLU benchmark with 200$\times$ fewer tokens via pruning from its larger counterparts, and discovers a new 1B model that surpasses LLaMA-3.2-1B in multiple benchmarks. 

**Abstract (ZH)**: 小语言模型（SLM）由于其在边缘设备中广泛的应用范围，引起了学术界和工业界的广泛关注。为了获得高性能的SLM，传统的做法要么从头开始预训练模型，这会带来大量的计算成本，要么压缩/剪枝现有的大型语言模型（LLM），这会导致性能下降并无法与预训练相比。本文探讨了同时包含结构剪枝和模型训练的一系列加速方法。我们发现：1）层次适配剪枝（Adapt-Pruner）在LLM中表现极其出色，其效果显著优于现有的剪枝技术；2）结合进一步训练的适配剪枝可以产生与从头预训练相当的模型；3）增量剪枝通过交替进行剪枝和训练，并分阶段移除少量神经元（约5%），能够带来显著的性能提升。在LLaMA-3.1-8B上的实验结果表明，Adapt-Pruner在常识基准测试中的平均准确率比传统的剪枝方法（如LLM-Pruner、FLAP、SliceGPT）高1%至7%。此外，Adapt-Pruner通过从其较大的模型中剪枝，将MobileLLM-125M在MMLU基准测试中的性能恢复到相当于600M的水平，且仅使用了后者的1/200的token量。Adapt-Pruner还发现了一个新的1B模型，在多个基准测试中超过了LLaMA-3.2-1B。 

---
# Harmony in Divergence: Towards Fast, Accurate, and Memory-efficient Zeroth-order LLM Fine-tuning 

**Title (ZH)**: 协调分歧：面向快速、准确且内存高效的零阶大模型微调 

**Authors**: Qitao Tan, Jun Liu, Zheng Zhan, Caiwei Ding, Yanzhi Wang, Jin Lu, Geng Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2502.03304)  

**Abstract**: Large language models (LLMs) excel across various tasks, but standard first-order (FO) fine-tuning demands considerable memory, significantly limiting real-world deployment. Recently, zeroth-order (ZO) optimization stood out as a promising memory-efficient training paradigm, avoiding backward passes and relying solely on forward passes for gradient estimation, making it attractive for resource-constrained scenarios. However, ZO method lags far behind FO method in both convergence speed and accuracy. To bridge the gap, we introduce a novel layer-wise divergence analysis that uncovers the distinct update pattern of FO and ZO optimization. Aiming to resemble the learning capacity of FO method from the findings, we propose \textbf{Di}vergence-driven \textbf{Z}eroth-\textbf{O}rder (\textbf{DiZO}) optimization. DiZO conducts divergence-driven layer adaptation by incorporating projections to ZO updates, generating diverse-magnitude updates precisely scaled to layer-wise individual optimization needs. Our results demonstrate that DiZO significantly reduces the needed iterations for convergence without sacrificing throughput, cutting training GPU hours by up to 48\% on various datasets. Moreover, DiZO consistently outperforms the representative ZO baselines in fine-tuning RoBERTa-large, OPT-series, and Llama-series on downstream tasks and, in some cases, even surpasses memory-intensive FO fine-tuning. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种任务中表现出色，但标准的一阶（FO）微调需要大量的内存，这极大地限制了其实用部署。最近，零阶（ZO）优化作为一种高效的内存使用训练范式脱颖而出，它避免了反向传播，仅依赖于前向传播进行梯度估计，使其适用于资源受限的场景。然而，ZO方法在收敛速度和准确性方面远远落后于FO方法。为了缩小差距，我们引入了一种新的逐层偏差分析，揭示了FO和ZO优化的不同更新模式。基于这一发现，我们提出了一种**Dev**iation-驱动的**Z**eroth-**O**rder（**DiZO**）优化方法。DiZO通过将投影纳入ZO更新中，进行偏差驱动的逐层适应，生成精确适应逐层个体优化需求的多样化幅度更新。我们的实验结果表明，DiZO能够在不牺牲吞吐量的情况下显著减少收敛所需迭代次数，从而在不同数据集上将训练GPU小时数降低了高达48%。此外，DiZO在对RoBERTa-large、OPT系列和Llama系列进行微调时，在下游任务上始终优于代表性的ZO基准方法，在某些情况下甚至超过了内存密集的一阶微调。 

---
# SymAgent: A Neural-Symbolic Self-Learning Agent Framework for Complex Reasoning over Knowledge Graphs 

**Title (ZH)**: SymAgent：一种用于知识图谱复杂推理的神经符号自我学习代理框架 

**Authors**: Ben Liu, Jihai Zhang, Fangquan Lin, Cheng Yang, Min Peng, Wotao Yin  

**Link**: [PDF](https://arxiv.org/pdf/2502.03283)  

**Abstract**: Recent advancements have highlighted that Large Language Models (LLMs) are prone to hallucinations when solving complex reasoning problems, leading to erroneous results. To tackle this issue, researchers incorporate Knowledge Graphs (KGs) to improve the reasoning ability of LLMs. However, existing methods face two limitations: 1) they typically assume that all answers to the questions are contained in KGs, neglecting the incompleteness issue of KGs, and 2) they treat the KG as a static repository and overlook the implicit logical reasoning structures inherent in KGs. In this paper, we introduce SymAgent, an innovative neural-symbolic agent framework that achieves collaborative augmentation between KGs and LLMs. We conceptualize KGs as dynamic environments and transform complex reasoning tasks into a multi-step interactive process, enabling KGs to participate deeply in the reasoning process. SymAgent consists of two modules: Agent-Planner and Agent-Executor. The Agent-Planner leverages LLM's inductive reasoning capability to extract symbolic rules from KGs, guiding efficient question decomposition. The Agent-Executor autonomously invokes predefined action tools to integrate information from KGs and external documents, addressing the issues of KG incompleteness. Furthermore, we design a self-learning framework comprising online exploration and offline iterative policy updating phases, enabling the agent to automatically synthesize reasoning trajectories and improve performance. Experimental results demonstrate that SymAgent with weak LLM backbones (i.e., 7B series) yields better or comparable performance compared to various strong baselines. Further analysis reveals that our agent can identify missing triples, facilitating automatic KG updates. 

**Abstract (ZH)**: 近年来的研究表明，大型语言模型（LLMs）在解决复杂推理问题时容易产生幻觉（hallucination），导致错误的结果。为了解决这一问题，研究人员通过引入知识图（KGs）来提高LLMs的推理能力。然而，现有方法存在两个局限性：1）它们通常假设所有答案都包含在KGs中，忽视了KGs的不完整性问题；2）它们将KG视为静态资源，并忽略了KG中固有的隐含逻辑推理结构。在此论文中，我们提出了SymAgent，这是一种创新的神经-符号代理框架，实现了KGs和LLMs之间的协作增强。我们将KGs视为动态环境，并将复杂的推理任务转化为多步交互过程，使KGs能够深度参与推理过程。SymAgent由两个模块组成：Agent-Planner和Agent-Executor。Agent-Planner利用LLMs的归纳推理能力从KGs中提取符号规则，指导有效的问题分解。Agent-Executor自主调用预定义的动作工具，整合KGs和外部文档中的信息，解决KG不完整的问题。此外，我们设计了一个自我学习框架，包括在线探索和离线迭代策略更新阶段，使代理能够自动综合推理轨迹并提高性能。实验结果表明，使用较弱的LLM底座（如7B系列）的SymAgent相较于各种强大的基线具有更好的或相当的性能。进一步分析表明，我们的代理能够识别缺失的三元组，从而促进自动更新KG。 

---
# Analyze Feature Flow to Enhance Interpretation and Steering in Language Models 

**Title (ZH)**: 分析特征流动以增强语言模型的解释性和可控性 

**Authors**: Daniil Laptev, Nikita Balagansky, Yaroslav Aksenov, Daniil Gavrilov  

**Link**: [PDF](https://arxiv.org/pdf/2502.03032)  

**Abstract**: We introduce a new approach to systematically map features discovered by sparse autoencoder across consecutive layers of large language models, extending earlier work that examined inter-layer feature links. By using a data-free cosine similarity technique, we trace how specific features persist, transform, or first appear at each stage. This method yields granular flow graphs of feature evolution, enabling fine-grained interpretability and mechanistic insights into model computations. Crucially, we demonstrate how these cross-layer feature maps facilitate direct steering of model behavior by amplifying or suppressing chosen features, achieving targeted thematic control in text generation. Together, our findings highlight the utility of a causal, cross-layer interpretability framework that not only clarifies how features develop through forward passes but also provides new means for transparent manipulation of large language models. 

**Abstract (ZH)**: 我们将提出一种新的方法，系统地将稀疏自编码器在大型语言模型连续层中发现的特征映射起来，扩展了早期研究中对跨层特征链接的探索。通过使用数据驱动的余弦相似性技术，我们追踪每个阶段中特定特征的持续性、变换性或首次出现情况。这种方法提供了细粒度的特征演化流程图，使模型计算具有精细的可解释性和机制洞察。关键的是，我们展示了这些跨层特征图如何直接引导模型行为，通过放大或抑制选定特征来实现有针对性的主题控制。总体而言，我们的研究结果突出了因果关系跨层解释框架的实用价值，该框架不仅澄清了特征在前向传递过程中如何演变，还提供了对大型语言模型的透明操控的新途径。 

---
# Scaling Laws for Upcycling Mixture-of-Experts Language Models 

**Title (ZH)**: 向上调整专家混合语言模型的缩放定律 

**Authors**: Seng Pei Liew, Takuya Kato, Sho Takase  

**Link**: [PDF](https://arxiv.org/pdf/2502.03009)  

**Abstract**: Pretraining large language models (LLMs) is resource-intensive, often requiring months of training time even with high-end GPU clusters. There are two approaches of mitigating such computational demands: reusing smaller models to train larger ones (upcycling), and training computationally efficient models like mixture-of-experts (MoE). In this paper, we study the upcycling of LLMs to MoE models, of which the scaling behavior remains underexplored. Through extensive experiments, we identify empirical scaling laws that describe how performance depends on dataset size and model configuration. Particularly, we show that, while scaling these factors improves performance, there is a novel interaction term between the dense and upcycled training dataset that limits the efficiency of upcycling at large computational budgets. Based on these findings, we provide guidance to scale upcycling, and establish conditions under which upcycling outperforms from-scratch trainings within budget constraints. 

**Abstract (ZH)**: 预训练大型语言模型（LLMs）需要大量的资源，即使使用高端GPU集群，也需要数月的训练时间。减缓这种计算需求的方法有两种：利用较小的模型来训练较大的模型（即“升级利用”），以及训练计算高效的模型，如专家混合模型（MoE）。在本文中，我们研究了将LLMs转化为MoE模型的升级利用问题，而这种模型扩展行为尚未得到充分探索。通过广泛实验，我们确定了经验扩展规律，描述了性能如何依赖于数据集大小和模型配置。特别地，我们证明了虽然扩大这些因素可以改善性能，但在大规模计算预算下，密集训练数据和升级利用训练数据之间的新型交互项限制了升级利用的效率。基于这些发现，我们提供了升级利用扩展的指导，并确立了在预算约束下，升级利用比从头训练更具优势的条件。 

---
# SPARC: Subspace-Aware Prompt Adaptation for Robust Continual Learning in LLMs 

**Title (ZH)**: SPARC：面向子空间的提示适配以提高大规模语言模型的稳健连续学习能力 

**Authors**: Dinithi Jayasuriya, Sina Tayebati, Davide Ettori, Ranganath Krishnan, Amit Ranjan Trivedi  

**Link**: [PDF](https://arxiv.org/pdf/2502.02909)  

**Abstract**: We propose SPARC, a lightweight continual learning framework for large language models (LLMs) that enables efficient task adaptation through prompt tuning in a lower-dimensional space. By leveraging principal component analysis (PCA), we identify a compact subspace of the training data. Optimizing prompts in this lower-dimensional space enhances training efficiency, as it focuses updates on the most relevant features while reducing computational overhead. Furthermore, since the model's internal structure remains unaltered, the extensive knowledge gained from pretraining is fully preserved, ensuring that previously learned information is not compromised during adaptation. Our method achieves high knowledge retention in both task-incremental and domain-incremental continual learning setups while fine-tuning only 0.04% of the model's parameters. Additionally, by integrating LoRA, we enhance adaptability to computational constraints, allowing for a tradeoff between accuracy and training cost. Experiments on the SuperGLUE benchmark demonstrate that our PCA-based prompt tuning combined with LoRA maintains full knowledge retention while improving accuracy, utilizing only 1% of the model's parameters. These results establish our approach as a scalable and resource-efficient solution for continual learning in LLMs. 

**Abstract (ZH)**: 我们提出了一种轻量级的持续学习框架SPARC，该框架通过在低维空间中的提示调优来实现大型语言模型（LLMs）的任务适配，从而提高效率。通过利用主成分分析（PCA），我们识别出训练数据的一个紧凑子空间。在这种低维空间中优化提示增强了训练效率，因为它将更新集中在最相关的特征上，同时减少了计算开销。此外，由于模型的内部结构保持不变，从预训练中获得的大量知识得到了完整保留，这确保了在适配过程中不会损害之前学习的信息。我们的方法在任务增量和领域增量的持续学习设置中实现了高知识保留率，同时只微调了模型参数的0.04%。此外，通过整合LoRA，我们增强了对计算约束的适应性，允许在准确性和训练成本之间做出权衡。在SuperGLUE基准测试中的实验表明，我们的基于PCA的提示调优与LoRA相结合，不仅保持了全部知识保留率，还提高了准确性，并且只使用了模型参数的1%。这些结果使我们的方法成为在LLMs中实现可扩展性和资源效率的持续学习解决方案。 

---
# ScholaWrite: A Dataset of End-to-End Scholarly Writing Process 

**Title (ZH)**: ScholaWrite：端到端学术写作过程数据集 

**Authors**: Linghe Wang, Minhwa Lee, Ross Volkov, Luan Tuyen Chau, Dongyeop Kang  

**Link**: [PDF](https://arxiv.org/pdf/2502.02904)  

**Abstract**: Writing is a cognitively demanding task involving continuous decision-making, heavy use of working memory, and frequent switching between multiple activities. Scholarly writing is particularly complex as it requires authors to coordinate many pieces of multiform knowledge. To fully understand writers' cognitive thought process, one should fully decode the end-to-end writing data (from individual ideas to final manuscript) and understand their complex cognitive mechanisms in scholarly writing. We introduce ScholaWrite dataset, the first-of-its-kind keystroke logs of an end-to-end scholarly writing process for complete manuscripts, with thorough annotations of cognitive writing intentions behind each keystroke. Our dataset includes LaTeX-based keystroke data from five preprints with nearly 62K total text changes and annotations across 4 months of paper writing. ScholaWrite shows promising usability and applications (e.g., iterative self-writing) for the future development of AI writing assistants for academic research, which necessitate complex methods beyond LLM prompting. Our experiments clearly demonstrated the importance of collection of end-to-end writing data, rather than the final manuscript, for the development of future writing assistants to support the cognitive thinking process of scientists. Our de-identified dataset, demo, and code repository are available on our project page. 

**Abstract (ZH)**: 写作是一项认知密集型任务，涉及持续的决策制定、大量使用工作记忆以及频繁在多项活动中切换。学术写作尤其复杂，因为作者需要协调多种多样的知识。要全面理解作家的认知思维过程，应该全面解读从个体思想到最终手稿的端到端写作数据，并理解其在学术写作中的复杂认知机制。我们介绍了ScholaWrite数据集，这是首个包含端到端学术写作全过程的键盘日志数据集，其中每一步键盘操作都有详细的认知书写意图标注。数据集包括五篇预印本的LaTeX基础键盘数据，总共包含近6.2万次文本更改，并有4个月时间跨度的注释。ScholaWrite展示了其对未来开发AI写作助手的支持潜力，这些助手对于学术研究来说需要超越大型语言模型（LLM）提示的复杂方法。我们的实验清楚地证明了，为了支持科学家的认知思维过程，未来的写作助手需要收集端到端的写作数据，而非仅限于最终的手稿。我们的去识别化数据集、演示和代码库可在项目页面上获取。 

---
# Leveraging the true depth of LLMs 

**Title (ZH)**: 利用大型语言模型的真正深度 

**Authors**: Ramón Calvo González, Daniele Paliotta, Matteo Pagliardini, Martin Jaggi, François Fleuret  

**Link**: [PDF](https://arxiv.org/pdf/2502.02790)  

**Abstract**: Large Language Models demonstrate remarkable capabilities at the cost of high compute requirements. While recent research has shown that intermediate layers can be removed or have their order shuffled without impacting performance significantly, these findings have not been employed to reduce the computational cost of inference. We investigate several potential ways to reduce the depth of pre-trained LLMs without significantly affecting performance. Leveraging our insights, we present a novel approach that exploits this decoupling between layers by grouping some of them into pairs that can be evaluated in parallel.
This modification of the computational graph -- through better parallelism -- results in an average improvement of around 1.20x on the number of tokens generated per second, without re-training nor fine-tuning, while retaining 95%-99% of the original accuracy. Empirical evaluation demonstrates that this approach significantly improves serving efficiency while maintaining model performance, offering a practical improvement for large-scale LLM deployment. 

**Abstract (ZH)**: 大规模语言模型在高计算需求的代价下展示了显著的能力。虽然最近的研究表明中间层可以被移除或重新排序而不显著影响性能，但这些发现尚未被应用于降低推理的计算成本。我们探讨了几种可能的方法，在不显著影响性能的前提下减少预训练语言模型的深度。利用我们的见解，我们提出了一种新颖的方法，通过将一些层分组，形成可以并行评估的对来利用层之间的分离。

通过这种计算图的修改——通过更好的并行性——在不重新训练或微调的情况下，我们获得了大约1.20倍的每秒生成token数的平均改进，同时保留了原始准确度的95%-99%。实证评估表明，这种方法在保持模型性能的同时显著提高了服务效率，为大规模语言模型的部署提供了实用的改进。 

---
# Twilight: Adaptive Attention Sparsity with Hierarchical Top-$p$ Pruning 

**Title (ZH)**: 黎明：自适应注意力稀疏性与分层Top-$p$剪枝 

**Authors**: Chaofan Lin, Jiaming Tang, Shuo Yang, Hanshuo Wang, Tian Tang, Boyu Tian, Ion Stoica, Song Han, Mingyu Gao  

**Link**: [PDF](https://arxiv.org/pdf/2502.02770)  

**Abstract**: Leveraging attention sparsity to accelerate long-context large language models (LLMs) has been a hot research topic. However, current algorithms such as sparse attention or key-value (KV) cache compression tend to use a fixed budget, which presents a significant challenge during deployment because it fails to account for the dynamic nature of real-world scenarios, where the optimal balance between accuracy and efficiency can vary greatly. In this paper, we find that borrowing top-$p$ sampling (nucleus sampling) to sparse attention can surprisingly achieve adaptive budgeting. Based on this, we propose Twilight, a framework to bring adaptive sparsity to any existing sparse attention algorithm without sacrificing their accuracy. Empirical results show that Twilight can adaptively prune at most 98% of redundant tokens, leading to $15.4\times$ acceleration in self-attention operations and $3.9\times$ acceleration in end-to-end per token latency in long context LLM decoding. 

**Abstract (ZH)**: 利用注意力稀疏性加速大型语言模型（LLMs）具有长上下文能力的研究已经成为一个热点研究领域。然而，当前的方法如稀疏注意力或键值（KV）缓存压缩往往采用固定预算，这在部署过程中带来了显著的挑战，因为它未能考虑现实世界场景中的动态特性，在这种场景中，准确性和效率之间的最优平衡可能会有很大差异。在本文中，我们发现借用top-$p$采样（核采样）技术来调整稀疏注意力可以出乎意料地实现自适应预算。基于此，我们提出了一种Twilight框架，该框架可以将自适应稀疏性引入任何现有的稀疏注意力算法中，而不牺牲其准确性。实验结果表明，Twilight可以自适应地剪枝多达98%的冗余 token，在长上下文LLMs解码中，自注意力操作加速了15.4倍，并且在端到端每token时延上加速了3.9倍。 

---
# Peri-LN: Revisiting Layer Normalization in the Transformer Architecture 

**Title (ZH)**: Peri-LN：重新审视Transformer架构中的层规范化 

**Authors**: Jeonghoon Kim, Byeongchan Lee, Cheonbok Park, Yeontaek Oh, Beomjun Kim, Taehwan Yoo, Seongjin Shin, Dongyoon Han, Jinwoo Shin, Kang Min Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2502.02732)  

**Abstract**: Designing Transformer architectures with the optimal layer normalization (LN) strategy that ensures large-scale training stability and expedite convergence has remained elusive, even in this era of large language models (LLMs). To this end, we present a comprehensive analytical foundation for understanding how different LN strategies influence training dynamics in large-scale Transformer training. Until recently, Pre-LN and Post-LN have long dominated standard practices despite their limitations in large-scale training. However, several open-source large-scale models have recently begun silently adopting a third strategy without much explanation. This strategy places layer normalization (LN) peripherally around sublayers, a design we term Peri-LN. While Peri-LN has demonstrated promising empirical performance, its precise mechanisms and benefits remain almost unexplored. Our in-depth analysis shows that Peri-LN strikes an ideal balance in variance growth -- unlike Pre-LN and Post-LN, which are prone to vanishing gradients and ``massive activations.'' To validate our theoretical insight, we conduct large-scale experiments on Transformers up to 3.2B parameters, showing that Peri-LN consistently achieves more balanced variance growth, steadier gradient flow, and convergence stability. Our results suggest that Peri-LN warrants broader consideration for large-scale Transformer architectures, providing renewed insights into the optimal placement and application of LN. 

**Abstract (ZH)**: 在大型语言模型（LLMs）的时代，设计具有最佳层标准化（LN）策略的Transformer架构，以确保大规模训练的稳定性和加速收敛，仍然是一个难题。为此，我们提供了一个全面的理论基础，以理解不同LN策略如何影响大规模Transformer训练的动力学。尽管预层标准化（Pre-LN）和后层标准化（Post-LN）长期以来一直是标准实践，但由于它们在大规模训练中的局限性，直到最近，一些开源的大规模模型才开始悄悄采用第三种策略，但没有详细说明。这种策略将层标准化（LN）放置在子层的周围，我们将其称为peri-LN。尽管peri-LN在经验上表现出色，但其精确机制和优势仍然几乎没有被探索。我们的深入分析表明，peri-LN在方差增长方面达到了理想的平衡——与Pre-LN和Post-LN相比，peri-LN不易出现梯度消失和“巨大激活”等问题。为了验证我们的理论洞见，我们在多达32亿参数的Transformer上进行了大规模实验，显示peri-LN能够实现更加均衡的方差增长、更稳定的梯度流动以及更好的收敛稳定性。研究结果表明，peri-LN值得在大规模Transformer架构中更广泛地考虑，为LN的最佳放置和应用提供了新的见解。 

---
# A Unified Understanding and Evaluation of Steering Methods 

**Title (ZH)**: 一个统一的理解和评估 steering 方法的框架 

**Authors**: Shawn Im, Yixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.02716)  

**Abstract**: Steering methods provide a practical approach to controlling large language models by applying steering vectors to intermediate activations, guiding outputs toward desired behaviors while avoiding retraining. Despite their growing importance, the field lacks a unified understanding and consistent evaluation across tasks and datasets, hindering progress. This paper introduces a unified framework for analyzing and evaluating steering methods, formalizing their core principles and offering theoretical insights into their effectiveness. Through comprehensive empirical evaluations on multiple-choice and open-ended text generation tasks, we validate these insights, identifying key factors that influence performance and demonstrating the superiority of certain methods. Our work bridges theoretical and practical perspectives, offering actionable guidance for advancing the design, optimization, and deployment of steering methods in LLMs. 

**Abstract (ZH)**: 引导方法通过将引导向量应用于中间激活来为控制大规模语言模型提供了一种实用的方法，从而引导输出向期望行为的方向发展，同时避免重新训练。尽管这些方法的重要性日益增加，但该领域缺乏统一的理解和跨任务和数据集的一致评估，阻碍了进展。本文提出了一种统一框架，用于分析和评估引导方法，正式化其核心原则，并提供了关于其有效性的理论见解。通过在多项选择和开放生成文本生成任务上的全面实证评估，我们验证了这些见解，识别了影响性能的关键因素，并证明了某些方法的优越性。我们的工作弥合了理论与实践的视角，提供了关于如何推进引导方法在大规模语言模型中的设计、优化和部署的实际指导。 

---
# Streaming Speaker Change Detection and Gender Classification for Transducer-Based Multi-Talker Speech Translation 

**Title (ZH)**: 基于转换器的多人讲话者语音翻译中的流式讲话者变更检测与性别分类 

**Authors**: Peidong Wang, Naoyuki Kanda, Jian Xue, Jinyu Li, Xiaofei Wang, Aswin Shanmugam Subramanian, Junkun Chen, Sunit Sivasankaran, Xiong Xiao, Yong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.02683)  

**Abstract**: Streaming multi-talker speech translation is a task that involves not only generating accurate and fluent translations with low latency but also recognizing when a speaker change occurs and what the speaker's gender is. Speaker change information can be used to create audio prompts for a zero-shot text-to-speech system, and gender can help to select speaker profiles in a conventional text-to-speech model. We propose to tackle streaming speaker change detection and gender classification by incorporating speaker embeddings into a transducer-based streaming end-to-end speech translation model. Our experiments demonstrate that the proposed methods can achieve high accuracy for both speaker change detection and gender classification. 

**Abstract (ZH)**: 流式多说话人语音翻译是一项不仅涉及在低延迟下生成准确流畅的翻译，还涉及识别说话人变更以及确定说话人性别的任务。说话人变更信息可以用于为零样本文本到语音系统创建音频提示，而性别信息则有助于在传统文本到语音模型中选择说话人配置文件。我们提出通过将说话人嵌入融入基于转换器的流式端到端语音翻译模型来解决流式说话人变更检测和性别分类问题。实验结果表明，所提出的方法在说话人变更检测和性别分类上均能取得高精度。 

---
# On Teacher Hacking in Language Model Distillation 

**Title (ZH)**: 《语言模型精炼中的教师篡改研究》

注：这里的翻译尽量保留了原文的学术意味，并且使用了较为准确的中文学术术语。在学术翻译中，确保专业性和准确性是非常重要的，因此在翻译时尽量使用专业词汇和表达方式。 

**Authors**: Daniil Tiapkin, Daniele Calandriello, Johan Ferret, Sarah Perrin, Nino Vieillard, Alexandre Ramé, Mathieu Blondel  

**Link**: [PDF](https://arxiv.org/pdf/2502.02671)  

**Abstract**: Post-training of language models (LMs) increasingly relies on the following two stages: (i) knowledge distillation, where the LM is trained to imitate a larger teacher LM, and (ii) reinforcement learning from human feedback (RLHF), where the LM is aligned by optimizing a reward model. In the second RLHF stage, a well-known challenge is reward hacking, where the LM over-optimizes the reward model. Such phenomenon is in line with Goodhart's law and can lead to degraded performance on the true objective. In this paper, we investigate whether a similar phenomenon, that we call teacher hacking, can occur during knowledge distillation. This could arise because the teacher LM is itself an imperfect approximation of the true distribution. To study this, we propose a controlled experimental setup involving: (i) an oracle LM representing the ground-truth distribution, (ii) a teacher LM distilled from the oracle, and (iii) a student LM distilled from the teacher. Our experiments reveal the following insights. When using a fixed offline dataset for distillation, teacher hacking occurs; moreover, we can detect it by observing when the optimization process deviates from polynomial convergence laws. In contrast, employing online data generation techniques effectively mitigates teacher hacking. More precisely, we identify data diversity as the key factor in preventing hacking. Overall, our findings provide a deeper understanding of the benefits and limitations of distillation for building robust and efficient LMs. 

**Abstract (ZH)**: 语言模型（LMs）的后训练过程越来越多地依赖于以下两个阶段：(i) 知识蒸馏（knowledge distillation），其中LM被训练以模仿一个较大的教师模型；(ii) 基于人类反馈的强化学习（reinforcement learning from human feedback, RLHF），其中通过优化奖励模型使LM与目标对齐。在第二阶段的RLHF中，一个众所周知的挑战是奖励作弊（reward hacking），即LM过度优化了奖励模型。这种现象符合Goodehart定律，并可能导致LM在真正的目标上表现退化。本文探讨了在知识蒸馏过程中，是否会出现类似的概念，我们称之为教师作弊（teacher hacking）。这可能是因为教师模型本身是对真实分布的一种不完美的近似。为研究这一问题，我们提出了一个受控实验设置，包括：(i) 一个代表真实分布的先验模型（oracle LM），(ii) 从先验模型蒸馏出的教师模型，以及(iii) 从教师模型蒸馏出的学生模型。实验结果显示了以下见解。当使用固定离线数据集进行蒸馏时，教师作弊现象会出现；此外，我们可以通过观察优化过程是否偏离多项式收敛法则来检测它。相比之下，采用在线数据生成技术可以有效减轻教师作弊现象。更具体而言，我们确定数据多样性是防止作弊的关键因素。总体而言，我们的发现提供了对蒸馏对构建稳健和高效语言模型的益处和限制的更深入理解。 

---
# ParetoQ: Scaling Laws in Extremely Low-bit LLM Quantization 

**Title (ZH)**: 帕列托Q：极低比特LLM量化中的标度定律 

**Authors**: Zechun Liu, Changsheng Zhao, Hanxian Huang, Sijia Chen, Jing Zhang, Jiawei Zhao, Scott Roy, Lisa Jin, Yunyang Xiong, Yangyang Shi, Lin Xiao, Yuandong Tian, Bilge Soran, Raghuraman Krishnamoorthi, Tijmen Blankevoort, Vikas Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2502.02631)  

**Abstract**: The optimal bit-width for achieving the best trade-off between quantized model size and accuracy has been a subject of ongoing debate. While some advocate for 4-bit quantization, others propose that 1.58-bit offers superior results. However, the lack of a cohesive framework for different bits has left such conclusions relatively tenuous. We present ParetoQ, the first unified framework that facilitates rigorous comparisons across 1-bit, 1.58-bit, 2-bit, 3-bit, and 4-bit quantization settings. Our findings reveal a notable learning transition between 2 and 3 bits: For 3-bits and above, the fine-tuned models stay close to their original pre-trained distributions, whereas for learning 2-bit networks or below, the representations change drastically. By optimizing training schemes and refining quantization functions, ParetoQ surpasses all previous methods tailored to specific bit widths. Remarkably, our ParetoQ ternary 600M-parameter model even outperforms the previous SoTA ternary 3B-parameter model in accuracy, using only one-fifth of the parameters. Extensive experimentation shows that ternary, 2-bit, and 3-bit quantization maintains comparable performance in the size-accuracy trade-off and generally exceeds 4-bit and binary quantization. Considering hardware constraints, 2-bit quantization offers promising potential for memory reduction and speedup. 

**Abstract (ZH)**: 关于在量化模型大小与精度之间找到最佳权衡的最优位宽问题一直是辩论的焦点。虽然有些人支持使用4位量化，另一些人则认为使用1.58位量化能获得更优的结果。然而，缺乏统一的框架使得这些结论相对脆弱。我们提出了ParetoQ，这是首个统一框架，能够促进在1位、1.58位、2位、3位和4位量化设置之间进行严谨的比较。我们的研究表明，在2位和3位之间存在显著的学习转变：对于3位及以上量化，微调后的模型保持接近最初的预训练分布；而对于学习2位及以下量化网络的情况，表示会发生巨大的变化。通过优化训练方案并改进量化函数，ParetoQ 在所有针对特定位宽优化的方法中表现更优。更令人惊讶的是，我们基于ParetoQ的600百万参数三值模型在准确率上超越了之前最先进的30亿参数三值模型，仅使用了其五分之一的参数量。广泛的经验表明，在大小与准确率之间的权衡中，三值、2位和3位量化能够保持相当的性能，并通常优于4位和二值量化。考虑到硬件限制，2位量化具有降低内存消耗和加速的潜在优势。 

---
# SEAL: Speech Embedding Alignment Learning for Speech Large Language Model with Retrieval-Augmented Generation 

**Title (ZH)**: SEAL：用于检索增强生成的语音大规模语言模型的语音嵌入对齐学习 

**Authors**: Chunyu Sun, Bingyu Liu, Zhichao Cui, Anbin Qi, Tian-hao Zhang, Dinghao Zhou, Lewei Lu  

**Link**: [PDF](https://arxiv.org/pdf/2502.02603)  

**Abstract**: Embedding-based retrieval models have made significant strides in retrieval-augmented generation (RAG) techniques for text and multimodal large language models (LLMs) applications. However, when it comes to speech larage language models (SLLMs), these methods are limited to a two-stage process, where automatic speech recognition (ASR) is combined with text-based retrieval. This sequential architecture suffers from high latency and error propagation. To address these limitations, we propose a unified embedding framework that eliminates the need for intermediate text representations. Specifically, the framework includes separate speech and text encoders, followed by a shared scaling layer that maps both modalities into a common embedding space. Our model reduces pipeline latency by 50\% while achieving higher retrieval accuracy compared to traditional two-stage methods. We also provide a theoretical analysis of the challenges inherent in end-to-end speech retrieval and introduce architectural principles for effective speech-to-document matching. Extensive experiments demonstrate the robustness of our approach across diverse acoustic conditions and speaker variations, paving the way for a new paradigm in multimodal SLLMs retrieval systems. 

**Abstract (ZH)**: 基于嵌入的检索模型在文本和多模态大型语言模型（LLMs）中的检索增强生成（RAG）技术中取得了显著进展。然而，当应用于语音大型语言模型（SLLMs）时，这些方法仅限于两阶段过程，其中自动语音识别（ASR）与基于文本的检索相结合。这种顺序架构存在高延迟和错误传播的问题。为了解决这些问题，我们提出了一种统一的嵌入框架，消除了中间文本表示的需求。具体来说，该框架包括独立的语音编码器和文本编码器，随后是一个共享缩放层，将两种模态映射到一个共同的嵌入空间。我们的模型将管线延迟减少了50%，同时在检索准确性上优于传统的两阶段方法。我们还对端到端语音检索固有的挑战进行了理论分析，并介绍了有效的语音到文档匹配的架构原则。广泛的实验表明，我们的方法在多种声学条件和说话人口音下具有鲁棒性，为多模态SLLMs检索系统的全新范式铺平了道路。 

---
