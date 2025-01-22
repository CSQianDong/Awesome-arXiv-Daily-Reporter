# Generating with Fairness: A Modality-Diffused Counterfactual Framework for Incomplete Multimodal Recommendations 

**Title (ZH)**: 生成公正的推荐：一种跨模态生成对抗框架用于不完整多模态推荐 

**Authors**: Jin Li, Shoujin Wang, Qi Zhang, Shui Yu, Fang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.11916)  

**Abstract**: Incomplete scenario is a prevalent, practical, yet challenging setting in Multimodal Recommendations (MMRec), where some item modalities are missing due to various factors. Recently, a few efforts have sought to improve the recommendation accuracy by exploring generic structures from incomplete data. However, two significant gaps persist: 1) the difficulty in accurately generating missing data due to the limited ability to capture modality distributions; and 2) the critical but overlooked visibility bias, where items with missing modalities are more likely to be disregarded due to the prioritization of items' multimodal data over user preference alignment. This bias raises serious concerns about the fair treatment of items. To bridge these two gaps, we propose a novel Modality-Diffused Counterfactual (MoDiCF) framework for incomplete multimodal recommendations. MoDiCF features two key modules: a novel modality-diffused data completion module and a new counterfactual multimodal recommendation module. The former, equipped with a particularly designed multimodal generative framework, accurately generates and iteratively refines missing data from learned modality-specific distribution spaces. The latter, grounded in the causal perspective, effectively mitigates the negative causal effects of visibility bias and thus assures fairness in recommendations. Both modules work collaboratively to address the two aforementioned significant gaps for generating more accurate and fair results. Extensive experiments on three real-world datasets demonstrate the superior performance of MoDiCF in terms of both recommendation accuracy and fairness 

**Abstract (ZH)**: 不完整场景是多模态推荐（MMRec）中一个普遍存在且具有挑战性的设置，在这种场景中，由于多种因素，某些项模态数据缺失。最近，一些努力试图通过探索不完整数据中的通用结构来提高推荐精度。然而，仍然存在两个重要的间隙：1）由于难以准确捕捉模态分布而导致的生成缺失数据的难度；2）忽视的但至关重要的可见性偏见，其中由于优先考虑项的多模态数据而非用户偏好匹配，具有缺失模态数据的项更容易被忽略。这种偏见引发了对公平对待项的严重关切。为了弥合这两个差距，我们提出了一种新颖的模态扩散反事实（MoDiCF）框架，用于不完整多模态推荐。MoDiCF 包含两个关键模块：一个新颖的模态扩散数据填充模块和一个新的反事实多模态推荐模块。前者配备了特别设计的多模态生成框架，能够从学习到的模态特定分布空间中准确生成并迭代细化缺失数据。后者以因果视角为基础，有效减轻了可见性偏见的负面影响，从而确保推荐的公平性。两个模块协作以解决上述两个关键差距，以生成更准确和公平的结果。在三个真实世界的数据集上进行的广泛实验表明，MoDiCF 在推荐准确性和公平性方面均表现出更优的性能。 

---
# PlotEdit: Natural Language-Driven Accessible Chart Editing in PDFs via Multimodal LLM Agents 

**Title (ZH)**: PlotEdit: 通过多模态大语言模型代理实现的基于自然语言的PDF图表编辑 

**Authors**: Kanika Goswami, Puneet Mathur, Ryan Rossi, Franck Dernoncourt  

**Link**: [PDF](https://arxiv.org/pdf/2501.11233)  

**Abstract**: Chart visualizations, while essential for data interpretation and communication, are predominantly accessible only as images in PDFs, lacking source data tables and stylistic information. To enable effective editing of charts in PDFs or digital scans, we present PlotEdit, a novel multi-agent framework for natural language-driven end-to-end chart image editing via self-reflective LLM agents. PlotEdit orchestrates five LLM agents: (1) Chart2Table for data table extraction, (2) Chart2Vision for style attribute identification, (3) Chart2Code for retrieving rendering code, (4) Instruction Decomposition Agent for parsing user requests into executable steps, and (5) Multimodal Editing Agent for implementing nuanced chart component modifications - all coordinated through multimodal feedback to maintain visual fidelity. PlotEdit outperforms existing baselines on the ChartCraft dataset across style, layout, format, and data-centric edits, enhancing accessibility for visually challenged users and improving novice productivity. 

**Abstract (ZH)**: 图表可视化虽然对于数据解释和传达至关重要，但在大多数情况下仅作为PDF中的图像存在，缺乏原始数据表和风格信息。为了使图表在PDF或数字扫描中的有效编辑成为可能，我们提出了PlotEdit，这是一种新颖的多代理框架，利用自省语言模型代理实现自然语言驱动的端到端图表图像编辑。PlotEdit协调了五个语言模型代理：（1）Chart2Table，用于提取数据表；（2）Chart2Vision，用于识别样式属性；（3）Chart2Code，用于检索渲染代码；（4）指令分解代理，用于将用户请求解析为可执行步骤；以及（5）多模态编辑代理，用于实现细腻的图表组件修改——所有这一切都通过多模态反馈来协调，以保持视觉保真度。在ChartCraft数据集上，PlotEdit在风格、布局、格式和数据导向编辑方面优于现有基线，提升了视觉障碍用户的可访问性，并改善了新手的生产力。 

---
# Verifying Cross-modal Entity Consistency in News using Vision-language Models 

**Title (ZH)**: 使用视觉语言模型在新闻中验证跨模态实体一致性 

**Authors**: Sahar Tahmasebi, Eric Müller-Budack, Ralph Ewerth  

**Link**: [PDF](https://arxiv.org/pdf/2501.11403)  

**Abstract**: The web has become a crucial source of information, but it is also used to spread disinformation, often conveyed through multiple modalities like images and text. The identification of inconsistent cross-modal information, in particular entities such as persons, locations, and events, is critical to detect disinformation. Previous works either identify out-of-context disinformation by assessing the consistency of images to the whole document, neglecting relations of individual entities, or focus on generic entities that are not relevant to news. So far, only few approaches have addressed the task of validating entity consistency between images and text in news. However, the potential of large vision-language models (LVLMs) has not been explored yet. In this paper, we propose an LVLM-based framework for verifying Cross-modal Entity Consistency~(LVLM4CEC), to assess whether persons, locations and events in news articles are consistent across both modalities. We suggest effective prompting strategies for LVLMs for entity verification that leverage reference images crawled from web. Moreover, we extend three existing datasets for the task of entity verification in news providing manual ground-truth data. Our results show the potential of LVLMs for automating cross-modal entity verification, showing improved accuracy in identifying persons and events when using evidence images. Moreover, our method outperforms a baseline for location and event verification in documents. The datasets and source code are available on GitHub at \url{this https URL}. 

**Abstract (ZH)**: 互联网已成为重要的信息来源，但也被用于传播假信息，这些信息通常通过多种模态（如图像和文本）进行传播。特别是识别一致性的跨模态信息（例如人物、地点和事件），对于检测假信息至关重要。先前的研究要么通过评估图像与整个文档的一致性来识别上下文不符的假信息，忽视了实体间的个体关系，要么专注于与新闻无关的通用实体。目前为止，仅有少数方法解决了新闻中的图像与文本实体一致性验证问题。然而，大规模多模态视觉语言模型（LVLM）的巨大潜力尚未被充分探索。在本文中，我们提出了一种基于LVLM的框架（LVLM4CEC），用于验证新闻文章中人物、地点和事件在两种模态下的一致性。我们为利用从网络抓取的参考图像构建了有效的LVLM提示策略，以进行实体验证。此外，我们为新闻场景下的实体验证任务扩展了三个现有数据集，并提供了手动标注的数据。我们的结果显示，LVLM在自动化跨模态实体验证方面具有潜力，使用证据图像时能够更准确地识别人物和事件。此外，我们的方法在文档中实体验证方面超过了基线方法。数据集和源代码已在GitHub上公开，地址为[this https URL](请将点击链接替换为实际的GitHub地址)。 

---
# LD-DETR: Loop Decoder DEtection TRansformer for Video Moment Retrieval and Highlight Detection 

**Title (ZH)**: LD-DETR：循环解码器检测变换器，用于视频关键 moment检索和高光检测 

**Authors**: Pengcheng Zhao, Zhixian He, Fuwei Zhang, Shujin Lin, Fan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2501.10787)  

**Abstract**: Video Moment Retrieval and Highlight Detection aim to find corresponding content in the video based on a text query. Existing models usually first use contrastive learning methods to align video and text features, then fuse and extract multimodal information, and finally use a Transformer Decoder to decode multimodal information. However, existing methods face several issues: (1) Overlapping semantic information between different samples in the dataset hinders the model's multimodal aligning performance; (2) Existing models are not able to efficiently extract local features of the video; (3) The Transformer Decoder used by the existing model cannot adequately decode multimodal features. To address the above issues, we proposed the LD-DETR model for Video Moment Retrieval and Highlight Detection tasks. Specifically, we first distilled the similarity matrix into the identity matrix to mitigate the impact of overlapping semantic information. Then, we designed a method that enables convolutional layers to extract multimodal local features more efficiently. Finally, we fed the output of the Transformer Decoder back into itself to adequately decode multimodal information. We evaluated LD-DETR on four public benchmarks and conducted extensive experiments to demonstrate the superiority and effectiveness of our approach. Our model outperforms the State-Of-The-Art models on QVHighlight, Charades-STA and TACoS datasets. Our code is available at this https URL. 

**Abstract (ZH)**: 视频片段检索和亮点检测旨在基于文本查询在视频中找到相应的內容。现有模型通常首先使用对比学习方法对视频和文本特征进行对齐，然后融合和提取多模态信息，最后使用Transformer Decoder解码多模态信息。然而，现有方法面临几个问题：（1）数据集中不同样本之间的重叠语义信息阻碍了模型的多模态对齐性能；（2）现有模型无法高效提取视频的局部特征；（3）现有模型中使用的Transformer Decoder无法充分解码多模态特征。为了解决上述问题，我们提出了LD-DETR模型，用于视频片段检索和亮点检测任务。具体而言，我们首先将相似度矩阵提炼成单位矩阵，以减轻不同样本之间的重叠语义信息的影响。然后，我们设计了一种方法，使卷积层能够更有效地提取多模态局部特征。最后，我们将Transformer Decoder的输出重新反馈到其自身，以充分解码多模态信息。我们将LD-DETR在四个公开基准上进行了评估，并进行了广泛的实验以展示我们方法的优越性和有效性。我们的模型在QVHighlight、Charades-STA和TACoS数据集上优于当前最佳模型。我们的代码可以在以下链接获取：[代码链接]。 

---
# The Value of Nothing: Multimodal Extraction of Human Values Expressed by TikTok Influencers 

**Title (ZH)**: 《一无所有？：TikTok影响者表达的人类价值观的多模态提取》 

**Authors**: Alina Starovolsky-Shitrit, Alon Neduva, Naama Appel Doron, Ella Daniel, Oren Tsur  

**Link**: [PDF](https://arxiv.org/pdf/2501.11770)  

**Abstract**: Societal and personal values are transmitted to younger generations through interaction and exposure. Traditionally, children and adolescents learned values from parents, educators, or peers. Nowadays, social platforms serve as a significant channel through which youth (and adults) consume information, as the main medium of entertainment, and possibly the medium through which they learn different values. In this paper we extract implicit values from TikTok movies uploaded by online influencers targeting children and adolescents. We curated a dataset of hundreds of TikTok movies and annotated them according to the Schwartz Theory of Personal Values. We then experimented with an array of Masked and Large language model, exploring how values can be detected. Specifically, we considered two pipelines -- direct extraction of values from video and a 2-step approach in which videos are first converted to elaborated scripts and then values are extracted.
Achieving state-of-the-art results, we find that the 2-step approach performs significantly better than the direct approach and that using a trainable Masked Language Model as a second step significantly outperforms a few-shot application of a number of Large Language Models. We further discuss the impact of fine-tuning and compare the performance of the different models on identification of values present or contradicted in the TikTok. Finally, we share the first values-annotated dataset of TikTok videos. Our results pave the way to further research on influence and value transmission in video-based social platforms. 

**Abstract (ZH)**: 社会和个人价值观通过互动和接触传给年轻一代。传统上，儿童和青少年从父母、教育者或其他同龄人那里学习价值观。如今，社交媒体平台已成为年轻人（及成年人）获取信息的重要渠道，是主要的娱乐媒介，也是他们可能获取不同价值观的媒介。在本文中，我们从针对儿童和青少年的网络影响者上传的TikTok视频中提取隐含的价值观。我们收集了一大批TikTok视频，并根据斯瓦西尔个人价值观理论对它们进行了标注。我们随后使用了一系列遮蔽和大规模语言模型进行实验，探索如何检测价值观。具体而言，我们考虑了两种方法——直接从视频中提取价值观和两步法，在这种方法中，首先将视频转换为详细的剧本，然后提取价值观。

我们取得了当前最佳成果，发现两步法明显优于直接方法，使用可训练的遮蔽语言模型作为第二步显著优于大量大规模语言模型的少量示例应用。我们进一步讨论了模型微调的影响，并比较了不同模型在识别TikTok中呈现或反驳的价值观方面的性能。最后，我们分享了首个TikTok视频的价值观标注数据集。我们的结果为视频为基础的社交平台上影响和价值观传递的研究开辟了新的方向。 

---
# Mobile-Agent-E: Self-Evolving Mobile Assistant for Complex Tasks 

**Title (ZH)**: 移动代理-E：用于复杂任务的自我进化移动助理 

**Authors**: Zhenhailong Wang, Haiyang Xu, Junyang Wang, Xi Zhang, Ming Yan, Ji Zhang, Fei Huang, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2501.11733)  

**Abstract**: Smartphones have become indispensable in modern life, yet navigating complex tasks on mobile devices often remains frustrating. Recent advancements in large multimodal model (LMM)-based mobile agents have demonstrated the ability to perceive and act in mobile environments. However, current approaches face significant limitations: they fall short in addressing real-world human needs, struggle with reasoning-intensive and long-horizon tasks, and lack mechanisms to learn and improve from prior experiences. To overcome these challenges, we introduce Mobile-Agent-E, a hierarchical multi-agent framework capable of self-evolution through past experience. By hierarchical, we mean an explicit separation of high-level planning and low-level action execution. The framework comprises a Manager, responsible for devising overall plans by breaking down complex tasks into subgoals, and four subordinate agents--Perceptor, Operator, Action Reflector, and Notetaker--which handle fine-grained visual perception, immediate action execution, error verification, and information aggregation, respectively. Mobile-Agent-E also features a novel self-evolution module which maintains a persistent long-term memory comprising Tips and Shortcuts. Tips are general guidance and lessons learned from prior tasks on how to effectively interact with the environment. Shortcuts are reusable, executable sequences of atomic operations tailored for specific subroutines. The inclusion of Tips and Shortcuts facilitates continuous refinement in performance and efficiency. Alongside this framework, we introduce Mobile-Eval-E, a new benchmark featuring complex mobile tasks requiring long-horizon, multi-app interactions. Empirical results show that Mobile-Agent-E achieves a 22% absolute improvement over previous state-of-the-art approaches across three foundation model backbones. Project page: this https URL. 

**Abstract (ZH)**: 智能手机已成为现代生活不可或缺的工具，但在移动设备上导航复杂任务往往仍然令人沮丧。基于大规模多模态模型（LMM）的移动代理最近在感知和执行移动环境方面展现了能力。然而，当前的方法在处理现实世界的人类需求、应对推理密集型和长期任务方面存在显著局限性，缺乏从先前经验中学习和改进的机制。为了克服这些挑战，我们提出了Mobile-Agent-E，一种能够通过以往经验进行自我进化的分层多代理框架。所谓分层，指的是明确区分高层级规划和低层级动作执行。该框架包括一个经理，负责将复杂任务分解为子目标以制定总体计划；以及四个下属代理——感知器、操作员、动作反思器和记录员，分别负责精细的视觉感知、立即的动作执行、错误验证和信息聚合。Mobile-Agent-E 还配备了一个新颖的自我进化模块，该模块维护了一个持久的长期记忆，包括提示和捷径。提示是关于如何有效与环境互动的一般指导和从先前任务中吸取的教训。捷径是为特定子例行程序量身定制的可重复使用、可执行的原子操作序列。提示和捷径的包含促进了持续的性能和效率改进。除此之外，我们还引入了Mobile-Eval-E，这是一个新的基准测试，包括需要长期交互和多应用互动的复杂移动任务。实验结果表明，Mobile-Agent-E 在三个基础模型框架上实现了比之前最先进的方法绝对改进22%。项目页面：请点击此处。 

---
# AIMA at SemEval-2024 Task 3: Simple Yet Powerful Emotion Cause Pair Analysis 

**Title (ZH)**: AIMA在SemEval-2024任务3中的简单而强大的情感原因配对分析 

**Authors**: Alireza Ghahramani Kure, Mahshid Dehghani, Mohammad Mahdi Abootorabi, Nona Ghazizadeh, Seyed Arshan Dalili, Ehsaneddin Asgari  

**Link**: [PDF](https://arxiv.org/pdf/2501.11170)  

**Abstract**: The SemEval-2024 Task 3 presents two subtasks focusing on emotion-cause pair extraction within conversational contexts. Subtask 1 revolves around the extraction of textual emotion-cause pairs, where causes are defined and annotated as textual spans within the conversation. Conversely, Subtask 2 extends the analysis to encompass multimodal cues, including language, audio, and vision, acknowledging instances where causes may not be exclusively represented in the textual data. Our proposed model for emotion-cause analysis is meticulously structured into three core segments: (i) embedding extraction, (ii) cause-pair extraction & emotion classification, and (iii) cause extraction using QA after finding pairs. Leveraging state-of-the-art techniques and fine-tuning on task-specific datasets, our model effectively unravels the intricate web of conversational dynamics and extracts subtle cues signifying causality in emotional expressions. Our team, AIMA, demonstrated strong performance in the SemEval-2024 Task 3 competition. We ranked as the 10th in subtask 1 and the 6th in subtask 2 out of 23 teams. 

**Abstract (ZH)**: SemEval-2024 任务3围绕对话情境中情感-因果对的提取设定了两个子任务。子任务1专注于提取文本形式的情感-因果对，其中因果关系被定义和标注为对话中的文本片段。相反，子任务2则扩展分析范围，包括语言、音频和视觉等多种模态的线索，承认在某些情况下，因果关系可能不完全体现在文本数据中。我们提出的用于情感-因果分析的模型被仔细地划分为三个核心模块：（i）嵌入提取，（ii）因果对提取与情感分类，以及（iii）在找到因果对后使用问答（QA）进行因果提取。利用最先进的技术和针对特定任务的数据集进行微调，我们的模型有效地揭示了对话动态的复杂网络，并提取了表示情感表达因果关系的细微线索。我们团队AIMA在SemEval-2024任务3的比赛中表现出色。我们在这两个子任务中分别排名第10位和第6位，参赛团队共有23支。 

---
# BAP v2: An Enhanced Task Framework for Instruction Following in Minecraft Dialogues 

**Title (ZH)**: BAP v2：一种增强的任务框架，用于Minecraft对话中的指令跟随 

**Authors**: Prashant Jayannavar, Liliang Ren, Marisa Hudspeth, Charlotte Lambert, Ariel Cordes, Elizabeth Kaplan, Anjali Narayan-Chen, Julia Hockenmaier  

**Link**: [PDF](https://arxiv.org/pdf/2501.10836)  

**Abstract**: Interactive agents capable of understanding and executing instructions in the physical world have long been a central goal in AI research. The Minecraft Collaborative Building Task (MCBT) provides one such setting to work towards this goal (Narayan-Chen, Jayannavar, and Hockenmaier 2019). It is a two-player game in which an Architect (A) instructs a Builder (B) to construct a target structure in a simulated Blocks World Environment. We focus on the challenging Builder Action Prediction (BAP) subtask of predicting correct action sequences in a given multimodal game context with limited training data (Jayannavar, Narayan-Chen, and Hockenmaier 2020). We take a closer look at evaluation and data for the BAP task, discovering key challenges and making significant improvements on both fronts to propose BAP v2, an upgraded version of the task. This will allow future work to make more efficient and meaningful progress on it. It comprises of: (1) an enhanced evaluation benchmark that includes a cleaner test set and fairer, more insightful metrics, and (2) additional synthetic training data generated from novel Minecraft dialogue and target structure simulators emulating the MCBT. We show that the synthetic data can be used to train more performant and robust neural models even with relatively simple training methods. Looking ahead, such data could also be crucial for training more sophisticated, data-hungry deep transformer models and training/fine-tuning increasingly large LLMs. Although modeling is not the primary focus of this work, we also illustrate the impact of our data and training methodologies on a simple LLM- and transformer-based model, thus validating the robustness of our approach, and setting the stage for more advanced architectures and LLMs going forward. 

**Abstract (ZH)**: 能够在物理世界中理解并执行指令的交互式代理一直是AI研究的中心目标之一。Minecraft协作建造任务（MCBT）提供了一个这样的环境，旨在向这个目标迈进（Narayan-Chen, Jayannavar, and Hockenmaier 2019）。这是一个两人游戏，在模拟的Blocks World环境中，建筑师（A）会指导建造者（B）建造一个目标结构。我们专注于“建造者动作预测”（BAP）子任务，该任务涉及根据有限的训练数据预测给定多模态游戏上下文中的正确动作序列（Jayannavar, Narayan-Chen, and Hockenmaier 2020）。我们更详细地审视了BAP任务的评估和数据，发现了关键挑战，并在两个方面取得了显著改进，提出了BAP v2，即任务的升级版本。这将使未来的工作能够更高效且有意义地推进该任务。具体包括：（1）改进的评估基准，包括更清洁的测试集和更公平、更深入的指标；（2）从模拟MCBT的新MC游戏对话和目标结构生成的额外合成训练数据。我们展示了合成数据即使使用相对简单的训练方法也能用于训练性能更优、更稳健的神经网络模型。展望未来，这样的数据对于训练更复杂的、数据需求更大的深度变换模型以及训练/微调越来越大的语言模型也可能至关重要。虽然建模不是本工作的主要焦点，但我们也展示了我们的数据和训练方法对简单基于语言模型和变换器模型的影响，从而验证了我们方法的稳健性，并为未来更高级架构和语言模型奠定了基础。 

---
# MMVU: Measuring Expert-Level Multi-Discipline Video Understanding 

**Title (ZH)**: MMVU：衡量多学科视频理解的专业水平 

**Authors**: Yilun Zhao, Lujing Xie, Haowei Zhang, Guo Gan, Yitao Long, Zhiyuan Hu, Tongyan Hu, Weiyuan Chen, Chuhan Li, Junyang Song, Zhijian Xu, Chengye Wang, Weifeng Pan, Ziyao Shangguan, Xiangru Tang, Zhenwen Liang, Yixin Liu, Chen Zhao, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2501.12380)  

**Abstract**: We introduce MMVU, a comprehensive expert-level, multi-discipline benchmark for evaluating foundation models in video understanding. MMVU includes 3,000 expert-annotated questions spanning 27 subjects across four core disciplines: Science, Healthcare, Humanities & Social Sciences, and Engineering. Compared to prior benchmarks, MMVU features three key advancements. First, it challenges models to apply domain-specific knowledge and perform expert-level reasoning to analyze specialized-domain videos, moving beyond the basic visual perception typically assessed in current video benchmarks. Second, each example is annotated by human experts from scratch. We implement strict data quality controls to ensure the high quality of the dataset. Finally, each example is enriched with expert-annotated reasoning rationals and relevant domain knowledge, facilitating in-depth analysis. We conduct an extensive evaluation of 32 frontier multimodal foundation models on MMVU. The latest System-2-capable models, o1 and Gemini 2.0 Flash Thinking, achieve the highest performance among the tested models. However, they still fall short of matching human expertise. Through in-depth error analyses and case studies, we offer actionable insights for future advancements in expert-level, knowledge-intensive video understanding for specialized domains. 

**Abstract (ZH)**: 我们将介绍MMVU，这是一个全面的专家级多学科基准，用于评估基础模型在视频理解方面的性能。MMVU 包括涵盖四个核心学科的 27 个主题的 3000 个由专家注释的问题：科学、医疗保健、人文学科与社会科学、以及工程学。与先前的基准相比，MMVU 具有三个关键的改进。首先，它要求模型应用特定领域的知识并执行专家级推理，以分析专门领域的视频，这超越了当前视频基准中通常评估的基本视觉感知。第二，每个示例都从零开始由人类专家进行标注。我们实施严格的数据质量控制，以确保数据集的质量。最后，每个示例都得到了专家注释的推理依据和相关领域知识的丰富，便于深入分析。我们在 MMVU 上对 32 个前沿多模态基础模型进行了广泛的评估。最新的系统-2 级模型 o1 和 Gemini 2.0 Flash Thinking 在测试模型中表现出最高的性能，但仍然无法达到人类专业知识的水平。通过深入的错误分析和案例研究，我们为未来在专门领域中实现专家级、知识密集型视频理解提供了可操作的见解。 

---
# InternLM-XComposer2.5-Reward: A Simple Yet Effective Multi-Modal Reward Model 

**Title (ZH)**: InternLM-XComposer2.5-奖励：一个简单而有效的多模态奖励模型 

**Authors**: Yuhang Zang, Xiaoyi Dong, Pan Zhang, Yuhang Cao, Ziyu Liu, Shengyuan Ding, Shenxi Wu, Yubo Ma, Haodong Duan, Wenwei Zhang, Kai Chen, Dahua Lin, Jiaqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.12368)  

**Abstract**: Despite the promising performance of Large Vision Language Models (LVLMs) in visual understanding, they occasionally generate incorrect outputs. While reward models (RMs) with reinforcement learning or test-time scaling offer the potential for improving generation quality, a critical gap remains: publicly available multi-modal RMs for LVLMs are scarce, and the implementation details of proprietary models are often unclear. We bridge this gap with InternLM-XComposer2.5-Reward (IXC-2.5-Reward), a simple yet effective multi-modal reward model that aligns LVLMs with human preferences. To ensure the robustness and versatility of IXC-2.5-Reward, we set up a high-quality multi-modal preference corpus spanning text, image, and video inputs across diverse domains, such as instruction following, general understanding, text-rich documents, mathematical reasoning, and video understanding. IXC-2.5-Reward achieves excellent results on the latest multi-modal reward model benchmark and shows competitive performance on text-only reward model benchmarks. We further demonstrate three key applications of IXC-2.5-Reward: (1) Providing a supervisory signal for RL training. We integrate IXC-2.5-Reward with Proximal Policy Optimization (PPO) yields IXC-2.5-Chat, which shows consistent improvements in instruction following and multi-modal open-ended dialogue; (2) Selecting the best response from candidate responses for test-time scaling; and (3) Filtering outlier or noisy samples from existing image and video instruction tuning training data. To ensure reproducibility and facilitate further research, we have open-sourced all model weights and training recipes at this https URL 

**Abstract (ZH)**: 尽管大规模视觉语言模型（LVLMs）在视觉理解方面表现出色，但它们偶尔会产生错误的输出。虽然通过强化学习或测试时缩放的方式生成奖励模型（RMs）有可能提高生成的质量，但仍有关键性的差距：公开的多模态奖励模型对LVLMs的支持不足，而且大多数私有模型的具体实现细节也不明确。我们通过InternLM-XComposer2.5-Reward（IXC-2.5-Reward）填补了这一差距，这是一种简单但有效的多模态奖励模型，能够使LVLMs与人类偏好保持一致。为了确保IXC-2.5-Reward的稳健性和灵活性，我们在文本、图像和视频等多个领域建立了高质量的多模态偏好语料库，涵盖指令遵循、通用理解、文本丰富的文档、数学推理和视频理解等不同领域。IXC-2.5-Reward在最新的多模态奖励模型基准测试中取得了优异的成果，并在仅文本的奖励模型基准测试中表现出竞争力。

我们进一步展示了IXC-2.5-Reward的三个关键应用：（1）作为强化学习（RL）训练的监督信号。我们将IXC-2.5-Reward集成到近端策略优化（PPO）中，从而获得IXC-2.5-Chat，它在指令遵循和多模态开放性对话中都表现出一致的改进；（2）从候选回答中选择最佳响应，用于测试时缩放；（3）从现有的图像和视频指令调优训练数据中过滤异常值或噪声样本。

为了确保可复制性并促进进一步的研究，我们已在以下链接开放了所有模型权重和训练食谱：[此链接]。 

---
# InsTALL: Context-aware Instructional Task Assistance with Multi-modal Large Language Models 

**Title (ZH)**: InsTALL：基于上下文的多模态大型语言模型辅助教学任务 

**Authors**: Pha Nguyen, Sailik Sengupta, Girik Malik, Arshit Gupta, Bonan Min  

**Link**: [PDF](https://arxiv.org/pdf/2501.12231)  

**Abstract**: The improved competence of generative models can help building multi-modal virtual assistants that leverage modalities beyond language. By observing humans performing multi-step tasks, one can build assistants that have situational awareness of actions and tasks being performed, enabling them to cater assistance based on this understanding. In this paper, we develop a Context-aware Instructional Task Assistant with Multi-modal Large Language Models (InsTALL) that leverages an online visual stream (e.g. a user's screen share or video recording) and responds in real-time to user queries related to the task at hand. To enable useful assistance, InsTALL 1) trains a multi-modal model on task videos and paired textual data, and 2) automatically extracts task graph from video data and leverages it at training and inference time. We show InsTALL achieves state-of-the-art performance across proposed sub-tasks considered for multimodal activity understanding -- task recognition (TR), action recognition (AR), next action prediction (AP), and plan prediction (PP) -- and outperforms existing baselines on two novel sub-tasks related to automatic error identification. 

**Abstract (ZH)**: 生成模型能力的提升有助于构建多模态虚拟助手，这些虚拟助手可以利用语言之外的多种模态信息。通过观察人类执行多步骤任务的过程，可以构建具有情境意识的助手，使其能够根据对这些任务的理解提供相应的帮助。在本文中，我们提出了一种基于多模态大规模语言模型的上下文感知指令任务助手（InsTALL），该助手利用在线视觉流（例如，用户的屏幕共享或视频录制）并实时响应与当前任务相关的用户查询。为了提供有用的帮助，InsTALL 1) 在任务视频及其配对的文本数据上训练一个多模态模型，2) 自动从视频数据中提取任务图，并在训练和推理过程中利用该图。我们展示了InsTALL在多模态活动理解所提出的子任务——任务识别（TR）、动作识别（AR）、下一个动作预测（AP）和计划预测（PP）——方面的性能达到了最先进的水平，并在两个与自动错误识别相关的新型子任务上优于现有 baseline。 

---
# EmbodiedEval: Evaluate Multimodal LLMs as Embodied Agents 

**Title (ZH)**: EmbodiedEval：评估多模态大语言模型作为具身代理 

**Authors**: Zhili Cheng, Yuge Tu, Ran Li, Shiqi Dai, Jinyi Hu, Shengding Hu, Jiahao Li, Yang Shi, Tianyu Yu, Weize Chen, Lei Shi, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2501.11858)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown significant advancements, providing a promising future for embodied agents. Existing benchmarks for evaluating MLLMs primarily utilize static images or videos, limiting assessments to non-interactive scenarios. Meanwhile, existing embodied AI benchmarks are task-specific and not diverse enough, which do not adequately evaluate the embodied capabilities of MLLMs. To address this, we propose EmbodiedEval, a comprehensive and interactive evaluation benchmark for MLLMs with embodied tasks. EmbodiedEval features 328 distinct tasks within 125 varied 3D scenes, each of which is rigorously selected and annotated. It covers a broad spectrum of existing embodied AI tasks with significantly enhanced diversity, all within a unified simulation and evaluation framework tailored for MLLMs. The tasks are organized into five categories: navigation, object interaction, social interaction, attribute question answering, and spatial question answering to assess different capabilities of the agents. We evaluated the state-of-the-art MLLMs on EmbodiedEval and found that they have a significant shortfall compared to human level on embodied tasks. Our analysis demonstrates the limitations of existing MLLMs in embodied capabilities, providing insights for their future development. We open-source all evaluation data and simulation framework at this https URL. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在实现重要进展的同时，为 embodied 代理带来了光明的未来。现有的 MLLM 评估基准主要依赖静态图像或视频，这限制了评估范围仅限于非交互式场景。同时，现有的 embodied AI 基准多为特定任务，缺乏多样性，不足以评估 MLLMs 的 embodied 能力。为解决这一问题，我们提出了一种名为 EmbodiedEval 的全面且交互式的评估基准，专为 MLLMs 设计，涵盖 embodied 任务。EmbodiedEval 包含 125 个不同场景中的 328 个独立任务，每个场景都经过严格选择和标注。它涵盖了现有的多种 embodied AI 任务，具备显著增强的多样性，并在为 MLLMs 设计的统一仿真和评估框架中进行了整合。任务被分类为五个类别：导航、对象交互、社会交互、属性问题回答和空间问题回答，以评估代理的不同能力。我们对当前最先进的 MLLMs 进行了评估，并发现它们在 embodied 任务上与人类水平相比存在显著差距。我们的分析揭示了现有 MLLMs 在 embodied 能力方面的局限性，为它们的未来发展方向提供了洞察。我们在以下网址开源了所有评估数据和仿真框架：[提供链接]。 

---
# Advancing General Multimodal Capability of Vision-language Models with Pyramid-descent Visual Position Encoding 

**Title (ZH)**: 通过金字塔降布局位编码提升视觉语言模型的通用多模态能力 

**Authors**: Zhanpeng Chen, Mingxiao Li, Ziyang Chen, Nan Du, Xiaolong Li, Yuexian Zou  

**Link**: [PDF](https://arxiv.org/pdf/2501.10967)  

**Abstract**: Vision-language Models (VLMs) have shown remarkable capabilities in advancing general artificial intelligence, yet the irrational encoding of visual positions persists in inhibiting the models' comprehensive perception performance across different levels of granularity. In this work, we propose Pyramid-descent Visual Position Encoding (PyPE), a novel approach designed to enhance the perception of visual tokens within VLMs. By assigning visual position indexes from the periphery to the center and expanding the central receptive field incrementally, PyPE addresses the limitations of traditional raster-scan methods and mitigates the long-term decay effects induced by Rotary Position Embedding (RoPE). Our method reduces the relative distance between interrelated visual elements and instruction tokens, promoting a more rational allocation of attention weights and allowing for a multi-granularity perception of visual elements and countering the over-reliance on anchor tokens. Extensive experimental evaluations demonstrate that PyPE consistently improves the general capabilities of VLMs across various sizes. Code is available at this https URL. 

**Abstract (ZH)**: 视觉-语言模型（VLMs）在推进通用人工智能方面展示出了显著的能力，然而传统的 raster-scan 方法在视觉位置编码上的不合理性仍然对模型的综合感知性能构成了阻碍。在这项工作中，我们提出了一种新颖的方法——金字塔下沉视觉位置编码（PyPE），旨在增强 VLMs 中视觉标记的感知能力。通过从边缘到中心分配视觉位置索引，并逐步扩展中心的感受野，PyPE 解决了传统 raster-scan 方法的局限性，并缓解了由旋转位置嵌入（RoPE）引起的长期衰减效应。该方法减少了相关视觉元素和指令标记之间的相对距离，促进了更合理的注意力权重分配，并允许对视觉元素进行多粒度感知，从而减少对锚标记的过度依赖。广泛的实验证明，PyPE 在不同大小的 VLMs 中一致地提高了其通用能力。代码可在以下链接获取：this https URL。 

---
# Know "No" Better: A Data-Driven Approach for Enhancing Negation Awareness in CLIP 

**Title (ZH)**: 更好地了解“不”：一种基于数据的方法以增强CLIP中的否定意识 

**Authors**: Junsung Park, Jungbeom Lee, Jongyoon Song, Sangwon Yu, Dahuin Jung, Sungroh Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2501.10913)  

**Abstract**: While CLIP has significantly advanced multimodal understanding by bridging vision and language, the inability to grasp negation - such as failing to differentiate concepts like "parking" from "no parking" - poses substantial challenges. By analyzing the data used in the public CLIP model's pre-training, we posit this limitation stems from a lack of negation-inclusive data. To address this, we introduce data generation pipelines that employ a large language model (LLM) and a multimodal LLM to produce negation-inclusive captions. Fine-tuning CLIP with data generated from our pipelines, we develop NegationCLIP, which enhances negation awareness while preserving the generality. Moreover, to enable a comprehensive evaluation of negation understanding, we propose NegRefCOCOg-a benchmark tailored to test VLMs' ability to interpret negation across diverse expressions and positions within a sentence. Experiments on various CLIP architectures validate the effectiveness of our data generation pipelines in enhancing CLIP's ability to perceive negation accurately. Additionally, NegationCLIP's enhanced negation awareness has practical applications across various multimodal tasks, demonstrated by performance gains in text-to-image generation and referring image segmentation. 

**Abstract (ZH)**: 尽管CLIP在通过视觉和语言融合显著提升多模态理解方面取得进展，但其无法理解和区分否定概念（如“停车”与“禁止停车”）的能力仍然存在重大挑战。通过对公共CLIP模型预训练数据进行分析，我们推测这一局限性来源于缺乏包含否定信息的数据。为解决这一问题，我们引入了一种数据生成管道，使用大型语言模型（LLM）和多模态LLM生成包含否定信息的描述。通过使用我们的管道生成的数据对CLIP进行微调，我们开发了NegationCLIP，该模型增强了对否定的理解能力，同时保持了普适性。此外，为了全面评估模型对否定的理解能力，我们提出了一种专门用于测试VLM（视觉语言模型）在不同句子位置和表达方式中对否定理解能力的基准——NegRefCOCOg。对多种CLIP架构的实验验证了我们的数据生成管道在提高CLIP准确感知否定方面的有效性。此外，NegationCLIP增强的否定理解能力在多种多模态任务中具有实际应用价值，尤其是在文本生成图像和引用图像分割等任务中表现出性能提升。 

---
# Can Multimodal LLMs do Visual Temporal Understanding and Reasoning? The answer is No! 

**Title (ZH)**: 多模态LLM能够进行视觉时间理解与推理吗？答案是否定的！ 

**Authors**: Mohamed Fazli Imam, Chenyang Lyu, Alham Fikri Aji  

**Link**: [PDF](https://arxiv.org/pdf/2501.10674)  

**Abstract**: Multimodal Large Language Models (MLLMs) have achieved significant advancements in tasks like Visual Question Answering (VQA) by leveraging foundational Large Language Models (LLMs). However, their abilities in specific areas such as temporal understanding, which is crucial for comprehending real-world dynamics, remain underexplored. To address this, we propose a challenging evaluation benchmark named TemporalVQA, consisting of two parts: (1) Temporal Order Understanding and (2) Time-lapse Estimation. The first part requires MLLMs to determine the sequence of events by analyzing temporally consecutive video frames. The second part presents image pairs with varying time differences, framed as multiple-choice questions, asking MLLMs to estimate the time-lapse between images with options ranging from seconds to years. Our evaluations of advanced MLLMs, including models like GPT-4o and Gemini-1.5-Pro, reveal significant challenges: GPT-4o achieved only 43.8% average consistent accuracy in temporal order tasks and 70% in time-lapse estimation, with open-source models performing even less effectively. These findings underscore the limitations of current MLLMs in visual temporal understanding and reasoning, highlighting the need for further improvements in their temporal capabilities. Our dataset can be found at this https URL. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）通过利用基础大型语言模型（LLMs）在视觉问答（VQA）等任务上取得了显著进展。然而，它们在特定领域，如时间理解方面的能力仍然未被充分探索，而时间理解对于理解现实世界的动态至关重要。为了解决这一问题，我们提出了一项具有挑战性的评估基准 TemporalVQA，它包括两个部分：(1) 时间顺序理解；(2) 时间间隔估计。第一部分要求MLLMs通过分析时间连续的视频帧来确定事件的顺序。第二部分则展示了具有不同时间间隔的图像对，提出多项选择题形式的问题，要求MLLMs估计两幅图像之间的时间间隔，选项范围从秒到数年。我们对先进MLLMs，包括GPT-4o和Gemini-1.5-Pro等模型的评估表明了巨大挑战：GPT-4o在时间顺序任务上的平均一致准确率为43.8%，在时间间隔估计任务上的准确率为70%，开源模型的表现甚至更差。这些发现突显出当前MLLMs在视觉时间理解与推理方面的局限性，强调了进一步提高其时间处理能力的必要性。我们的数据集可以在以下网址获取：[此链接]。 

---
# When language and vision meet road safety: leveraging multimodal large language models for video-based traffic accident analysis 

**Title (ZH)**: 当语言与视觉携手共进交通安全：利用多模态大型语言模型进行基于视频的道路交通事故分析 

**Authors**: Ruixuan Zhang, Beichen Wang, Juexiao Zhang, Zilin Bian, Chen Feng, Kaan Ozbay  

**Link**: [PDF](https://arxiv.org/pdf/2501.10604)  

**Abstract**: The increasing availability of traffic videos functioning on a 24/7/365 time scale has the great potential of increasing the spatio-temporal coverage of traffic accidents, which will help improve traffic safety. However, analyzing footage from hundreds, if not thousands, of traffic cameras in a 24/7/365 working protocol remains an extremely challenging task, as current vision-based approaches primarily focus on extracting raw information, such as vehicle trajectories or individual object detection, but require laborious post-processing to derive actionable insights. We propose SeeUnsafe, a new framework that integrates Multimodal Large Language Model (MLLM) agents to transform video-based traffic accident analysis from a traditional extraction-then-explanation workflow to a more interactive, conversational approach. This shift significantly enhances processing throughput by automating complex tasks like video classification and visual grounding, while improving adaptability by enabling seamless adjustments to diverse traffic scenarios and user-defined queries. Our framework employs a severity-based aggregation strategy to handle videos of various lengths and a novel multimodal prompt to generate structured responses for review and evaluation and enable fine-grained visual grounding. We introduce IMS (Information Matching Score), a new MLLM-based metric for aligning structured responses with ground truth. We conduct extensive experiments on the Toyota Woven Traffic Safety dataset, demonstrating that SeeUnsafe effectively performs accident-aware video classification and visual grounding by leveraging off-the-shelf MLLMs. Source code will be available at \url{this https URL}. 

**Abstract (ZH)**: 随着交通视频在全年无休（24/7/365）模式下变得越来越普及，这为提升交通事故的空间-时间覆盖范围提供了巨大潜力，进而有助于提高交通安全。然而，按照24/7/365的工作模式分析数百甚至数千个交通摄像头的视频内容仍然是一个极其具有挑战性的任务，因为当前基于视觉的方法主要集中在提取诸如车辆轨迹或个体对象检测等原始信息上，但需要大量的后处理才能得出有效的洞察。我们提出了SeeUnsafe这一新框架，将多模态大语言模型（MLLM）代理集成进来，将基于视频的交通事故分析从传统的提取-解释工作流程转变为一种更交互式的对话式方法。这一转变通过自动化复杂任务（如视频分类和视觉定位）大幅提高了处理吞吐量，并通过使系统能够无缝适应各种交通场景和用户定义的查询而提高了可适应性。我们的框架采用一种基于严重程度的聚合策略来处理不同长度的视频，并引入一种新型的多模态提示来生成结构化响应，进行审核和评估，并实现细粒度的视觉定位。我们引入了IMS（信息匹配评分）这一新的MLLM基元度量标准，用于将结构化响应与地面真相对齐。我们在Toyota Woven Traffic Safety数据集上进行了广泛的实验，证明SeeUnsafe能够有效利用现成的MLLM进行事故意识视频分类和视觉定位。源代码将发布在\url{this https URL}。 

---
# Bridging Visualization and Optimization: Multimodal Large Language Models on Graph-Structured Combinatorial Optimization 

**Title (ZH)**: 跨模态连接与优化：基于图结构组合优化的多模态大型语言模型 

**Authors**: Jie Zhao, Kang Hao Cheong, Witold Pedrycz  

**Link**: [PDF](https://arxiv.org/pdf/2501.11968)  

**Abstract**: Graph-structured combinatorial challenges are inherently difficult due to their nonlinear and intricate nature, often rendering traditional computational methods ineffective or expensive. However, these challenges can be more naturally tackled by humans through visual representations that harness our innate ability for spatial reasoning. In this study, we propose transforming graphs into images to preserve their higher-order structural features accurately, revolutionizing the representation used in solving graph-structured combinatorial tasks. This approach allows machines to emulate human-like processing in addressing complex combinatorial challenges. By combining the innovative paradigm powered by multimodal large language models (MLLMs) with simple search techniques, we aim to develop a novel and effective framework for tackling such problems. Our investigation into MLLMs spanned a variety of graph-based tasks, from combinatorial problems like influence maximization to sequential decision-making in network dismantling, as well as addressing six fundamental graph-related issues. Our findings demonstrate that MLLMs exhibit exceptional spatial intelligence and a distinctive capability for handling these problems, significantly advancing the potential for machines to comprehend and analyze graph-structured data with a depth and intuition akin to human cognition. These results also imply that integrating MLLMs with simple optimization strategies could form a novel and efficient approach for navigating graph-structured combinatorial challenges without complex derivations, computationally demanding training and fine-tuning. 

**Abstract (ZH)**: 由于其非线性和错综复杂的特点，基于图的组合挑战本质上是困难的，常常使传统的计算方法变得无效或成本高昂。然而，这些挑战可以通过视觉表示更容易地由人类解决，利用我们天生的空间推理能力。在这项研究中，我们提出将图转换为图像以准确保留其高阶结构特征，从而彻底变革解决图结构组合任务的表示方式。这种方法使得机器能够在解决复杂组合挑战时模拟人类的处理方式。通过结合由多模态大语言模型（MLLMs）驱动的新颖范式与简单的搜索技术，我们旨在开发一个新颖且有效的框架来解决此类问题。我们对MLLMs的研究涵盖了多种基于图的任务，从组合问题如影响力最大化到网络拆解中的顺序决策，以及解决六项基本的图相关问题。我们的研究结果表明，MLLMs展示了卓越的空间智能和独特的处理这些任务的能力，这大大提高了机器理解并分析图结构数据的潜力，使其具备与人类认知相似的深度和直觉。这些结果还暗示，将MLLMs与简单的优化策略相结合，可能形成一种无需复杂推导、计算资源消耗低且高效的框架，以应对图结构组合挑战。 

---
# MAPS: Advancing Multi-Modal Reasoning in Expert-Level Physical Science 

**Title (ZH)**: MAPS：推进专家级物理科学中的多模态推理 

**Authors**: Erle Zhu, Yadi Liu, Zhe Zhang, Xujun Li, Jin Zhou, Xinjie Yu, Minlie Huang, Hongning Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.10768)  

**Abstract**: Pre-trained on extensive text and image corpora, current Multi-Modal Large Language Models (MLLM) have shown strong capabilities in general visual reasoning tasks. However, their performance is still lacking in physical domains that require understanding diagrams with complex physical structures and quantitative analysis based on multi-modal information. To address this, we develop a new framework, named Multi-Modal Scientific Reasoning with Physics Perception and Simulation (MAPS) based on an MLLM. MAPS decomposes expert-level multi-modal reasoning task into physical diagram understanding via a Physical Perception Model (PPM) and reasoning with physical knowledge via a simulator. The PPM module is obtained by fine-tuning a visual language model using carefully designed synthetic data with paired physical diagrams and corresponding simulation language descriptions. At the inference stage, MAPS integrates the simulation language description of the input diagram provided by PPM and results obtained through a Chain-of-Simulation process with MLLM to derive the underlying rationale and the final answer. Validated using our collected college-level circuit analysis problems, MAPS significantly improves reasoning accuracy of MLLM and outperforms all existing models. The results confirm MAPS offers a promising direction for enhancing multi-modal scientific reasoning ability of MLLMs. We will release our code, model and dataset used for our experiments upon publishing of this paper. 

**Abstract (ZH)**: 基于广泛的文本和图像语料库进行预训练，当前的多模态大型语言模型（Multimodal Large Language Models, MLLM）展示了在通用视觉推理任务中的强大能力。然而，它们在需要理解和分析复杂物理结构的图表以及基于多模态信息的定量分析的实际领域中的性能仍然不足。为了解决这个问题，我们基于MLLM开发了一个新的框架，名为多模态科学推理与物理感知和模拟（Multi-Modal Scientific Reasoning with Physics Perception and Simulation, MAPS）。MAPS将专家级别的多模态推理任务分解为通过物理感知模型（Physical Perception Model, PPM）理解物理图表，并通过模拟器利用物理知识进行推理。PPM模块通过使用精心设计的合成数据（配对了物理图表和相应的模拟语言描述）对视觉语言模型进行微调而获得。在推理阶段，MAPS将PPM提供的输入图表的模拟语言描述与通过链式模拟过程与MLLM获得的结果整合起来，以推导出潜在理由和最终答案。通过我们收集的大学级别电路分析问题进行验证，MAPS显著提高了MLLM的推理准确性并优于所有现有模型。结果证实，MAPS为增强MLLM的多模态科学推理能力提供了有前景的方向。我们将在本文发表后发布我们实验中使用的代码、模型和数据集。 

---
# Human-AI Collaborative Game Testing with Vision Language Models 

**Title (ZH)**: 基于视觉语言模型的人机协作游戏测试 

**Authors**: Boran Zhang, Muhan Xu, Zhijun Pan  

**Link**: [PDF](https://arxiv.org/pdf/2501.11782)  

**Abstract**: As modern video games become increasingly complex, traditional manual testing methods are proving costly and inefficient, limiting the ability to ensure high-quality game experiences. While advancements in Artificial Intelligence (AI) offer the potential to assist human testers, the effectiveness of AI in truly enhancing real-world human performance remains underexplored. This study investigates how AI can improve game testing by developing and experimenting with an AI-assisted workflow that leverages state-of-the-art machine learning models for defect detection. Through an experiment involving 800 test cases and 276 participants of varying backgrounds, we evaluate the effectiveness of AI assistance under four conditions: with or without AI support, and with or without detailed knowledge of defects and design documentation. The results indicate that AI assistance significantly improves defect identification performance, particularly when paired with detailed knowledge. However, challenges arise when AI errors occur, negatively impacting human decision-making. Our findings show the importance of optimizing human-AI collaboration and implementing strategies to mitigate the effects of AI inaccuracies. By this research, we demonstrate AI's potential and problems in enhancing efficiency and accuracy in game testing workflows and offers practical insights for integrating AI into the testing process. 

**Abstract (ZH)**: 随着现代视频游戏变得越来越复杂，传统的手工测试方法越来越昂贵且效率低下，这限制了确保高质量游戏体验的能力。虽然人工智能（AI）的进步提供了辅助人工测试者的潜力，但AI在真正提升实际人类性能方面的有效性仍被广泛忽视。本研究调查了AI如何通过开发并实验一种利用先进机器学习模型进行缺陷检测的AI辅助工作流程来提高游戏测试的效果。通过涉及800个测试案例和276名背景各异的参与者的一项实验，我们在四种条件下评估了AI辅助的效果：有或没有AI支持，以及有或没有详细的缺陷和设计文档知识。结果显示，当与详细的缺陷和设计文档知识结合使用时，AI辅助显著提高了缺陷识别性能。然而，当AI出现错误时，这会对人类的决策产生负面影响。我们的研究结果强调了优化人类与AI的合作的重要性，并提出了缓解AI不准确性影响的策略。通过本研究，我们展示了AI在提高游戏测试工作流程的效率和准确性方面的潜力和问题，并提供了将AI整合到测试过程中的实用见解。 

---
# TSVC:Tripartite Learning with Semantic Variation Consistency for Robust Image-Text Retrieval 

**Title (ZH)**: TSVC：具有语义变异性一致性的三方学习方法在稳健的图像-文本检索中的应用 

**Authors**: Shuai Lyu, Zijing Tian, Zhonghong Ou, Yifan Zhu, Xiao Zhang, Qiankun Ha, Haoran Luo, Meina Song  

**Link**: [PDF](https://arxiv.org/pdf/2501.10935)  

**Abstract**: Cross-modal retrieval maps data under different modality via semantic relevance. Existing approaches implicitly assume that data pairs are well-aligned and ignore the widely existing annotation noise, i.e., noisy correspondence (NC). Consequently, it inevitably causes performance degradation. Despite attempts that employ the co-teaching paradigm with identical architectures to provide distinct data perspectives, the differences between these architectures are primarily stemmed from random initialization. Thus, the model becomes increasingly homogeneous along with the training process. Consequently, the additional information brought by this paradigm is severely limited. In order to resolve this problem, we introduce a Tripartite learning with Semantic Variation Consistency (TSVC) for robust image-text retrieval. We design a tripartite cooperative learning mechanism comprising a Coordinator, a Master, and an Assistant model. The Coordinator distributes data, and the Assistant model supports the Master model's noisy label prediction with diverse data. Moreover, we introduce a soft label estimation method based on mutual information variation, which quantifies the noise in new samples and assigns corresponding soft labels. We also present a new loss function to enhance robustness and optimize training effectiveness. Extensive experiments on three widely used datasets demonstrate that, even at increasing noise ratios, TSVC exhibits significant advantages in retrieval accuracy and maintains stable training performance. 

**Abstract (ZH)**: 跨模态检索通过语义相关性将不同模态的数据映射到同一空间。现有方法隐含地假定数据对是良好对齐的，并且忽视了广泛存在的标注噪声，即噪声对应关系（NC）。因此，这不可避免地导致了性能下降。尽管有采用协同教学范式的方法，并利用相同的架构从不同角度提供数据视角，但这些架构之间的差异主要源于随机初始化。因此，随着训练过程的推进，模型变得越来越同质化。由此带来的附加信息严重受限。为了解决这一问题，我们提出了一种基于语义变异性一致性的三元学习框架（TSVC）以提高图像-文本检索的鲁棒性。我们设计了一种包含协调器、主模型和助手模型的三元合作学习机制。协调器负责分配数据，助手模型利用多样化数据支持主模型噪声标签预测。此外，我们引入了一种基于互信息变性的软标签估计方法，用于量化新样本中的噪声并分配相应的软标签。我们还提出了一种新的损失函数以增强鲁棒性并优化训练效果。在三个广泛使用的数据集上的广泛实验表明，即使在增加噪声比例的情况下，TSVC在检索准确性和保持稳定的训练性能方面仍具有显著优势。 

---
# Fake Advertisements Detection Using Automated Multimodal Learning: A Case Study for Vietnamese Real Estate Data 

**Title (ZH)**: 使用自动多模态学习进行虚假广告检测：越南房地产数据案例研究 

**Authors**: Duy Nguyen, Trung T. Nguyen, Cuong V. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2501.10848)  

**Abstract**: The popularity of e-commerce has given rise to fake advertisements that can expose users to financial and data risks while damaging the reputation of these e-commerce platforms. For these reasons, detecting and removing such fake advertisements are important for the success of e-commerce websites. In this paper, we propose FADAML, a novel end-to-end machine learning system to detect and filter out fake online advertisements. Our system combines techniques in multimodal machine learning and automated machine learning to achieve a high detection rate. As a case study, we apply FADAML to detect fake advertisements on popular Vietnamese real estate websites. Our experiments show that we can achieve 91.5% detection accuracy, which significantly outperforms three different state-of-the-art fake news detection systems. 

**Abstract (ZH)**: 电子商务的流行催生了虚假广告，这些广告不仅可能使用户面临财务和数据风险，还会损害这些电子商务平台的声誉。出于这些原因，检测和清除虚假广告对于电子商务网站的成功至关重要。本文提出了一种新颖的端到端机器学习系统FADAML，用于检测和过滤虚假在线广告。该系统结合了多模态机器学习技术和自动化机器学习方法，以实现较高的检测率。作为案例研究，我们将FADAML应用于检测越南房地产网站上的虚假广告。我们的实验结果显示，我们可以达到91.5%的检测准确率，这在很大程度上超越了三种不同的先进虚假新闻检测系统。 

---
# Visual RAG: Expanding MLLM visual knowledge without fine-tuning 

**Title (ZH)**: 视觉RAG：在不微调的情况下扩展MLLM视觉知识 

**Authors**: Mirco Bonomo, Simone Bianco  

**Link**: [PDF](https://arxiv.org/pdf/2501.10834)  

**Abstract**: Multimodal Large Language Models (MLLMs) have achieved notable performance in computer vision tasks that require reasoning across visual and textual modalities, yet their capabilities are limited to their pre-trained data, requiring extensive fine-tuning for updates. Recent researches have explored the use of In-Context Learning (ICL) to overcome these challenges by providing a set of demonstrating examples as context to augment MLLMs performance in several tasks, showing that many-shot ICL leads to substantial improvements compared to few-shot ICL. However, the reliance on numerous demonstrating examples and the limited MLLMs context windows presents significant obstacles. This paper aims to address these challenges by introducing a novel approach, Visual RAG, that synergically combines the MLLMs capability to learn from the context, with a retrieval mechanism. The crux of this approach is to ensure to augment the MLLM knowledge by selecting only the most relevant demonstrating examples for the query, pushing it to learn by analogy. In this way, relying on the new information provided dynamically during inference time, the resulting system is not limited to the knowledge extracted from the training data, but can be updated rapidly and easily without fine-tuning. Furthermore, this greatly reduces the computational costs for improving the model image classification performance, and augments the model knowledge to new visual domains and tasks it was not trained for. Extensive experiments on eight different datasets in the state of the art spanning several domains and image classification tasks show that the proposed Visual RAG, compared to the most recent state of the art (i.e., many-shot ICL), is able to obtain an accuracy that is very close or even higher (approx. +2% improvement on average) while using a much smaller set of demonstrating examples (approx. only 23% on average). 

**Abstract (ZH)**: 多模态大规模语言模型（Multimodal Large Language Models, MLLMs）在要求视觉和文本模态之间推理的计算机视觉任务中取得了显著性能，但其能力受限于预训练数据，需要大量的微调才能进行更新。近期研究探索了通过上下文学习（In-Context Learning, ICL）来应对这些挑战，通过提供一组示范示例作为上下文来增强MLLM在若干任务上的表现，研究表明大量示例的ICL相比于少量示例的ICL带来了显著的提升。然而，依赖大量示范示例和MLLM上下文窗口的限制带来了显著的障碍。本文旨在通过引入一种新颖的方法——Visual RAG，其结合了MLLM从上下文学习的能力与检索机制，来应对这些挑战。这种方法的核心在于通过选择与查询最相关的示范示例来增强MLLM的知识，促使它通过类比学习。这样，在推理时依赖于动态提供的新信息，该系统不仅限于训练数据中提取的知识，还可以快速且容易地进行更新，无需微调。此外，这种方法大大降低了提高模型图像分类性能的计算成本，并增强了模型在未训练的新视觉领域和任务中的知识。大量的实验表明，与最新的前沿技术（即大量示例的ICL）相比，提出的Visual RAG在使用更少的示范示例（约平均23%）的情况下，能够获得非常接近或甚至更高的准确率（约平均2%的改善）。实验涵盖了来自不同领域的八个最先进的数据集，涉及多个图像分类任务。 

---
