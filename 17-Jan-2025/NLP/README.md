# OmniThink: Expanding Knowledge Boundaries in Machine Writing through Thinking 

**Title (ZH)**: OmniThink：通过思考扩展机器写作的知识边界 

**Authors**: Zekun Xi, Wenbiao Yin, Jizhan Fang, Jialong Wu, Runnan Fang, Ningyu Zhang, Jiang Yong, Pengjun Xie, Fei Huang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.09751)  

**Abstract**: Machine writing with large language models often relies on retrieval-augmented generation. However, these approaches remain confined within the boundaries of the model's predefined scope, limiting the generation of content with rich information. Specifically, vanilla-retrieved information tends to lack depth, utility, and suffers from redundancy, which negatively impacts the quality of generated articles, leading to shallow, repetitive, and unoriginal outputs. To address these issues, we propose OmniThink, a machine writing framework that emulates the human-like process of iterative expansion and reflection. The core idea behind OmniThink is to simulate the cognitive behavior of learners as they progressively deepen their knowledge of the topics. Experimental results demonstrate that OmniThink improves the knowledge density of generated articles without compromising metrics such as coherence and depth. Human evaluations and expert feedback further highlight the potential of OmniThink to address real-world challenges in the generation of long-form articles. 

**Abstract (ZH)**: 大规模语言模型进行机器写作往往依赖于检索增强生成。然而，这些方法仍然受限于模型预定义的范围之内，限制了丰富信息内容的生成。具体来说，直接检索的信息往往缺乏深度、实用性和易冗余，这会负面影响生成文章的质量，导致输出内容浅薄、重复且缺乏原创性。为了解决这些问题，我们提出了一种名为OmniThink的机器写作框架，该框架模拟了人类逐步深化知识掌握过程中的迭代扩展和反思机制。OmniThink的核心思想是模拟学习者在逐渐加深对主题理解过程中的认知行为。实验结果表明，OmniThink能够在不牺牲一致性和深度等指标的情况下，提高生成文章的知识密度。进一步的人类评估和专家反馈凸显了OmniThink在应对长篇文章生成中的现实挑战方面的潜力。 

---
# Enhancing Lexicon-Based Text Embeddings with Large Language Models 

**Title (ZH)**: 使用大型语言模型增强基于词典的文本嵌入 

**Authors**: Yibin Lei, Tao Shen, Yu Cao, Andrew Yates  

**Link**: [PDF](https://arxiv.org/pdf/2501.09749)  

**Abstract**: Recent large language models (LLMs) have demonstrated exceptional performance on general-purpose text embedding tasks. While dense embeddings have dominated related research, we introduce the first Lexicon-based EmbeddiNgS (LENS) leveraging LLMs that achieve competitive performance on these tasks. Regarding the inherent tokenization redundancy issue and unidirectional attention limitations in traditional causal LLMs, LENS consolidates the vocabulary space through token embedding clustering, and investigates bidirectional attention and various pooling strategies. Specifically, LENS simplifies lexicon matching by assigning each dimension to a specific token cluster, where semantically similar tokens are grouped together, and unlocking the full potential of LLMs through bidirectional attention. Extensive experiments demonstrate that LENS outperforms dense embeddings on the Massive Text Embedding Benchmark (MTEB), delivering compact feature representations that match the sizes of dense counterparts. Notably, combining LENSE with dense embeddings achieves state-of-the-art performance on the retrieval subset of MTEB (i.e. BEIR). 

**Abstract (ZH)**: 近期的大规模语言模型（LLMs）在通用文本嵌入任务中展现了卓越的表现。尽管密集嵌入在过去相关研究中占主导地位，我们首次引入了基于词典的嵌入（Lexicon-based EmbeddiNgS，LENS），并通过LLMs实现了这些任务的竞争力。针对传统因果LLMs中固有的 token 化冗余问题和单向注意限制，LENS 通过 token 嵌入聚类缩小了词典空间，并研究了双向注意和各种池化策略。具体而言，LENS 通过将每个维度分配给特定的 token 聚类来简化词典匹配，使 semantically 相似的 token 被分组在一起，并通过双向注意充分发挥了LLMs的潜力。大规模文本嵌入基准测试（MTEB）的广泛实验表明，LENS 在密集嵌入上表现出优越性，提供了与密集嵌入相当大小的紧凑特征表示。值得注意的是，将LENS与密集嵌入结合起来，在MTEB的检索子集（即BEIR）上达到了最先进的性能。 

---
# Attention based Bidirectional GRU hybrid model for inappropriate content detection in Urdu language 

**Title (ZH)**: 基于注意力机制的双向GRU混合模型在乌尔都语不当内容检测中的应用 

**Authors**: Ezzah Shoukat, Rabia Irfan, Iqra Basharat, Muhammad Ali Tahir, Sameen Shaukat  

**Link**: [PDF](https://arxiv.org/pdf/2501.09722)  

**Abstract**: With the increased use of the internet and social networks for online discussions, the spread of toxic and inappropriate content on social networking sites has also increased. Several studies have been conducted in different languages. However, there is less work done for South Asian languages for inappropriate content identification using deep learning techniques. In Urdu language, the spellings are not unique, and people write different common spellings for the same word, while mixing it other languages, like English in the text makes it more challenging, and limited research work is available to process such language with the finest algorithms. The use of attention layer with a deep learning model can help handling the long-term dependencies and increase its efficiency . To explore the effects of the attention layer, this study proposes attention-based Bidirectional GRU hybrid model for identifying inappropriate content in Urdu Unicode text language. Four different baseline deep learning models; LSTM, Bi-LSTM, GRU, and TCN, are used to compare the performance of the proposed model. The results of these models were compared based on evaluation metrics, dataset size, and impact of the word embedding layer. The pre-trained Urdu word2Vec embeddings were utilized for our case. Our proposed model BiGRU-A outperformed all other baseline models by yielding 84\% accuracy without using pre-trained word2Vec layer. From our experiments, we have established that the attention layer improves the model's efficiency, and pre-trained word2Vec embedding does not work well with an inappropriate content dataset. 

**Abstract (ZH)**: 随着互联网和社交媒体在网络讨论中的使用增加，社交网络上不适当内容的传播也相应增加。已有多种语言的研究，但在使用深度学习技术识别不适当内容方面，对于南亚语言的研究较少。在乌尔都语中，拼写并不唯一，人们经常为同一个单词使用不同的常见拼写，并在文本中混合其他语言（如英语），这使得处理这种语言变得更加复杂，相关的研究成果也比较有限。利用注意力层与深度学习模型结合可以有效处理长期依赖问题，并提高模型的效率。为了探索注意力层的效果，本研究提出了一种基于注意力机制的双向GRU混合模型，用于识别乌尔都语Unicode文本中的不适当内容。使用了四种不同的基线深度学习模型：LSTM、Bi-LSTM、GRU和TCN，用于对比所提出模型的性能。通过评估指标、数据集大小以及词嵌入层的影响来比较这些模型的表现。在我们的实验中，使用了预先训练好的乌尔都语word2Vec嵌入。我们所提出模型BiGRU-A在不使用预先训练好的word2Vec层的情况下，达到了84%的准确率，优于所有基线模型。实验结果表明，注意力层可以提高模型的效率，而预先训练的word2Vec嵌入层并不适用于不适当内容的数据集。 

---
# Comparative Insights from 12 Machine Learning Models in Extracting Economic Ideology from Political Text 

**Title (ZH)**: 来自12种机器学习模型在提取政治文本中的经济意识形态方面的比较洞察 

**Authors**: Jihed Ncib  

**Link**: [PDF](https://arxiv.org/pdf/2501.09719)  

**Abstract**: This study conducts a systematic assessment of the capabilities of 12 machine learning models and model variations in detecting economic ideology. As an evaluation benchmark, I use manifesto data spanning six elections in the United Kingdom and pre-annotated by expert and crowd coders. The analysis assesses the performance of several generative, fine-tuned, and zero-shot models at the granular and aggregate levels. The results show that generative models such as GPT-4o and Gemini 1.5 Flash consistently outperform other models against all benchmarks. However, they pose issues of accessibility and resource availability. Fine-tuning yielded competitive performance and offers a reliable alternative through domain-specific optimization. But its dependency on training data severely limits scalability. Zero-shot models consistently face difficulties with identifying signals of economic ideology, often resulting in negative associations with human coding. Using general knowledge for the domain-specific task of ideology scaling proved to be unreliable. Other key findings include considerable within-party variation, fine-tuning benefiting from larger training data, and zero-shot's sensitivity to prompt content. The assessments include the strengths and limitations of each model and derive best-practices for automated analyses of political content. 

**Abstract (ZH)**: 本研究对12种机器学习模型及其变种在识别经济意识形态方面的能力进行了系统的评估。作为评估基准，我使用了涵盖英国六次选举的政纲数据，并由专家和众包编码人员进行了预标注。分析在颗粒度和汇总层面评估了多种生成、微调和零样本模型的性能。结果表明，生成模型如GPT-4o和Gemini 1.5 Flash在所有基准测试中表现一致优异。然而，这些模型面临着可访问性和资源可用性的挑战。微调模型获得了竞争性的性能，通过领域特定优化提供了可靠的替代方案。但其对训练数据的依赖性严重限制了其可扩展性。零样本模型在识别经济意识形态的信号方面始终存在困难，往往与人类编码产生负面关联。使用一般知识进行特定领域的意识形态扩展证明不靠谱。其他主要发现包括政党内部显著的差异性、通过更大规模的训练数据受益于微调、以及零样本模型对提示内容的高度敏感性。评估包括每种模型的优势和局限性，并推导出自动分析政治内容的最佳实践。 

---
# Domain Adaptation of Foundation LLMs for e-Commerce 

**Title (ZH)**: 基于电子商务领域的基础大语言模型的领域适应研究 

**Authors**: Christian Herold, Michael Kozielski, Tala Bazazo, Pavel Petrushkov, Hadi Hashemi, Patrycja Cieplicka, Dominika Basaj, Shahram Khadivi  

**Link**: [PDF](https://arxiv.org/pdf/2501.09706)  

**Abstract**: We present the e-Llama models: 8 billion and 70 billion parameter large language models that are adapted towards the e-commerce domain. These models are meant as foundation models with deep knowledge about e-commerce, that form a base for instruction- and fine-tuning. The e-Llama models are obtained by continuously pretraining the Llama 3.1 base models on 1 trillion tokens of domain-specific data.
We discuss our approach and motivate our choice of hyperparameters with a series of ablation studies. To quantify how well the models have been adapted to the e-commerce domain, we define and implement a set of multilingual, e-commerce specific evaluation tasks.
We show that, when carefully choosing the training setup, the Llama 3.1 models can be adapted towards the new domain without sacrificing significant performance on general domain tasks. We also explore the possibility of merging the adapted model and the base model for a better control of the performance trade-off between domains. 

**Abstract (ZH)**: 我们介绍了e-Llama模型：这是两个大型语言模型，分别拥有80亿和700亿参数，经过调整适用于电子商务领域。这些模型旨在成为基础模型，在电子商务领域具有深厚的专门知识，可用于指令调整和微调。e-Llama模型通过连续在特定领域数据的1万亿个标记上预训练LLaMA 3.1基础模型而获得。

我们通过一系列消融研究讨论了我们的方法，并解释了选择超参数的理由。为了量化模型在电子商务领域的适应程度，我们定义并实现了一系列多语言的电子商务特定评估任务。

研究表明，通过精心选择训练设置，可以从不牺牲通用领域任务性能的角度将LLaMA 3.1模型调整到新领域。我们还探讨了将调整后的模型与基础模型合并的可能性，以更好地控制不同领域之间的性能权衡。 

---
# The Heap: A Contamination-Free Multilingual Code Dataset for Evaluating Large Language Models 

**Title (ZH)**: 堆(heap)：一种无污染的多语言代码数据集，用于评估大规模语言模型 

**Authors**: Jonathan Katzy, Razvan Mihai Popescu, Arie van Deursen, Maliheh Izadi  

**Link**: [PDF](https://arxiv.org/pdf/2501.09653)  

**Abstract**: The recent rise in the popularity of large language models has spurred the development of extensive code datasets needed to train them. This has left limited code available for collection and use in the downstream investigation of specific behaviors, or evaluation of large language models without suffering from data contamination. To address this problem, we release The Heap, a large multilingual dataset covering 57 programming languages that has been deduplicated with respect to other open datasets of code, enabling researchers to conduct fair evaluations of large language models without significant data cleaning overhead. 

**Abstract (ZH)**: 近年来，大型语言模型的流行度大幅提升，推动了用于训练它们的大量代码数据集的发展。这导致可用于下游特定行为研究或大型语言模型评估的代码集合相对有限，且缺乏数据污染。为了解决这一问题，我们发布了《Heap》，这是一个包含57种编程语言的大规模多语言代码数据集，并已在与其他开放代码数据集去重的基础上构建，使研究人员能够在无需大量数据清理工作的情况下公平评估大型语言模型。 

---
# From Scarcity to Capability: Empowering Fake News Detection in Low-Resource Languages with LLMs 

**Title (ZH)**: 从稀缺到能力：利用大语言模型在低资源语言中赋能假新闻检测 

**Authors**: Hrithik Majumdar Shibu, Shrestha Datta, Md. Sumon Miah, Nasrullah Sami, Mahruba Sharmin Chowdhury, Md. Saiful Islam  

**Link**: [PDF](https://arxiv.org/pdf/2501.09604)  

**Abstract**: The rapid spread of fake news presents a significant global challenge, particularly in low-resource languages like Bangla, which lack adequate datasets and detection tools. Although manual fact-checking is accurate, it is expensive and slow to prevent the dissemination of fake news. Addressing this gap, we introduce BanFakeNews-2.0, a robust dataset to enhance Bangla fake news detection. This version includes 11,700 additional, meticulously curated fake news articles validated from credible sources, creating a proportional dataset of 47,000 authentic and 13,000 fake news items across 13 categories. In addition, we created a manually curated independent test set of 460 fake and 540 authentic news items for rigorous evaluation. We invest efforts in collecting fake news from credible sources and manually verified while preserving the linguistic richness. We develop a benchmark system utilizing transformer-based architectures, including fine-tuned Bidirectional Encoder Representations from Transformers variants (F1-87\%) and Large Language Models with Quantized Low-Rank Approximation (F1-89\%), that significantly outperforms traditional methods. BanFakeNews-2.0 offers a valuable resource to advance research and application in fake news detection for low-resourced languages. We publicly release our dataset and model on Github to foster research in this direction. 

**Abstract (ZH)**: 假新闻的迅速传播构成了一个重要的全球性挑战，尤其是在像孟加拉语这样的低资源语言中尤为突出，这些语言缺乏足够的数据集和检测工具。尽管人工事实核查很准确，但它成本高昂且速度慢，难以预防假新闻的传播。为应对这一缺口，我们介绍了BanFakeNews-2.0，这是一个增强孟加拉语假新闻检测的数据集。该版本新增了11,700篇经过仔细筛选和验证的假新闻文章，来自可信的来源，从而形成了一个同比例的数据集，包含47,000条真实新闻和13,000条假新闻，涵盖13个类别。此外，我们还创建了一个包含460条假新闻和540条真实新闻项目的独立手动筛选测试集，以进行严格的评估。我们在收集假新闻时，特别注重从可信来源获取，并进行人工验证，同时保持语言的丰富性。我们开发了一个基准系统，利用基于变换器的架构，包括微调双向编码器表示（F1-87%）的变换器变体和量化低秩近似的大语言模型（F1-89%），这些模型显著优于传统方法。BanFakeNews-2.0为低资源语言中的假新闻检测研究和应用提供了宝贵的资源。我们已在GitHub上公开发布了我们的数据集和模型，以促进该领域的研究。 

---
# Stylomech: Unveiling Authorship via Computational Stylometry in English and Romanized Sinhala 

**Title (ZH)**: Stylomech：通过计算文体学揭示作者身份——以英语和罗马化僧伽罗语为例 

**Authors**: Nabeelah Faumi, Adeepa Gunathilake, Benura Wickramanayake, Deelaka Dias, TGDK Sumanathilaka  

**Link**: [PDF](https://arxiv.org/pdf/2501.09561)  

**Abstract**: With the advent of Web 2.0, the development in social technology coupled with global communication systematically brought positive and negative impacts to society. Copyright claims and Author identification are deemed crucial as there has been a considerable amount of increase in content violation owing to the lack of proper ethics in society. The Author's attribution in both English and Romanized Sinhala became a major requirement in the last few decades. As an area largely unexplored, particularly within the context of Romanized Sinhala, the research contributes significantly to the field of computational linguistics. The proposed author attribution system offers a unique approach, allowing for the comparison of only two sets of text: suspect author and anonymous text, a departure from traditional methodologies which often rely on larger corpora. This work focuses on using the numerical representation of various pairs of the same and different authors allowing for, the model to train on these representations as opposed to text, this allows for it to apply to a multitude of authors and contexts, given that the suspected author text, and the anonymous text are of reasonable quality. By expanding the scope of authorship attribution to encompass diverse linguistic contexts, the work contributes to fostering trust and accountability in digital communication, especially in Sri Lanka. This research presents a pioneering approach to author attribution in both English and Romanized Sinhala, addressing a critical need for content verification and intellectual property rights enforcement in the digital age. 

**Abstract (ZH)**: 随着Web 2.0的到来，社会技术的发展和全球通信的系统性影响在社会上带来了积极和消极的影响。版权主张和作者识别变得至关重要，因为由于社会缺乏适当的伦理规范，内容违规行为显著增加。近年来，作者的归属在英语和罗马化僧伽罗语中成为了一个主要需求。作为在罗马化僧伽罗语背景下尚未充分探索的领域，本研究对计算语言学领域做出了重要贡献。所提议的作者归属系统提供了独特的途径，仅比较涉疑作者和匿名文本两组文本，与传统方法依赖大量语料库不同。该研究工作侧重于利用各种相同和不同作者的数值表示，使模型可以基于这些表示而非文本进行训练，从而能够在涉疑作者文本和匿名文本质量合理的条件下适用于多种作者和背景。通过将作者身份归属的范围扩展到涵盖多样的语言环境，该研究工作有助于在数字通信中培养信任和问责制，特别是在斯里兰卡。本研究展示了在英语和罗马化僧伽罗语中进行作者归属的开创性方法，以满足在数字时代内容验证和知识产权保护方面的需求。 

---
# Analyzing Continuous Semantic Shifts with Diachronic Word Similarity Matrices 

**Title (ZH)**: 使用历时词相似矩阵分析连续语义变化 

**Authors**: Hajime Kiyama, Taichi Aida, Mamoru Komachi, Toshinobu Ogiso, Hiroya Takamura, Daichi Mochihashi  

**Link**: [PDF](https://arxiv.org/pdf/2501.09538)  

**Abstract**: The meanings and relationships of words shift over time. This phenomenon is referred to as semantic this http URL focused on understanding how semantic shifts occur over multiple time periods is essential for gaining a detailed understanding of semantic this http URL, detecting change points only between adjacent time periods is insufficient for analyzing detailed semantic shifts, and using BERT-based methods to examine word sense proportions incurs a high computational this http URL address those issues, we propose a simple yet intuitive framework for how semantic shifts occur over multiple time periods by leveraging a similarity matrix between the embeddings of the same word through this http URL compute a diachronic word similarity matrix using fast and lightweight word embeddings across arbitrary time periods, making it deeper to analyze continuous semantic this http URL, by clustering the similarity matrices for different words, we can categorize words that exhibit similar behavior of semantic shift in an unsupervised manner. 

**Abstract (ZH)**: 词语的意义和关系会随时间而变化，这种现象称为语义演变（semantic change）。理解词语意义随时间演变的过程对于获得对语义演变的深入理解至关重要。仅在相邻时间段之间检测变化点是不充分的，因为这无法全面分析语义演变。使用基于BERT的方法来检查词义比例会带来高计算成本。为了应对这些问题，我们提出了一种简单直观的框架，通过利用相同词语的嵌入向量之间的相似度矩阵来理解词语意义在多个时间段内的演变过程。我们使用轻量级的快速词嵌入在任意时间段内计算历时词相似度矩阵，这使得连续语义演变的分析更加深入。通过聚类不同词语的相似度矩阵，我们可以以无监督的方式对表现出类似语义演变行为的词语进行分类。 

---
# PIER: A Novel Metric for Evaluating What Matters in Code-Switching 

**Title (ZH)**: PIER：一种新型的代码切换关键因素评估指标 

**Authors**: Enes Yavuz Ugan, Ngoc-Quan Pham, Leonard Bärmann, Alex Waibel  

**Link**: [PDF](https://arxiv.org/pdf/2501.09512)  

**Abstract**: Code-switching, the alternation of languages within a single discourse, presents a significant challenge for Automatic Speech Recognition. Despite the unique nature of the task, performance is commonly measured with established metrics such as Word-Error-Rate (WER). However, in this paper, we question whether these general metrics accurately assess performance on code-switching. Specifically, using both Connectionist-Temporal-Classification and Encoder-Decoder models, we show fine-tuning on non-code-switched data from both matrix and embedded language improves classical metrics on code-switching test sets, although actual code-switched words worsen (as expected). Therefore, we propose Point-of-Interest Error Rate (PIER), a variant of WER that focuses only on specific words of interest. We instantiate PIER on code-switched utterances and show that this more accurately describes the code-switching performance, showing huge room for improvement in future work. This focused evaluation allows for a more precise assessment of model performance, particularly in challenging aspects such as inter-word and intra-word code-switching. 

**Abstract (ZH)**: 代码转换，即在单个话语中交替使用多种语言，给自动语音识别（ASR）带来了显著的挑战。尽管这一任务具有独特性，性能通常还是用现有的指标，如词错误率（WER）等进行衡量。然而，在本文中，我们质疑这些通用指标是否准确评估了代码转换任务的性能。具体而言，我们使用连接主义时序分类（Connectionist Temporal Classification, CTC）和编码-解码（Encoder-Decoder）模型进行研究，结果显示，针对矩阵语言和嵌入语言的非代码转换数据进行微调可以提高代码转换测试集上的经典指标，尽管实际的代码转换词汇表现有所下降（正如预期）。因此，我们提出了一种名为兴趣点错误率（Point-of-Interest Error Rate, PIER）的新指标，这是一种改进后的WER，专注于特定的兴趣词汇。我们对代码转换音节实例化PIER，并展示了这一方法更准确地描述了代码转换的性能，表明在未来的改进空间很大。这种聚焦评估使得对模型性能进行更精确的评估成为可能，特别是在诸如跨词和同词内代码转换等具有挑战性的方面。 

---
# Exploring the Inquiry-Diagnosis Relationship with Advanced Patient Simulators 

**Title (ZH)**: 探索基于高级患者模拟器的 inquiry-diagnosis 关系 

**Authors**: Zhaocheng Liu, Quan Tu, Wen Ye, Yu Xiao, Zhishou Zhang, Hengfu Cui, Yalun Zhu, Qiang Ju, Shizheng Li, Jian Xie  

**Link**: [PDF](https://arxiv.org/pdf/2501.09484)  

**Abstract**: Online medical consultation (OMC) restricts doctors to gathering patient information solely through inquiries, making the already complex sequential decision-making process of diagnosis even more challenging. Recently, the rapid advancement of large language models has demonstrated a significant potential to transform OMC. However, most studies have primarily focused on improving diagnostic accuracy under conditions of relatively sufficient information, while paying limited attention to the "inquiry" phase of the consultation process. This lack of focus has left the relationship between "inquiry" and "diagnosis" insufficiently explored. In this paper, we first extract real patient interaction strategies from authentic doctor-patient conversations and use these strategies to guide the training of a patient simulator that closely mirrors real-world behavior. By inputting medical records into our patient simulator to simulate patient responses, we conduct extensive experiments to explore the relationship between "inquiry" and "diagnosis" in the consultation process. Experimental results demonstrate that inquiry and diagnosis adhere to the Liebig's law: poor inquiry quality limits the effectiveness of diagnosis, regardless of diagnostic capability, and vice versa. Furthermore, the experiments reveal significant differences in the inquiry performance of various models. To investigate this phenomenon, we categorize the inquiry process into four types: (1) chief complaint inquiry; (2) specification of known symptoms; (3) inquiry about accompanying symptoms; and (4) gathering family or medical history. We analyze the distribution of inquiries across the four types for different models to explore the reasons behind their significant performance differences. We plan to open-source the weights and related code of our patient simulator at this https URL. 

**Abstract (ZH)**: 在线医疗咨询（OMC）限制医生仅通过询问来收集患者信息，使得原本复杂的诊断顺序决策过程变得更加复杂。近年来，大型语言模型的迅速发展展示了其在OMC方面巨大的潜力。然而，大多数研究主要集中在提高诊断准确性上，尤其是在信息相对充足的情况下，对咨询过程中的“询问”阶段关注较少。这种关注不足导致了“询问”与“诊断”之间关系的探索不够充分。在本文中，我们首先从真实的医生-患者对话中提取患者的互动策略，并利用这些策略指导训练一个模拟患者模型，使其接近真实世界的行为。通过输入医疗记录来模拟患者的反应，我们进行了大量实验以探索咨询过程中“询问”与“诊断”的关系。实验结果表明，“询问”与“诊断”遵循Liebig定律：低质量的“询问”会限制诊断的有效性，无论诊断能力如何，反之亦然。此外，实验揭示了不同模型在“询问”性能上的显著差异。为了探究这一现象，我们将“询问”过程分为四种类型：（1）主要症状询问；（2）具体化已知症状；（3）询问伴随症状；（4）收集家族或医疗史。我们分析了不同模型在四种类型中的询问分布，以探索它们表现差异的原因。我们计划在此网址 <https://example.com> 开放源代码和相关模型权重。

注：上述开放源代码的网址仅为示例，请根据实际情况替换为正确的网址。 

---
# Scaling Graph-Based Dependency Parsing with Arc Vectorization and Attention-Based Refinement 

**Title (ZH)**: 基于图的依赖句法解析的弧向量化与注意力强化 refined

更详细的翻译可以是：

基于图的依赖句法解析的弧向量化解码与注意力机制增强精炼

这个标题翻译尽量保持了原文的专业性和准确性，同时符合中文的学术表达习惯。 

**Authors**: Nicolas Floquet, Joseph Le Roux, Nadi Tomeh, Thierry Charnois  

**Link**: [PDF](https://arxiv.org/pdf/2501.09451)  

**Abstract**: We propose a novel architecture for graph-based dependency parsing that explicitly constructs vectors, from which both arcs and labels are scored. Our method addresses key limitations of the standard two-pipeline approach by unifying arc scoring and labeling into a single network, reducing scalability issues caused by the information bottleneck and lack of parameter sharing. Additionally, our architecture overcomes limited arc interactions with transformer layers to efficiently simulate higher-order dependencies. Experiments on PTB and UD show that our model outperforms state-of-the-art parsers in both accuracy and efficiency. 

**Abstract (ZH)**: 我们提出了一种基于图的依赖解析的新架构，该架构明确构建了向量，从这些向量中对连接和标签进行评分。我们的方法通过将连接评分和标记统一到单个网络中，解决了标准两步管道方法的关键局限性，减少了由于信息瓶颈和参数共享不足引起的可扩展性问题。此外，我们的架构通过使用变压器层高效模拟高阶依赖关系，克服了有限的连接交互问题。实验结果表明，我们的模型在PTB和UD数据集上在准确性和效率上均优于当前最先进的解析器。 

---
# Solving the unsolvable: Translating case law in Hong Kong 

**Title (ZH)**: 解决看似无法解决的问题：香港判例法的翻译 

**Authors**: King-kui Sin, Xi Xuan, Chunyu Kit, Clara Ho-yan Chan, Honic Ho-kin Ip  

**Link**: [PDF](https://arxiv.org/pdf/2501.09444)  

**Abstract**: This paper addresses the challenges translating case law under Hong Kong's bilingual legal system. It highlights the initial success of translating all written statutes into Chinese before the 1997 handover, a task mandated by the Basic Law. The effort involved significant collaboration among legal, linguistic, and translation experts, resulting in a comprehensive and culturally appropriate bilingual legal system. However, translating case law remains a significant challenge due to the sheer volume and continuous growth of judicial decisions. The paper critiques the governments and judiciarys sporadic and uncoordinated efforts to translate case law, contrasting it with the thorough approach previously taken for statute translation. Although the government acknowledges the importance of legal bilingualism, it lacks a sustainable strategy for translating case law. The Judiciarys position that translating all judgments is unnecessary, unrealistic, and not cost-effectiveis analyzed and critiqued for its impact on legal transparency and public trust. A proposed solution involves leveraging machine translation technology through a human-machine interactive translation platform, which undergoes two major transitions. Initially based on a neural model, the platform transitions to using a large language model for improved translation accuracy. Furthermore, it evolves from a single-agent system to a multi-agent system, incorporating Translator, Annotator, and Proofreader agents. This multi-agent approach, supported by a grant, aims to facilitate efficient, high-quality translation of judicial judgments by integrating advanced artificial intelligence and continuous feedback mechanisms, thus better meeting the needs of a bilingual legal system. 

**Abstract (ZH)**: 本文探讨了在港台双语法律体系下翻译判例法所面临的挑战。文章回顾了1997年主权移交前，所有书面立法均已被译成中文的初步成功，这一任务由基本法要求执行。此过程中，法律、语言学和翻译专家的合作至关重要，形成了全面且符合文化特点的双语法律体系。然而，由于判例数量庞大且持续增长，翻译判例法仍是一个重大挑战。文章批评政府和司法机构在翻译判例法方面缺乏系统性和协调性，对抗基本法立法时采取的全面且彻底的方法提出了质疑。尽管政府认识到双语法律的重要性，但在制定长期策略以翻译判例法方面仍显不足。司法部门认为，全部翻译判决是不必要的、不现实的，并且不具有成本效益，这种观点被分析并认为这将影响法律透明度和公众信任。本文提出了一个解决方案，即利用机器翻译技术并通过人机交互平台进行翻译。该平台经历了两大转变：最初依赖神经网络模型，随后转向使用大规模语言模型以提高翻译准确性；进一步从单链条系统发展为多链条系统，引入翻译者、注释者和校对者等多个代理。这种多代理系统旨在通过经费支持下的先进人工智能与连续反馈机制相结合的方式，促进高效、高质量的司法判决翻译，更好地满足双语法律体系的需求。 

---
# AutoCBT: An Autonomous Multi-agent Framework for Cognitive Behavioral Therapy in Psychological Counseling 

**Title (ZH)**: 自动认知行为疗法（AutoCBT）：一种用于心理辅导的认知行为疗法自主多智能体框架 

**Authors**: Ancheng Xu, Di Yang, Renhao Li, Jingwei Zhu, Minghuan Tan, Min Yang, Wanxin Qiu, Mingchen Ma, Haihong Wu, Bingyu Li, Feng Sha, Chengming Li, Xiping Hu, Qiang Qu, Derek F.Wong, Ruifeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2501.09426)  

**Abstract**: Traditional in-person psychological counseling remains primarily niche, often chosen by individuals with psychological issues, while online automated counseling offers a potential solution for those hesitant to seek help due to feelings of shame. Cognitive Behavioral Therapy (CBT) is an essential and widely used approach in psychological counseling. The advent of large language models (LLMs) and agent technology enables automatic CBT diagnosis and treatment. However, current LLM-based CBT systems use agents with a fixed structure, limiting their self-optimization capabilities, or providing hollow, unhelpful suggestions due to redundant response patterns. In this work, we utilize Quora-like and YiXinLi single-round consultation models to build a general agent framework that generates high-quality responses for single-turn psychological consultation scenarios. We use a bilingual dataset to evaluate the quality of single-response consultations generated by each framework. Then, we incorporate dynamic routing and supervisory mechanisms inspired by real psychological counseling to construct a CBT-oriented autonomous multi-agent framework, demonstrating its general applicability. Experimental results indicate that AutoCBT can provide higher-quality automated psychological counseling services. 

**Abstract (ZH)**: 传统的面对面心理辅导仍然主要针对有特定需求的人群，通常由遇到心理问题的个人选择；而在线自动化心理辅导为因羞耻感而犹豫不寻求帮助的人提供了一种潜在的解决方案。认知行为疗法（CBT）是心理辅导中广泛应用且至关重要的一种方法。大型语言模型（LLMs）和代理技术的出现使得自动化的CBT诊断和治疗成为可能。然而，当前基于LLM的CBT系统使用具有固定结构的代理，这限制了它们的自我优化能力，或者由于重复的响应模式提供了空洞且无用的建议。在本工作中，我们利用类似于Quora和YiXinLi的一轮咨询模型构建了一个通用的代理框架，用于生成高质量的心理咨询服务单轮场景的响应。我们使用双语数据集评估每个框架生成的单轮咨询质量。随后，我们借鉴实际心理辅导中的动态路由和监督机制，构建了一个面向CBT的自主多代理框架，展示了其普遍适用性。实验结果表明，AutoCBT能够提供更高质量的自动化心理咨询服务。 

---
# mGeNTE: A Multilingual Resource for Gender-Neutral Language and Translation 

**Title (ZH)**: mGeNTE：一种用于中性语言和翻译的多语言资源 

**Authors**: Beatrice Savoldi, Eleonora Cupin, Manjinder Thind, Anne Lauscher, Luisa Bentivogli  

**Link**: [PDF](https://arxiv.org/pdf/2501.09409)  

**Abstract**: Gender-neutral language reflects societal and linguistic shifts towards greater inclusivity by avoiding the implication that one gender is the norm over others. This is particularly relevant for grammatical gender languages, which heavily encode the gender of terms for human referents and over-relies on masculine forms, even when gender is unspecified or irrelevant. Language technologies are known to mirror these inequalities, being affected by a male bias and perpetuating stereotypical associations when translating into languages with extensive gendered morphology. In such cases, gender-neutral language can help avoid undue binary assumptions. However, despite its importance for creating fairer multi- and cross-lingual technologies, inclusive language research remains scarce and insufficiently supported in current resources. To address this gap, we present the multilingual mGeNTe dataset. Derived from the bilingual GeNTE (Piergentili et al., 2023), mGeNTE extends the original corpus to include the English-Italian/German/Spanish language pairs. Since each language pair is English-aligned with gendered and neutral sentences in the target languages, mGeNTE enables research in both automatic Gender-Neutral Translation (GNT) and language modelling for three grammatical gender languages. 

**Abstract (ZH)**: 性别中立语言体现了社会和语言向着更大包容性的转变，通过避免暗示某一性别是其他性别的标准，从而实现这一点。这对于使用性别的文法语言尤其重要，这类语言高度编码了人类指代词的性别，并且经常过度依赖男性形式，即使性别未指定或无关。语言技术众所周知会反映出这些不平等，会受到男性偏见的影响，并在翻译成具有广泛性态形态的语言时维持刻板印象的关联。在这种情况下，性别中立语言可以帮助避免不适当的二分假设。然而，尽管性别中立语言对于创建更公平的跨lingual技术至关重要，但相关研究仍然稀缺且在当前资源中支持不足。为填补这一缺口，我们提出了多语言mGeNTe数据集。该数据集源自双语GeNTE（Piergentili等人，2023），mGeNTE将原始语料库扩展至包含英语-意大利语/德语/西班牙语等语言对。由于每对语言都是以英语对齐，并包括目标语言中的性别化和中性句子，mGeNTE使得在三种语法性别语言中的自动性别中立翻译（GNT）和语言模型研究成为可能。 

---
# Evaluating LLM Abilities to Understand Tabular Electronic Health Records: A Comprehensive Study of Patient Data Extraction and Retrieval 

**Title (ZH)**: 评估大型语言模型理解电子健康记录表的能力：关于患者数据提取与检索的全面研究 

**Authors**: Jesus Lovon, Martin Mouysset, Jo Oleiwan, Jose G. Moreno, Christine Damase-Michel, Lynda Tamine  

**Link**: [PDF](https://arxiv.org/pdf/2501.09384)  

**Abstract**: Electronic Health Record (EHR) tables pose unique challenges among which is the presence of hidden contextual dependencies between medical features with a high level of data dimensionality and sparsity. This study presents the first investigation into the abilities of LLMs to comprehend EHRs for patient data extraction and retrieval. We conduct extensive experiments using the MIMICSQL dataset to explore the impact of the prompt structure, instruction, context, and demonstration, of two backbone LLMs, Llama2 and Meditron, based on task performance. Through quantitative and qualitative analyses, our findings show that optimal feature selection and serialization methods can enhance task performance by up to 26.79% compared to naive approaches. Similarly, in-context learning setups with relevant example selection improve data extraction performance by 5.95%. Based on our study findings, we propose guidelines that we believe would help the design of LLM-based models to support health search. 

**Abstract (ZH)**: 电子健康记录（EHR）表在数据维度和稀疏性方面具有较高的特征，其中隐藏的上下文依赖关系是一个独特挑战。本研究首次探讨了大型语言模型（LLMs）在理解EHR以提取和检索病人数据方面的能力。我们使用MIMICSQL数据集进行大量的实验，以研究两种主干模型——Llama2和Meditron——在基于任务性能的不同提示结构、指令、上下文和示范的影响。通过定量和定性分析，我们的研究发现，最优特征选择和序列化方法可以将任务性能提高多达26.79%，相较于朴素方法。同样，具有相关示例选择的上下文学习设置可以将数据提取性能提高5.95%。基于我们的研究结果，我们提出了一些建议，我们相信这些建议将有助于设计支持健康搜索的LLM基模型。 

---
# ChartInsighter: An Approach for Mitigating Hallucination in Time-series Chart Summary Generation with A Benchmark Dataset 

**Title (ZH)**: ChartInsighter：一种减轻时间序列图表摘要生成中幻觉的方法及其基准数据集 

**Authors**: Fen Wang, Bomiao Wang, Xueli Shu, Zhen Liu, Zekai Shao, Chao Liu, Siming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.09349)  

**Abstract**: Effective chart summary can significantly reduce the time and effort decision makers spend interpreting charts, enabling precise and efficient communication of data insights. Previous studies have faced challenges in generating accurate and semantically rich summaries of time-series data charts. In this paper, we identify summary elements and common hallucination types in the generation of time-series chart summaries, which serve as our guidelines for automatic generation. We introduce ChartInsighter, which automatically generates chart summaries of time-series data, effectively reducing hallucinations in chart summary generation. Specifically, we assign multiple agents to generate the initial chart summary and collaborate iteratively, during which they invoke external data analysis modules to extract insights and compile them into a coherent summary. Additionally, we implement a self-consistency test method to validate and correct our summary. We create a high-quality benchmark of charts and summaries, with hallucination types annotated on a sentence-by-sentence basis, facilitating the evaluation of the effectiveness of reducing hallucinations. Our evaluations using our benchmark show that our method surpasses state-of-the-art models, and that our summary hallucination rate is the lowest, which effectively reduces various hallucinations and improves summary quality. The benchmark is available at this https URL. 

**Abstract (ZH)**: 有效的图表总结可以显著减少决策者在解读图表时所需的时间和精力，从而实现数据洞察的精确和高效沟通。先前的研究在生成时间序列数据图表的准确和语义丰富的摘要方面遇到了挑战。本文中，我们明确了时间序列图表总结中的摘要元素和常见的幻觉类型，这些作为我们自动生成的指南。我们引入了ChartInsighter，它可以自动生成时间序列数据的图表总结，有效地减少了图表总结生成中的幻觉。具体来说，我们分配多个代理生成初始图表总结，并在迭代过程中协作，其间它们调用外部数据分析模块以提取洞察并将其汇总为一个连贯的摘要。此外，我们实施了一种自我一致性测试方法来验证和纠正我们的总结。我们创建了一个高质量的图表和总结基准，其中对每个句子的幻觉类型进行了标注，便于评估减少幻觉的有效性。我们的基准测试显示，我们的方法超越了现有的最佳模型，我们的摘要幻觉率最低，有效地减少了各种幻觉并提高了摘要质量。该基准已发布在 <https://this_is_a_dummy_url>。 

---
# Algorithm for Semantic Network Generation from Texts of Low Resource Languages Such as Kiswahili 

**Title (ZH)**: 低资源语言（如斯瓦希里语）文本的语义网络生成算法 

**Authors**: Barack Wamkaya Wanjawa, Lawrence Muchemi, Evans Miriti  

**Link**: [PDF](https://arxiv.org/pdf/2501.09326)  

**Abstract**: Processing low-resource languages, such as Kiswahili, using machine learning is difficult due to lack of adequate training data. However, such low-resource languages are still important for human communication and are already in daily use and users need practical machine processing tasks such as summarization, disambiguation and even question answering (QA). One method of processing such languages, while bypassing the need for training data, is the use semantic networks. Some low resource languages, such as Kiswahili, are of the subject-verb-object (SVO) structure, and similarly semantic networks are a triple of subject-predicate-object, hence SVO parts of speech tags can map into a semantic network triple. An algorithm to process raw natural language text and map it into a semantic network is therefore necessary and desirable in structuring low resource languages texts. This algorithm tested on the Kiswahili QA task with upto 78.6% exact match. 

**Abstract (ZH)**: 使用机器学习处理低资源语言（如斯瓦希里语）是一项具有挑战性的工作，主要是由于缺乏足够的训练数据。然而，这些低资源语言在人类交流中仍然非常重要，并已被日常使用。用户需要诸如摘要、消歧和甚至问答（QA）等实际的机器处理任务。一种处理这些语言的方法是使用语义网络，从而无需训练数据。一些低资源语言，如斯瓦希里语，具有主语-谓语-宾语（SVO）结构，同样的，语义网络是一个三元组（主语-谓词-宾语），因此SVO句法标注可以映射到语义网络三元组。因此，一种能够处理原始自然语言文本并将其映射到语义网络的算法对其结构化尤为必要。该算法在斯瓦希里语问答任务上的测试结果显示，其准确匹配率最高可达78.6%。 

---
# A Study of In-Context-Learning-Based Text-to-SQL Errors 

**Title (ZH)**: 基于上下文学习的文本到SQL错误研究 

**Authors**: Jiawei Shen, Chengcheng Wan, Ruoyi Qiao, Jiazhen Zou, Hang Xu, Yuchen Shao, Yueling Zhang, Weikai Miao, Geguang Pu  

**Link**: [PDF](https://arxiv.org/pdf/2501.09310)  

**Abstract**: Large language models (LLMs) have been adopted to perform text-to-SQL tasks, utilizing their in-context learning (ICL) capability to translate natural language questions into structured query language (SQL). However, such a technique faces correctness problems and requires efficient repairing solutions. In this paper, we conduct the first comprehensive study of text-to-SQL errors. Our study covers four representative ICL-based techniques, five basic repairing methods, two benchmarks, and two LLM settings. We find that text-to-SQL errors are widespread and summarize 29 error types of 7 categories. We also find that existing repairing attempts have limited correctness improvement at the cost of high computational overhead with many mis-repairs. Based on the findings, we propose MapleRepair, a novel text-to-SQL error detection and repairing framework. The evaluation demonstrates that MapleRepair outperforms existing solutions by repairing 13.8% more queries with neglectable mis-repairs and 67.4% less overhead. 

**Abstract (ZH)**: 大型语言模型（LLMs）已被用于执行文本到SQL任务，并利用其上下文学习（ICL）能力将自然语言问题转换为结构化查询语言（SQL）。然而，这种方法面临着正确性问题，并需要高效的修复方案。本文旨在进行第一次全面的文本到SQL错误研究。我们的研究涵盖了四种代表性ICL技术、五种基本修复方法、两个基准和两种LLM设置。我们发现文本到SQL错误普遍存在，并总结了7类29种错误类型。我们还发现，现有的修复尝试在增加正确性的同时，面临着高计算开销和大量误修复的问题。基于这些发现，我们提出了MapleRepair，一种新的文本到SQL错误检测和修复框架。评估表明，MapleRepair在可忽略的误修复和显著降低67.4%计算开销的情况下，能够修复更多（13.8%）的查询。 

---
# To Retrieve or Not to Retrieve? Uncertainty Detection for Dynamic Retrieval Augmented Generation 

**Title (ZH)**: 取舍之间：动态检索增强生成中的不确定性检测 

**Authors**: Kaustubh D. Dhole  

**Link**: [PDF](https://arxiv.org/pdf/2501.09292)  

**Abstract**: Retrieval-Augmented Generation equips large language models with the capability to retrieve external knowledge, thereby mitigating hallucinations by incorporating information beyond the model's intrinsic abilities. However, most prior works have focused on invoking retrieval deterministically, which makes it unsuitable for tasks such as long-form question answering. Instead, dynamically performing retrieval by invoking it only when the underlying LLM lacks the required knowledge can be more efficient. In this context, we delve deeper into the question, "To Retrieve or Not to Retrieve?" by exploring multiple uncertainty detection methods. We evaluate these methods for the task of long-form question answering, employing dynamic retrieval, and present our comparisons. Our findings suggest that uncertainty detection metrics, such as Degree Matrix Jaccard and Eccentricity, can reduce the number of retrieval calls by almost half, with only a slight reduction in question-answering accuracy. 

**Abstract (ZH)**: 检索增强生成赋予大型语言模型检索外部知识的能力，从而通过引入模型固有能力之外的信息来减轻幻觉现象。然而，大多数先前的工作主要集中在确定性地触发检索上，这使得这种方法不适合长篇问答等任务。相反，只有在底层语言模型缺乏所需知识时才动态地进行检索可能更为高效。在这一背景下，我们深入探讨了“检索还是不检索？”这一问题，探索了多种不确定性检测方法。我们评估了这些方法在长篇问答任务中的表现，采用了动态检索，并展示了我们的比较结果。我们的研究发现，度矩阵Jaccard和离心率等不确定性检测指标可以将检索调用次数减少近一半，同时仅稍微降低问答准确性。 

---
# Perspective Transition of Large Language Models for Solving Subjective Tasks 

**Title (ZH)**: 大型语言模型视角转换在解决主观任务中的应用研究 

**Authors**: Xiaolong Wang, Yuanchi Zhang, Ziyue Wang, Yuzhuang Xu, Fuwen Luo, Yile Wang, Peng Li, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.09265)  

**Abstract**: Large language models (LLMs) have revolutionized the field of natural language processing, enabling remarkable progress in various tasks. Different from objective tasks such as commonsense reasoning and arithmetic question-answering, the performance of LLMs on subjective tasks is still limited, where the perspective on the specific problem plays crucial roles for better interpreting the context and giving proper response. For example, in certain scenarios, LLMs may perform better when answering from an expert role perspective, potentially eliciting their relevant domain knowledge. In contrast, in some scenarios, LLMs may provide more accurate responses when answering from a third-person standpoint, enabling a more comprehensive understanding of the problem and potentially mitigating inherent biases. In this paper, we propose Reasoning through Perspective Transition (RPT), a method based on in-context learning that enables LLMs to dynamically select among direct, role, and third-person perspectives for the best way to solve corresponding subjective problem. Through extensive experiments on totally 12 subjective tasks by using both closed-source and open-source LLMs including GPT-4, GPT-3.5, Llama-3, and Qwen-2, our method outperforms widely used single fixed perspective based methods such as chain-of-thought prompting and expert prompting, highlights the intricate ways that LLMs can adapt their perspectives to provide nuanced and contextually appropriate responses for different problems. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在自然语言处理领域掀起了一场革命，使其在各种任务上取得了显著的进展。不同于常识推理和数学问题解答等客观任务，LLMs在主观任务上的表现仍然有限，特定问题的角度对于更好地解读上下文和给出恰当的回答至关重要。例如，在某些场景中，当LLMs从专家的角度回答时，可能会表现出更好的性能，从而激发其相关领域的知识。相反，在某些场景中，当LLMs从第三人称的角度回答时，可以提供更准确的回答，有助于更全面地理解问题，从而可能减轻固有的偏见。在本文中，我们提出了一种基于上下文学习的方法——视角转换推理（RPT），该方法使LLMs能够动态选择直接、角色和第三人称视角，以最适合的方式解决相应的主观问题。通过在12项完全不同的主观任务上进行广泛的实验，使用包括GPT-4、GPT-3.5、Llama-3和Qwen-2在内的闭源和开源LLMs，我们的方法优于广泛使用的基于单一固定视角的方法，如链式思维提示和专家提示，突显了LLMs能够灵活调整视角以提供细致且上下文适当回答的不同方式。 

---
# Delayed Fusion: Integrating Large Language Models into First-Pass Decoding in End-to-end Speech Recognition 

**Title (ZH)**: 延迟融合：将大型语言模型集成到端到端语音识别的首轮解码中 

**Authors**: Takaaki Hori, Martin Kocour, Adnan Haider, Erik McDermott, Xiaodan Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2501.09258)  

**Abstract**: This paper presents an efficient decoding approach for end-to-end automatic speech recognition (E2E-ASR) with large language models (LLMs). Although shallow fusion is the most common approach to incorporate language models into E2E-ASR decoding, we face two practical problems with LLMs. (1) LLM inference is computationally costly. (2) There may be a vocabulary mismatch between the ASR model and the LLM. To resolve this mismatch, we need to retrain the ASR model and/or the LLM, which is at best time-consuming and in many cases not feasible. We propose "delayed fusion," which applies LLM scores to ASR hypotheses with a delay during decoding and enables easier use of pre-trained LLMs in ASR tasks. This method can reduce not only the number of hypotheses scored by the LLM but also the number of LLM inference calls. It also allows re-tokenizion of ASR hypotheses during decoding if ASR and LLM employ different tokenizations. We demonstrate that delayed fusion provides improved decoding speed and accuracy compared to shallow fusion and N-best rescoring using the LibriHeavy ASR corpus and three public LLMs, OpenLLaMA 3B & 7B and Mistral 7B. 

**Abstract (ZH)**: 本文提出了一个高效解码方法，用于端到端自动语音识别（E2E-ASR）系统，结合了大规模语言模型（LLMs）。尽管浅融合是最常见的将语言模型融入E2E-ASR解码的方法，但在使用LLMs时我们面临两个实际问题：(1) LLM推理计算开销大；(2) ASR模型与LLM之间可能存在词汇表不匹配。为了解决这一不匹配问题，我们需要重新训练ASR模型和/或LLM，这在最理想的情况下耗时长，而在许多情况下是不切实际的。我们提出了一种“延迟融合”的方法，在解码过程中以延迟的方式应用LLM分数，从而在ASR任务中更方便地使用预训练的LLMs。该方法不仅能减少需要LLM评分的假设数量，还能减少LLM推理调用次数。此外，该方法还允许在解码过程中进行ASR假设的重新分词，如果ASR和LLM使用不同的分词方式时。我们使用LibriHeavy ASR语料库以及三个公共LLM（OpenLLaMA 3B & 7B 和 Mistral 7B）证明了延迟融合相比浅融合和N-best评分具有更高的解码速度和准确率。 

---
# Foundations of Large Language Models 

**Title (ZH)**: 大语言模型的基础 

**Authors**: Tong Xiao, Jingbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2501.09223)  

**Abstract**: This is a book about large language models. As indicated by the title, it primarily focuses on foundational concepts rather than comprehensive coverage of all cutting-edge technologies. The book is structured into four main chapters, each exploring a key area: pre-training, generative models, prompting techniques, and alignment methods. It is intended for college students, professionals, and practitioners in natural language processing and related fields, and can serve as a reference for anyone interested in large language models. 

**Abstract (ZH)**: 这是一本关于大型语言模型的书籍。如书名所示，本书主要侧重于基础概念的介绍，而非全面涵盖所有前沿技术。本书结构分为四章，分别探讨了核心领域：预训练、生成模型、提示技术以及对齐方法。本书旨在为自然语言处理领域的大学生、专业人员及从业者提供参考，并可作为对大型语言模型感兴趣的任何人的参考资料。 

---
# A Simple Graph Contrastive Learning Framework for Short Text Classification 

**Title (ZH)**: 一种简单的图形对比学习框架用于短文本分类 

**Authors**: Yonghao Liu, Fausto Giunchiglia, Lan Huang, Ximing Li, Xiaoyue Feng, Renchu Guan  

**Link**: [PDF](https://arxiv.org/pdf/2501.09219)  

**Abstract**: Short text classification has gained significant attention in the information age due to its prevalence and real-world applications. Recent advancements in graph learning combined with contrastive learning have shown promising results in addressing the challenges of semantic sparsity and limited labeled data in short text classification. However, existing models have certain limitations. They rely on explicit data augmentation techniques to generate contrastive views, resulting in semantic corruption and noise. Additionally, these models only focus on learning the intrinsic consistency between the generated views, neglecting valuable discriminative information from other potential views. To address these issues, we propose a Simple graph contrastive learning framework for Short Text Classification (SimSTC). Our approach involves performing graph learning on multiple text-related component graphs to obtain multi-view text embeddings. Subsequently, we directly apply contrastive learning on these embeddings. Notably, our method eliminates the need for data augmentation operations to generate contrastive views while still leveraging the benefits of multi-view contrastive learning. Despite its simplicity, our model achieves outstanding performance, surpassing large language models on various datasets. 

**Abstract (ZH)**: 短文本分类在信息时代由于其广泛的应用和实际需求而受到了广泛关注。近年来，结合图学习和对比学习的进步已经在应对短文本分类中语义稀疏性和标注数据有限的挑战方面显示出有希望的结果。然而，现有的模型存在一些局限性。它们依赖于显式的数据增强技术来生成对比视图，导致语义破坏和噪声。另外，这些模型仅关注生成视图之间的内在一致性，忽视了其他潜在视图中的宝贵判别信息。为了解决这些问题，我们提出了一种用于短文本分类的简单图对比学习框架（SimSTC）。我们的方法涉及在多个文本相关组件图上执行图学习，以获得多视图文本嵌入。随后，我们直接在这些嵌入上应用对比学习。值得注意的是，我们的方法消除了生成对比视图所需的數據增强操作，同时仍利用多视图对比学习的优势。尽管结构简单，我们的模型在各种数据集上的性能出色，超过了大型语言模型。 

---
# Boosting Short Text Classification with Multi-Source Information Exploration and Dual-Level Contrastive Learning 

**Title (ZH)**: 利用多源信息探索与双层对比学习增强短文本分类 

**Authors**: Yonghao Liu, Mengyu Li, Wei Pang, Fausto Giunchiglia, Lan Huang, Xiaoyue Feng, Renchu Guan  

**Link**: [PDF](https://arxiv.org/pdf/2501.09214)  

**Abstract**: Short text classification, as a research subtopic in natural language processing, is more challenging due to its semantic sparsity and insufficient labeled samples in practical scenarios. We propose a novel model named MI-DELIGHT for short text classification in this work. Specifically, it first performs multi-source information (i.e., statistical information, linguistic information, and factual information) exploration to alleviate the sparsity issues. Then, the graph learning approach is adopted to learn the representation of short texts, which are presented in graph forms. Moreover, we introduce a dual-level (i.e., instance-level and cluster-level) contrastive learning auxiliary task to effectively capture different-grained contrastive information within massive unlabeled data. Meanwhile, previous models merely perform the main task and auxiliary tasks in parallel, without considering the relationship among tasks. Therefore, we introduce a hierarchical architecture to explicitly model the correlations between tasks. We conduct extensive experiments across various benchmark datasets, demonstrating that MI-DELIGHT significantly surpasses previous competitive models. It even outperforms popular large language models on several datasets. 

**Abstract (ZH)**: 短文本分类是自然语言处理领域的一个研究子课题，由于其语义稀疏性和实际应用场景中缺乏足够的标注样本，因此更具挑战性。本文提出了一种名为MI-DELIGHT的新模型，用于解决短文本分类问题。具体而言，该模型首先通过多源信息探索（即统计信息、语言信息和事实信息）来缓解语义稀疏性问题。然后，采用图学习方法来学习短文本的表示，这些短文本以图的形式呈现。此外，本文引入了一种双重层次对比学习辅助任务（即实例层次和聚类层次），以有效捕获大规模未标注数据中的不同粒度对比信息。同时，之前的研究模型仅并行执行主任务和辅助任务，而不考虑任务之间的关系。因此，本文引入了层次架构以明确建模任务之间的相关性。我们在多个基准数据集上进行了广泛的实验，结果表明，MI-DELIGHT 显著优于之前的竞争模型。甚至在某些数据集上，它还超过了流行的大型语言模型。 

---
# FineMedLM-o1: Enhancing the Medical Reasoning Ability of LLM from Supervised Fine-Tuning to Test-Time Training 

**Title (ZH)**: FineMedLM-o1：从监督微调到测试时训练，增强大语言模型的医学推理能力 

**Authors**: Hongzhou Yu, Tianhao Cheng, Ying Cheng, Rui Feng  

**Link**: [PDF](https://arxiv.org/pdf/2501.09213)  

**Abstract**: Recent advancements in large language models (LLMs) have shown promise in medical applications such as disease diagnosis and treatment planning. However, most existing medical LLMs struggle with the advanced reasoning required for complex clinical scenarios, such as differential diagnosis or personalized treatment suggestions. We proposed FineMedLM-o1, which leverages high-quality synthetic medical data and long-form reasoning data for Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO), enabling advanced dialogue and deep reasoning capabilities. Additionally, we introduced Test-Time Training (TTT) in the medical domain for the first time, facilitating domain adaptation and ensuring reliable, accurate reasoning. Experimental results demonstrate that FineMedLM-o1 achieves a 23% average performance improvement over prior models on key medical benchmarks. Furthermore, the introduction of TTT provides an additional 14% performance boost, highlighting its effectiveness in enhancing medical reasoning capabilities. To support this process, we also proposed a novel method for synthesizing medical dialogue. Compared to other open-source datasets, our dataset stands out as superior in both quality and complexity. The project and data will be released on GitHub. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）在医学应用方面取得了进展，如疾病诊断和治疗规划。然而，现有的大多数医学LLMs在处理复杂临床场景所需的高级推理方面存在不足，例如鉴别诊断或个性化治疗建议。我们提出了FineMedLM-o1，它利用高质量的合成医学数据和长篇推理数据进行监督微调（SFT）和直接偏好优化（DPO），从而实现高级对话和深入的推理能力。此外，我们首次在医学领域引入了测试时训练（TTT），促进领域适应并确保可靠的准确推理。实验结果表明，FineMedLM-o1在关键医学基准测试上的平均性能改进达到了23%，而TTT的引入又额外提供了14%的性能提升，突显其在增强医学推理能力方面的有效性。为了支持这一过程，我们还提出了一种新的合成医学对话方法。与现有的开源数据集相比，我们的数据集在质量和复杂性方面都更为优越。该项目和数据将发布在GitHub上。 

---
# The Veln(ia)s is in the Details: Evaluating LLM Judgment on Latvian and Lithuanian Short Answer Matching 

**Title (ZH)**: velni（斯）就在细节中：评估大模型对拉脱维亚语和立陶宛语文短答案匹配的判断能力 

**Authors**: Yevhen Kostiuk, Oxana Vitman, Łukasz Gagała, Artur Kiulian  

**Link**: [PDF](https://arxiv.org/pdf/2501.09164)  

**Abstract**: In this work, we address the challenge of evaluating large language models (LLMs) on the short answer matching task for Latvian and Lithuanian languages. We introduce novel datasets consisting of 502 Latvian and 690 Lithuanian question-answer pairs. For each question-answer pair, we generated matched and non-matched answers using a set of alteration rules specifically designed to introduce small but meaningful changes in the text. These generated answers serve as test cases to assess the ability of LLMs to detect subtle differences in matching of the original answers. A subset of the datasets was manually verified for quality and accuracy. Our results show that while larger LLMs, such as QWEN2.5 72b and LLaMa3.1 70b, demonstrate near-perfect performance in distinguishing matched and non-matched answers, smaller models show more variance. For instance, LLaMa3.1 8b and EuroLLM 9b benefited from few-shot examples, while Mistral Nemo 12b underperformed on detection of subtle text alteration, particularly in Lithuanian, even with additional examples. QWEN2.5 7b and Mistral 7b were able to obtain a strong and comparable performance to the larger 70b models in zero and few shot experiments. Moreover, the performance of Mistral 7b was weaker in few shot experiments. 

**Abstract (ZH)**: 在本文中，我们探讨了在拉脱维亚语和立陶宛语的简答匹配任务中评估大型语言模型（LLMs）的挑战。我们引入了新的数据集，其中包括502个拉脱维亚问题-答案对和690个立陶宛问题-答案对。对于每个问题-答案对，我们使用一组特定设计的修改规则生成匹配和不匹配的答案，旨在引入细微但有意义的文本变化。这些生成的答案用作测试案例，以评估LLMs检测原始答案间细微差异的能力。经过手动验证的质量和准确性的子数据集显示，虽然更大规模的LLMs（如QWEN2.5 72b和LLaMa3.1 70b）在区分匹配和不匹配答案方面表现出几乎完美的性能，但较小规模的模型则显示出更大的波动。例如，LLaMa3.1 8b和EuroLLM 9b受益于少量示例，而Mistral Nemo 12b在检测细微文本修改方面表现不佳，特别是在立陶宛语中，即便增加了额外的示例也是如此。QWEN2.5 7b和Mistral 7b在零样本和少量样本实验中能够获得与70b模型相似的强大性能。此外，在少量样本实验中，Mistral 7b的性能较弱。 

---
# Evaluating GenAI for Simplifying Texts for Education: Improving Accuracy and Consistency for Enhanced Readability 

**Title (ZH)**: 评估GenAI在教育中简化文本的应用：提高准确性和一致性以增强可读性 

**Authors**: Stephanie L. Day, Jacapo Cirica, Steven R. Clapp, Veronika Penkova, Amy E. Giroux, Abbey Banta, Catherine Bordeau, Poojitha Mutteneni, Ben D. Sawyer  

**Link**: [PDF](https://arxiv.org/pdf/2501.09158)  

**Abstract**: Generative artificial intelligence (GenAI) holds great promise as a tool to support personalized learning. Teachers need tools to efficiently and effectively enhance content readability of educational texts so that they are matched to individual students reading levels, while retaining key details. Large Language Models (LLMs) show potential to fill this need, but previous research notes multiple shortcomings in current approaches. In this study, we introduced a generalized approach and metrics for the systematic evaluation of the accuracy and consistency in which LLMs, prompting techniques, and a novel multi-agent architecture to simplify sixty informational reading passages, reducing each from the twelfth grade level down to the eighth, sixth, and fourth grade levels. We calculated the degree to which each LLM and prompting technique accurately achieved the targeted grade level for each passage, percentage change in word count, and consistency in maintaining keywords and key phrases (semantic similarity). One-sample t-tests and multiple regression models revealed significant differences in the best performing LLM and prompt technique for each of the four metrics. Both LLMs and prompting techniques demonstrated variable utility in grade level accuracy and consistency of keywords and key phrases when attempting to level content down to the fourth grade reading level. These results demonstrate the promise of the application of LLMs for efficient and precise automated text simplification, the shortcomings of current models and prompting methods in attaining an ideal balance across various evaluation criteria, and a generalizable method to evaluate future systems. 

**Abstract (ZH)**: 生成式人工智能（GenAI）作为个性化学习工具持有巨大潜力。教师需要工具来高效有效地提高教育文本的可读性，使其与学生个体的阅读水平相匹配，同时保留关键细节。大型语言模型（LLMs）显示出满足这一需求的潜力，但先前的研究指出当前方法存在多方面不足。在本研究中，我们引入了一种通用的方法和指标，用于系统地评估LLMs、提示技术以及一种新颖的多代理架构，这些方法和技术简化了六十篇信息性阅读段落，将每个段落的难度等级从高中降低到初中八年级、六年级和四年级。我们计算了每个LLM和提示技术在每篇段落中准确达到目标年级水平的程度、词数变化百分比，以及保持关键词和关键短语（语义相似性）的一致性。单因素t检验和多种线性回归模型表明，在四个指标中，最佳性能的LLM和提示技术存在显著差异。无论是LLM还是提示技术，在试图将内容简化到四年级阅读水平时，都能在年级水平准确性和关键词和关键短语的一致性方面展现出不同的适用性。这些结果表明，在简化文本方面应用LLMs的高效和精准自动化的前景，当前模型和提示方法在满足各种评价标准的综合平衡方面存在的不足，以及评估未来系统的一种通用方法。 

---
# Towards Multilingual LLM Evaluation for Baltic and Nordic languages: A study on Lithuanian History 

**Title (ZH)**: 面向波罗的海和北欧语言的多语言大规模语言模型评估：立陶宛历史研究 

**Authors**: Yevhen Kostiuk, Oxana Vitman, Łukasz Gagała, Artur Kiulian  

**Link**: [PDF](https://arxiv.org/pdf/2501.09154)  

**Abstract**: In this work, we evaluated Lithuanian and general history knowledge of multilingual Large Language Models (LLMs) on a multiple-choice question-answering task. The models were tested on a dataset of Lithuanian national and general history questions translated into Baltic, Nordic, and other languages (English, Ukrainian, Arabic) to assess the knowledge sharing from culturally and historically connected groups. We evaluated GPT-4o, LLaMa3.1 8b and 70b, QWEN2.5 7b and 72b, Mistral Nemo 12b, LLaMa3 8b, Mistral 7b, LLaMa3.2 3b, and Nordic fine-tuned models (GPT-SW3 and LLaMa3 8b).
Our results show that GPT-4o consistently outperformed all other models across language groups, with slightly better results for Baltic and Nordic languages. Larger open-source models like QWEN2.5 72b and LLaMa3.1 70b performed well but showed weaker alignment with Baltic languages. Smaller models (Mistral Nemo 12b, LLaMa3.2 3b, QWEN 7B, LLaMa3.1 8B, and LLaMa3 8b) demonstrated gaps with LT-related alignment with Baltic languages while performing better on Nordic and other languages. The Nordic fine-tuned models did not surpass multilingual models, indicating that shared cultural or historical context alone does not guarantee better performance. 

**Abstract (ZH)**: 在本研究中，我们评估了多语言大型语言模型（LLMs）在多项选择题问答任务中对立陶宛历史和通用历史知识的理解。这些模型在包含立陶宛国家历史和通用历史问题的数据集中进行了测试，这些问题已翻译成波罗的海语言、北欧语言以及英语、乌克兰语和阿拉伯语等其他语言，以评估文化和历史上相互联系的群体之间的知识共享情况。我们测试了以下模型：GPT-4o、LLaMa3.1 8b 和 70b、QWEN2.5 7b 和 72b、Mistral Nemo 12b、LLaMa3 8b、Mistral 7b、LLaMa3.2 3b 以及北欧语言微调模型（GPT-SW3 和 LLaMa3 8b）。

研究结果显示，GPT-4o 在所有语言组中均表现最佳，特别是在波罗的海和北欧语言组中表现稍好。大型开源模型如 QWEN2.5 72b 和 LLaMa3.1 70b 表现良好，但与波罗的海语言的匹配程度较弱。较小的模型（Mistral Nemo 12b、LLaMa3.2 3b、QWEN 7b、LLaMa3.1 8B 和 LLaMa3 8b）在与立陶宛相关的历史背景匹配方面存在差距，但在北欧及其他语言方面表现更好。北欧语言微调模型未能超越多语言模型，这表明仅共享文化和历史背景并不足以保证更好的性能。 

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

**Title (ZH)**: SteLLA：一种结合RAG的结构化评分系统，使用大型语言模型 

**Authors**: Hefei Qiu, Brian White, Ashley Ding, Reinaldo Costa, Ali Hachem, Wei Ding, Ping Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.09092)  

**Abstract**: Large Language Models (LLMs) have shown strong general capabilities in many applications. However, how to make them reliable tools for some specific tasks such as automated short answer grading (ASAG) remains a challenge. We present SteLLA (Structured Grading System Using LLMs with RAG) in which a) Retrieval Augmented Generation (RAG) approach is used to empower LLMs specifically on the ASAG task by extracting structured information from the highly relevant and reliable external knowledge based on the instructor-provided reference answer and rubric, b) an LLM performs a structured and question-answering-based evaluation of student answers to provide analytical grades and feedback. A real-world dataset that contains students' answers in an exam was collected from a college-level Biology course. Experiments show that our proposed system can achieve substantial agreement with the human grader while providing break-down grades and feedback on all the knowledge points examined in the problem. A qualitative and error analysis of the feedback generated by GPT4 shows that GPT4 is good at capturing facts while may be prone to inferring too much implication from the given text in the grading task which provides insights into the usage of LLMs in the ASAG system. 

**Abstract (ZH)**: 大型语言模型（LLMs）在许多应用中展示了强大的通用能力。然而，如何将它们转化为可依赖的工具，用于特定任务如自动简短答案评分（ASAG），仍然是一个挑战。我们提出了一种名为SteLLA（利用LLMs和RAG的结构化评分系统）的方法，其中：
a) 采用检索增强生成（RAG）方法，通过从教师提供的参考答案和评分标准中提取高度相关且可靠的外部结构化信息，特别增强了LLMs在ASAG任务上的性能；
b) 一个LLM执行结构化的基于问题的回答的评估，为学生的答案提供分析性评分和反馈。

从一门大学生物学课程的考试中收集了包含学生答案的真实世界数据集。实验结果显示，我们提出的方法能够与人工评分者在评分上有显著的一致性，并且能够为问题中所有检查的知识点提供细化评分和反馈。对GPT4生成的反馈进行定性和错误分析表明，GPT4擅长捕捉事实，但在评分任务中可能会从给定文本中推断过多的隐含含义，这为在ASAG系统中使用LLMs提供了见解。 

---
# Decompose-ToM: Enhancing Theory of Mind Reasoning in Large Language Models through Simulation and Task Decomposition 

**Title (ZH)**: 分解共情推理：通过模拟和任务分解在大型语言模型中增强理论心智推理 

**Authors**: Sneheel Sarangi, Maha Elgarf, Hanan Salam  

**Link**: [PDF](https://arxiv.org/pdf/2501.09056)  

**Abstract**: Theory of Mind (ToM) is the ability to understand and reflect on the mental states of others. Although this capability is crucial for human interaction, testing on Large Language Models (LLMs) reveals that they possess only a rudimentary understanding of it. Although the most capable closed-source LLMs have come close to human performance on some ToM tasks, they still perform poorly on complex variations of the task that involve more structured reasoning. In this work, we utilize the concept of "pretend-play", or ``Simulation Theory'' from cognitive psychology to propose ``Decompose-ToM'': an LLM-based inference algorithm that improves model performance on complex ToM tasks. We recursively simulate user perspectives and decompose the ToM task into a simpler set of functions: subject identification, question-reframing, world model updation, and knowledge availability. We test the algorithm on higher-order ToM tasks and a task testing for ToM capabilities in a conversational setting, demonstrating that our approach shows significant improvement across models compared to baseline methods while requiring minimal prompt tuning across tasks and no additional model training. 

**Abstract (ZH)**: 心智理论（Theory of Mind，ToM）是指理解并反思他人心理状态的能力。尽管这种能力对于人类交往至关重要，但在对大规模语言模型（Large Language Models，LLMs）进行测试时发现，它们对ToM的理解仅停留在基本层面。尽管最强大的封闭源代码LLMs在某些ToM任务上接近了人类的表现，但在涉及到更多结构化推理的复杂变体任务中，它们的表现仍然不佳。本文中，我们利用认知心理学中的“假装游戏”概念，即“模拟理论”，提出了一种基于LLM的推理算法——“Decompose-ToM”：该算法能够提升模型在复杂ToM任务中的性能。我们递归地模拟用户视角，并将ToM任务分解成一组更简单的函数：主体识别、问题重述、世界模型更新和知识可用性。我们在更高层次的ToM任务以及测试会话中ToM能力的任务上进行了算法测试，结果表明，与基线方法相比，我们的方法在各种任务上表现出了显著的改进，同时在各个任务中仅需少量提示调优，并且无需额外的模型训练。 

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

**Abstract (ZH)**: 语言长期以来一直被视为人类推理的重要工具。大型语言模型（LLMs）的突破引发了利用这些模型解决复杂推理任务的研究兴趣。研究人员已经超越了简单的自回归令牌生成，引入了“思维”的概念——这个概念表示推理过程中的一系列中间步骤。这种创新的范式使LLMs能够模仿复杂的human reasoning过程，如树搜索和反思性思考。最近，一种新兴的学习推理趋势应用强化学习（RL）来训练LLMs掌握推理过程。这种方法通过试错搜索算法自动生成高质量的推理轨迹，显著扩大了LLMs的推理能力，提供了大量额外的训练数据。此外，近期的研究表明，鼓励LLMs在测试时使用更多的令牌进行“思考”可以进一步显著提高推理准确性。因此，训练时间和测试时间的扩展展示了新的研究方向——大型推理模型的道路。OpenAI的o1系列引入在这一研究方向上标志着一个重要里程碑。在这篇综述中，我们对LLMs推理的最新进展进行了全面回顾。我们首先介绍了LLMs的基础背景，然后探讨了推动大型推理模型发展的关键技术组件，重点是自动化数据构建、学习推理技术以及测试时间扩展。我们还分析了构建大型推理模型的流行开源项目，最后讨论了面临的挑战和未来的研究方向。 

---
# CarMem: Enhancing Long-Term Memory in LLM Voice Assistants through Category-Bounding 

**Title (ZH)**: CarMem：通过类别边界增强大规模语言模型语音助手的长期记忆 

**Authors**: Johannes Kirmayr, Lukas Stappen, Phillip Schneider, Florian Matthes, Elisabeth André  

**Link**: [PDF](https://arxiv.org/pdf/2501.09645)  

**Abstract**: In today's assistant landscape, personalisation enhances interactions, fosters long-term relationships, and deepens engagement. However, many systems struggle with retaining user preferences, leading to repetitive user requests and disengagement. Furthermore, the unregulated and opaque extraction of user preferences in industry applications raises significant concerns about privacy and trust, especially in regions with stringent regulations like Europe. In response to these challenges, we propose a long-term memory system for voice assistants, structured around predefined categories. This approach leverages Large Language Models to efficiently extract, store, and retrieve preferences within these categories, ensuring both personalisation and transparency. We also introduce a synthetic multi-turn, multi-session conversation dataset (CarMem), grounded in real industry data, tailored to an in-car voice assistant setting. Benchmarked on the dataset, our system achieves an F1-score of .78 to .95 in preference extraction, depending on category granularity. Our maintenance strategy reduces redundant preferences by 95% and contradictory ones by 92%, while the accuracy of optimal retrieval is at .87. Collectively, the results demonstrate the system's suitability for industrial applications. 

**Abstract (ZH)**: 在当今的助手环境中，个性化能够增强互动性，促进长期关系，并加深用户参与度。然而，许多系统在保留用户偏好方面存在困难，导致用户重复提出相同请求并最终失去兴趣。此外，在行业应用中不恰当和不透明的用户偏好提取引发了对隐私和信任的重大担忧，尤其是在像欧洲这样的监管严格的地区。为应对这些挑战，我们提出了一种基于预定义类别的长期记忆系统，该系统利用大型语言模型高效地提取、存储和检索这些类别的偏好，确保个性化和透明度。此外，我们还引入了一个合成的多轮、多会话对话数据集（CarMem），该数据集基于实际行业数据，并针对车载语音助手的环境进行了定制。在该数据集上的测试结果显示，我们的系统在偏好提取上的F1分数范围为0.78到0.95，这取决于类别的细度。我们的维护策略将重复的偏好减少了95%，矛盾的偏好减少了92%，而最优检索的准确率为0.87。综合这些结果表明，该系统适合工业应用。 

---
# Confidence Estimation for Error Detection in Text-to-SQL Systems 

**Title (ZH)**: 文本转SQL系统中错误检测的置信度估计 

**Authors**: Oleg Somov, Elena Tutubalina  

**Link**: [PDF](https://arxiv.org/pdf/2501.09527)  

**Abstract**: Text-to-SQL enables users to interact with databases through natural language, simplifying the retrieval and synthesis of information. Despite the success of large language models (LLMs) in converting natural language questions into SQL queries, their broader adoption is limited by two main challenges: achieving robust generalization across diverse queries and ensuring interpretative confidence in their predictions. To tackle these issues, our research investigates the integration of selective classifiers into Text-to-SQL systems. We analyse the trade-off between coverage and risk using entropy based confidence estimation with selective classifiers and assess its impact on the overall performance of Text-to-SQL models. Additionally, we explore the models' initial calibration and improve it with calibration techniques for better model alignment between confidence and accuracy. Our experimental results show that encoder-decoder T5 is better calibrated than in-context-learning GPT 4 and decoder-only Llama 3, thus the designated external entropy-based selective classifier has better performance. The study also reveal that, in terms of error detection, selective classifier with a higher probability detects errors associated with irrelevant questions rather than incorrect query generations. 

**Abstract (ZH)**: 将以下论文内容或标题翻译成中文，符合学术规范：

文本到SQL查询（Text-to-SQL）使用户能够通过自然语言与数据库进行交互，简化信息的检索和合成。尽管大型语言模型（LLMs）在将自然语言问题转换为SQL查询方面取得了成功，但它们的广泛应用受到两大主要挑战的限制：跨多样化的查询实现稳健泛化，以及在预测过程中确保解释性信心。为了解决这些问题，我们的研究探讨了将选择性分类器整合到Text-to-SQL系统中。我们使用基于熵的信任估计方法分析选择性分类器的覆盖率与风险之间的权衡，并评估其对Text-to-SQL模型整体性能的影响。此外，我们研究了模型的初始校准，并通过校准技术改进了校准，以更好地实现信心与准确性之间的对齐。我们的实验结果表明，编码器-解码器T5在校准方面优于具有上下文学习的GPT 4和仅解码器的Llama 3，因此指定的外部熵基选择性分类器的性能更优。研究还揭示，从错误检测的角度来看，具有较高概率的选择性分类器更倾向于检测与无关问题相关的错误，而不是查询生成错误。 

---
# Augmenting a Large Language Model with a Combination of Text and Visual Data for Conversational Visualization of Global Geospatial Data 

**Title (ZH)**: 将一大型语言模型与文本和视觉数据相结合，以实现全球地理空间数据的对话可视化 

**Authors**: Omar Mena, Alexandre Kouyoumdjian, Lonni Besançon, Michael Gleicher, Ivan Viola, Anders Ynnerman  

**Link**: [PDF](https://arxiv.org/pdf/2501.09521)  

**Abstract**: We present a method for augmenting a Large Language Model (LLM) with a combination of text and visual data to enable accurate question answering in visualization of scientific data, making conversational visualization possible. LLMs struggle with tasks like visual data interaction, as they lack contextual visual information. We address this problem by merging a text description of a visualization and dataset with snapshots of the visualization. We extract their essential features into a structured text file, highly compact, yet descriptive enough to appropriately augment the LLM with contextual information, without any fine-tuning. This approach can be applied to any visualization that is already finally rendered, as long as it is associated with some textual description. 

**Abstract (ZH)**: 我们提出了一种方法，将大型语言模型（LLM）与文本和视觉数据结合起来，以实现科学数据可视化中的准确问题回答，从而使对话式可视化成为可能。LLM 在处理如视觉数据交互等任务时存在困难，因为它们缺乏上下文视觉信息。我们通过将可视化和数据集的文本描述与可视化快照相结合来解决这一问题。我们将这些信息的关键特征提取到一个结构化的文本文件中，该文件虽高度紧凑，但描述足够详细，能够为LLM提供必要的上下文信息，而无需进行任何微调。这种方法可以应用于任何已最终渲染的可视化，只要它与某些文本描述相关联即可。 

---
# A Survey on Responsible LLMs: Inherent Risk, Malicious Use, and Mitigation Strategy 

**Title (ZH)**: 负责任的大语言模型综述：固有风险、恶意使用及缓解策略 

**Authors**: Huandong Wang, Wenjie Fu, Yingzhou Tang, Zhilong Chen, Yuxi Huang, Jinghua Piao, Chen Gao, Fengli Xu, Tao Jiang, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.09431)  

**Abstract**: While large language models (LLMs) present significant potential for supporting numerous real-world applica- tions and delivering positive social impacts, they still face significant challenges in terms of the inherent risk of privacy leakage, hallucinated outputs, and value misalignment, and can be maliciously used for generating toxic content and unethical purposes after been jailbroken. Therefore, in this survey, we present a comprehensive review of recent advancements aimed at mitigating these issues, organized across the four phases of LLM development and usage: data collecting and pre-training, fine-tuning and alignment, prompting and reasoning, and post-processing and auditing. We elaborate on the recent advances for enhancing the performance of LLMs in terms of privacy protection, hallucination reduction, value alignment, toxicity elimination, and jailbreak defenses. In contrast to previous surveys that focus on a single dimension of responsible LLMs, this survey presents a unified framework that encompasses these diverse dimensions, providing a comprehensive view of enhancing LLMs to better serve real-world applications. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在支持众多实际应用并带来积极的社会影响方面展现出巨大潜力，它们仍然面临诸多根本性的挑战，包括隐私泄露风险、虚构输出和价值观偏离等问题。此外，在被破解后，这些模型还可能被恶意使用，生成有害内容和不道德的内容。因此，在本文综述中，我们对旨在缓解这些问题的最新进展进行了全面回顾，并将这些进展按大型语言模型开发和使用过程的四个阶段进行组织：数据收集和预训练、微调和价值观对齐、提示和推理以及后处理和审核。我们详细阐述了在隐私保护、减少虚构输出、价值观对齐、消除有害内容以及破解防护方面的最新进展。与其他单一维度关注负责任的LLMs的综述不同，本文提供了一个综合框架，涵盖了这些多维方面，为更好地服务于实际应用的大型语言模型提供了全面的观点。 

---
# Vision-Language Models Do Not Understand Negation 

**Title (ZH)**: 视觉-语言模型不懂得否定意义 

**Authors**: Kumail Alhamoud, Shaden Alshammari, Yonglong Tian, Guohao Li, Philip Torr, Yoon Kim, Marzyeh Ghassemi  

**Link**: [PDF](https://arxiv.org/pdf/2501.09425)  

**Abstract**: Many practical vision-language applications require models that understand negation, e.g., when using natural language to retrieve images which contain certain objects but not others. Despite advancements in vision-language models (VLMs) through large-scale training, their ability to comprehend negation remains underexplored. This study addresses the question: how well do current VLMs understand negation? We introduce NegBench, a new benchmark designed to evaluate negation understanding across 18 task variations and 79k examples spanning image, video, and medical datasets. The benchmark consists of two core tasks designed to evaluate negation understanding in diverse multimodal settings: Retrieval with Negation and Multiple Choice Questions with Negated Captions. Our evaluation reveals that modern VLMs struggle significantly with negation, often performing at chance level. To address these shortcomings, we explore a data-centric approach wherein we finetune CLIP models on large-scale synthetic datasets containing millions of negated captions. We show that this approach can result in a 10% increase in recall on negated queries and a 40% boost in accuracy on multiple-choice questions with negated captions. 

**Abstract (ZH)**: 许多实际的视觉-语言应用需要能够理解否定的模型，例如，使用自然语言检索包含某些对象但不包含其他对象的图片。尽管通过大规模训练在视觉-语言模型（VLMs）方面取得了进展，但它们对否定的理解能力仍远未探索充分。本研究旨在回答一个问题：当前的VLMs在理解否定方面究竟做得如何？我们引入了NegBench，这是首个旨在评估18种任务变体和79,000个示例（涵盖图像、视频和医疗数据集）中的否定理解能力的新基准。该基准包含两个核心任务，旨在评估在不同多模态设置中对否定的理解：带有否定检索和带有否定描述的多项选择题。我们的评估结果表明，现代VLMs在处理否定方面面临显著挑战，往往表现水平仅相当于随机猜测。为了应对这些不足，我们探索了一种以数据为中心的方法，在包含数百万否定描述的大型合成数据集上微调CLIP模型。我们发现，这种方法能够在否定查询上的召回率提升10%，并在带有否定描述的多项选择题上的准确率提升40%。 

---
# Shape-Based Single Object Classification Using Ensemble Method Classifiers 

**Title (ZH)**: 基于形状的单对象分类：ensemble方法分类器的应用 

**Authors**: Nur Shazwani Kamarudin, Mokhairi Makhtar, Syadiah Nor Wan Shamsuddin, Syed Abdullah Fadzli  

**Link**: [PDF](https://arxiv.org/pdf/2501.09311)  

**Abstract**: Nowadays, more and more images are available. Annotation and retrieval of the images pose classification problems, where each class is defined as the group of database images labelled with a common semantic label. Various systems have been proposed for content-based retrieval, as well as for image classification and indexing. In this paper, a hierarchical classification framework has been proposed for bridging the semantic gap effectively and achieving multi-category image classification. A well known pre-processing and post-processing method was used and applied to three problems; image segmentation, object identification and image classification. The method was applied to classify single object images from Amazon and Google datasets. The classification was tested for four different classifiers; BayesNetwork (BN), Random Forest (RF), Bagging and Vote. The estimated classification accuracies ranged from 20% to 99% (using 10-fold cross validation). The Bagging classifier presents the best performance, followed by the Random Forest classifier. 

**Abstract (ZH)**: 如今，可供使用的图像越来越多。对图像的标注和检索构成了分类问题，其中每个类别被定义为一组具有相同语义标签的数据库图像。针对基于内容的检索、图像分类和索引，已经提出了多种系统。本文提出了一种分层分类框架，有效缩小了语义差距，并实现了多类别图像分类。一种熟知的预处理和后处理方法被应用于三个问题：图像分割、物体识别和图像分类。该方法被应用于对来自Amazon和Google数据集的单个物体图像进行分类。分类测试使用了四种不同的分类器：贝叶斯网络（BN）、随机森林（RF）、Bagging和投票。估计的分类准确率范围从20%到99%（使用10折交叉验证）。Bagging分类器表现出最佳性能，其次是随机森林分类器。 

---
# Understanding Mental Health Content on Social Media and Its Effect Towards Suicidal Ideation 

**Title (ZH)**: 理解社交媒体上的心理健康内容及其对自杀念头的影响 

**Authors**: Mohaiminul Islam Bhuiyan, Nur Shazwani Kamarudin, Nur Hafieza Ismail  

**Link**: [PDF](https://arxiv.org/pdf/2501.09309)  

**Abstract**: This review underscores the critical need for effective strategies to identify and support individuals with suicidal ideation, exploiting technological innovations in ML and DL to further suicide prevention efforts. The study details the application of these technologies in analyzing vast amounts of unstructured social media data to detect linguistic patterns, keywords, phrases, tones, and contextual cues associated with suicidal thoughts. It explores various ML and DL models like SVMs, CNNs, LSTM, neural networks, and their effectiveness in interpreting complex data patterns and emotional nuances within text data. The review discusses the potential of these technologies to serve as a life-saving tool by identifying at-risk individuals through their digital traces. Furthermore, it evaluates the real-world effectiveness, limitations, and ethical considerations of employing these technologies for suicide prevention, stressing the importance of responsible development and usage. The study aims to fill critical knowledge gaps by analyzing recent studies, methodologies, tools, and techniques in this field. It highlights the importance of synthesizing current literature to inform practical tools and suicide prevention efforts, guiding innovation in reliable, ethical systems for early intervention. This research synthesis evaluates the intersection of technology and mental health, advocating for the ethical and responsible application of ML, DL, and NLP to offer life-saving potential worldwide while addressing challenges like generalizability, biases, privacy, and the need for further research to ensure these technologies do not exacerbate existing inequities and harms. 

**Abstract (ZH)**: 这篇综述强调了在识别和支持有自杀意念的个体方面迫切需要有效的策略，并利用机器学习（ML）和深度学习（DL）技术进一步推进自杀预防工作。研究详细探讨了这些技术在分析大量非结构化的社交媒体数据中检测语言模式、关键词、短语、语气以及与自杀思想相关的上下文线索的应用。综述研究了支持向量机（SVMs）、卷积神经网络（CNNs）、长短期记忆网络（LSTM）、神经网络等各种ML和DL模型在这方面的有效性，特别是在解释文本数据中的复杂模式和情绪细微差别的能力。该综述讨论了这些技术如何作为一种救生工具，通过识别个体的数字痕迹来发现潜在风险个体。此外，它还评估了这些技术在实际应用中的有效性、局限性和伦理考虑，强调负责任的开发和使用的重要性。研究旨在通过分析该领域的最新研究、方法论、工具和技术来填补重要知识空白，强调综合当前文献以指导实用工具和自杀预防工作的必要性，并指导建立可靠、伦理的早期干预系统。该研究综述评估了技术和心理健康之间的交汇点，倡导负责任和伦理地应用机器学习、深度学习和自然语言处理，以在全球范围内提供挽救生命的潜力，同时解决泛化性、偏见、隐私等问题，并强调进一步研究的必要性，以确保这些技术不会加剧现有的不平等和伤害。 

---
# Efficient Few-Shot Medical Image Analysis via Hierarchical Contrastive Vision-Language Learning 

**Title (ZH)**: 通过分层对比视图语言学习的高效少样本医疗图像分析 

**Authors**: Harrison Fuller, Fernando Gabriela Garcia, Victor Flores  

**Link**: [PDF](https://arxiv.org/pdf/2501.09294)  

**Abstract**: Few-shot learning in medical image classification presents a significant challenge due to the limited availability of annotated data and the complex nature of medical imagery. In this work, we propose Adaptive Vision-Language Fine-tuning with Hierarchical Contrastive Alignment (HiCA), a novel framework that leverages the capabilities of Large Vision-Language Models (LVLMs) for medical image analysis. HiCA introduces a two-stage fine-tuning strategy, combining domain-specific pretraining and hierarchical contrastive learning to align visual and textual representations at multiple levels. We evaluate our approach on two benchmark datasets, Chest X-ray and Breast Ultrasound, achieving state-of-the-art performance in both few-shot and zero-shot settings. Further analyses demonstrate the robustness, generalizability, and interpretability of our method, with substantial improvements in performance compared to existing baselines. Our work highlights the potential of hierarchical contrastive strategies in adapting LVLMs to the unique challenges of medical imaging tasks. 

**Abstract (ZH)**: 在医学图像分类中，基于少量样本的学习面临着显著的挑战，这主要是由于标注数据的稀缺性和医学图像的复杂性。本文提出了一种名为Hierarchical Contrastive Alignment（HiCA）的适应性视觉-语言微调框架，该框架利用大型视觉-语言模型（LVLMs）的能力来解决医学图像分析问题。HiCA引入了一种两阶段微调策略，结合领域特定的预训练和层次对比学习，以在多个层次上对齐视觉和文本表示。我们在胸部X光和乳腺超声两个基准数据集上评估了该方法，在少量样本和零样本设置中均取得了最先进的性能。进一步的分析表明，该方法具有较强的鲁棒性、泛化能力和可解释性，并显著优于现有基线方法。本文强调了层次对比策略在适应LVLMs解决医学影像任务的独特挑战方面的潜力。 

---
# VCRScore: Image captioning metric based on V\&L Transformers, CLIP, and precision-recall 

**Title (ZH)**: VCRScore：基于V&L变压器、CLIP和精确召回的图像配对评估指标 

**Authors**: Guillermo Ruiz, Tania Ramírez, Daniela Moctezuma  

**Link**: [PDF](https://arxiv.org/pdf/2501.09155)  

**Abstract**: Image captioning has become an essential Vision & Language research task. It is about predicting the most accurate caption given a specific image or video. The research community has achieved impressive results by continuously proposing new models and approaches to improve the overall model's performance. Nevertheless, despite increasing proposals, the performance metrics used to measure their advances have remained practically untouched through the years. A probe of that, nowadays metrics like BLEU, METEOR, CIDEr, and ROUGE are still very used, aside from more sophisticated metrics such as BertScore and ClipScore.
Hence, it is essential to adjust how are measure the advances, limitations, and scopes of the new image captioning proposals, as well as to adapt new metrics to these new advanced image captioning approaches.
This work proposes a new evaluation metric for the image captioning problem. To do that, first, it was generated a human-labeled dataset to assess to which degree the captions correlate with the image's content. Taking these human scores as ground truth, we propose a new metric, and compare it with several well-known metrics, from classical to newer ones. Outperformed results were also found, and interesting insights were presented and discussed. 

**Abstract (ZH)**: 图像标注已成为视觉与语言研究中不可或缺的任务。它涉及根据特定图像或视频预测最准确的描述性语句。研究社区通过不断提出新的模型和方法来提高整体模型的性能，取得了令人印象深刻的结果。然而，尽管不断提出新的提案，用于衡量这些进步的性能指标多年来基本保持不变。如今，像 BLEU、METEOR、CIDEr 和 ROUGE 等指标仍然广泛使用，而且还有一些更复杂的指标，如 BERTScore 和 ClipScore。

因此，调整用于衡量新图像标注提案的进步、局限性及适用范围的评估方法至关重要，同时还需要将新指标适应到这些新的高级图像标注方法中。

本文提出了一种新的图像标注评估指标。首先，我们生成了一个由人工标注的数据集，以评估描述性语句与图像内容的相关程度。利用这些人工评分作为基准，我们提出了一个新的指标，并将其与一些广为人知的指标进行了比较——从经典的到较新的指标。我们还发现了一些优越的性能，并提出了有趣的观点并进行了讨论。 

---
# Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG 

**Title (ZH)**: 代理检索增强生成：代理型RAG综述

在这个翻译中，“Agentic Retrieval-Augmented Generation”被译为“代理检索增强生成”，“Agentic RAG”被译为“代理型RAG”，确保了学术规范和准确性。 

**Authors**: Aditi Singh, Abul Ehtesham, Saket Kumar, Tala Talaei Khoei  

**Link**: [PDF](https://arxiv.org/pdf/2501.09136)  

**Abstract**: Large Language Models (LLMs) have revolutionized artificial intelligence (AI) by enabling human like text generation and natural language understanding. However, their reliance on static training data limits their ability to respond to dynamic, real time queries, resulting in outdated or inaccurate outputs. Retrieval Augmented Generation (RAG) has emerged as a solution, enhancing LLMs by integrating real time data retrieval to provide contextually relevant and up-to-date responses. Despite its promise, traditional RAG systems are constrained by static workflows and lack the adaptability required for multistep reasoning and complex task management.
Agentic Retrieval-Augmented Generation (Agentic RAG) transcends these limitations by embedding autonomous AI agents into the RAG pipeline. These agents leverage agentic design patterns reflection, planning, tool use, and multiagent collaboration to dynamically manage retrieval strategies, iteratively refine contextual understanding, and adapt workflows to meet complex task requirements. This integration enables Agentic RAG systems to deliver unparalleled flexibility, scalability, and context awareness across diverse applications.
This survey provides a comprehensive exploration of Agentic RAG, beginning with its foundational principles and the evolution of RAG paradigms. It presents a detailed taxonomy of Agentic RAG architectures, highlights key applications in industries such as healthcare, finance, and education, and examines practical implementation strategies. Additionally, it addresses challenges in scaling these systems, ensuring ethical decision making, and optimizing performance for real-world applications, while providing detailed insights into frameworks and tools for implementing Agentic RAG 

**Abstract (ZH)**: 大型语言模型（LLMs）通过实现类似人类的文本生成和自然语言理解，彻底改变了人工智能（AI）。然而，它们依赖于静态训练数据，限制了它们对动态、实时查询的响应能力，从而导致输出过时或不准确。检索增强生成（RAG）作为一种解决方案已经出现，通过整合实时数据检索来提供上下文相关且最新的响应，增强了LLMs。尽管RAG具有巨大的潜力，但传统的RAG系统受限于静态的流程工作模式，缺乏多步骤推理和复杂任务管理所需的适应性。

Active Retrieval-Augmented Generation（Agentic RAG）超越了这些限制，将自主AI代理嵌入到RAG流水线中。这些代理利用自主设计模式中的反射、规划、工具使用和多代理协作，动态管理检索策略，迭代性地精炼上下文理解，并适应性地调整流程以满足复杂的任务需求。这种整合使Agentic RAG系统能够在各种应用中提供无与伦比的灵活性、可扩展性和上下文意识。

本文综述了Agentic RAG，从其基础原理和RAG范式的演变开始，详细探讨了Agentic RAG架构的分类，指出了其在医疗、金融和教育等行业的关键应用，并分析了实现策略。此外，本文还讨论了在这些系统中扩展、确保伦理决策和优化实际应用性能的挑战，提供了实施Agentic RAG框架和工具的详细见解。 

---
# Benchmarking Robustness of Contrastive Learning Models for Medical Image-Report Retrieval 

**Title (ZH)**: 对比学习模型在医学图像-报告检索中稳健性的benchmark研究 

**Authors**: Demetrio Deanda, Yuktha Priya Masupalli, Jeong Yang, Young Lee, Zechun Cao, Gongbo Liang  

**Link**: [PDF](https://arxiv.org/pdf/2501.09134)  

**Abstract**: Medical images and reports offer invaluable insights into patient health. The heterogeneity and complexity of these data hinder effective analysis. To bridge this gap, we investigate contrastive learning models for cross-domain retrieval, which associates medical images with their corresponding clinical reports. This study benchmarks the robustness of four state-of-the-art contrastive learning models: CLIP, CXR-RePaiR, MedCLIP, and CXR-CLIP. We introduce an occlusion retrieval task to evaluate model performance under varying levels of image corruption. Our findings reveal that all evaluated models are highly sensitive to out-of-distribution data, as evidenced by the proportional decrease in performance with increasing occlusion levels. While MedCLIP exhibits slightly more robustness, its overall performance remains significantly behind CXR-CLIP and CXR-RePaiR. CLIP, trained on a general-purpose dataset, struggles with medical image-report retrieval, highlighting the importance of domain-specific training data. The evaluation of this work suggests that more effort needs to be spent on improving the robustness of these models. By addressing these limitations, we can develop more reliable cross-domain retrieval models for medical applications. 

**Abstract (ZH)**: 医学图像和报告为患者的健康提供了宝贵的信息。然而，这些数据的异质性和复杂性阻碍了有效的分析。为了弥合这一差距，我们研究了对比学习模型在跨域检索中的应用，将医学图像与其相应的临床报告关联起来。本研究对四种最新的对比学习模型（CLIP、CXR-RePaiR、MedCLIP和CXR-CLIP）的鲁棒性进行了基准测试。我们引入了一种遮挡检索任务，以评估模型在图像受损程度变化下的性能。研究结果表明，所有评估的模型都对分布外数据表现出高度敏感性，随着遮挡程度的增加，性能呈现出比例下降的趋势。虽然MedCLIP显示出一定的鲁棒性，但其整体性能仍明显落后于CXR-CLIP和CXR-RePaiR。CLIP作为一种通用目的模型，在医学图像-报告检索方面表现不佳，强调了领域特定训练数据的重要性。这项工作在评估模型时指出，需要更多精力来提高这些模型的鲁棒性。通过解决这些局限性，我们可以开发出更可靠的应用于医学领域的跨域检索模型。 

---
# Generative Visual Commonsense Answering and Explaining with Generative Scene Graph Constructing 

**Title (ZH)**: 生成式视觉常识回答与生成场景图构建解释 

**Authors**: Fan Yuan, Xiaoyuan Fang, Rong Quan, Jing Li, Wei Bi, Xiaogang Xu, Piji Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.09041)  

**Abstract**: Visual Commonsense Reasoning, which is regarded as one challenging task to pursue advanced visual scene comprehension, has been used to diagnose the reasoning ability of AI systems. However, reliable reasoning requires a good grasp of the scene's details. Existing work fails to effectively exploit the real-world object relationship information present within the scene, and instead overly relies on knowledge from training memory. Based on these observations, we propose a novel scene-graph-enhanced visual commonsense reasoning generation method named \textit{\textbf{G2}}, which first utilizes the image patches and LLMs to construct a location-free scene graph, and then answer and explain based on the scene graph's information. We also propose automatic scene graph filtering and selection strategies to absorb valuable scene graph information during training. Extensive experiments are conducted on the tasks and datasets of scene graph constructing and visual commonsense answering and explaining, respectively. Experimental results and ablation analysis demonstrate the effectiveness of our proposed framework. 

**Abstract (ZH)**: 视觉常识推理（Visual Commonsense Reasoning）被视为追求高级视觉场景理解的一项具有挑战性的任务，已被用于评估AI系统的推理能力。然而，可靠推理需要对场景细节有较好的掌握。现有工作未能有效地利用场景内真实世界对象之间的关系信息，而是过度依赖训练记忆中的知识。基于这些观察，我们提出了一种新颖的场景图增强视觉常识推理生成方法，命名为**G2**。该方法首先利用图像片段和LLMs构建一个位置无关的场景图，然后基于场景图的信息进行回答和解释。我们还提出了自动场景图筛选和选择策略，以在训练过程中吸收有价值的信息。我们在场景图构建和视觉常识回答与解释的任务和数据集上进行了广泛的实验。实验结果和消融分析证明了我们提出的框架的有效性。 

---
