# Agent-centric Information Access 

**Title (ZH)**: 以代理为中心的信息访问 

**Authors**: Evangelos Kanoulas, Panagiotis Eustratiadis, Yongkang Li, Yougang Lyu, Vaishali Pal, Gabrielle Poerwawinata, Jingfen Qiao, Zihan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.19298)  

**Abstract**: As large language models (LLMs) become more specialized, we envision a future where millions of expert LLMs exist, each trained on proprietary data and excelling in specific domains. In such a system, answering a query requires selecting a small subset of relevant models, querying them efficiently, and synthesizing their responses. This paper introduces a framework for agent-centric information access, where LLMs function as knowledge agents that are dynamically ranked and queried based on their demonstrated expertise. Unlike traditional document retrieval, this approach requires inferring expertise on the fly, rather than relying on static metadata or predefined model descriptions. This shift introduces several challenges, including efficient expert selection, cost-effective querying, response aggregation across multiple models, and robustness against adversarial manipulation. To address these issues, we propose a scalable evaluation framework that leverages retrieval-augmented generation and clustering techniques to construct and assess thousands of specialized models, with the potential to scale toward millions. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）变得越来越专业化，我们设想一个未来场景，在这个场景中，存在数百万个专家型LLMs，它们各自经过专业数据训练并在特定领域表现出色。在这种系统中，回答查询需要选择一小部分相关模型，高效地查询这些模型，并综合它们的回答。本文介绍了一个以代理为中心的信息访问框架，在该框架中，LLMs充当知识代理，并根据它们展示的专业能力进行动态排名和查询。与传统的文档检索不同，这种方法要求实时推断模型的专业能力，而不仅仅是依赖静态元数据或预定义的模型描述。这种转变带来了一系列挑战，包括高效的专家选择、成本效益高效的查询、跨多个模型的响应聚合，以及对抗性操纵的鲁棒性。为了应对这些挑战，我们提出了一种可扩展的评估框架，该框架利用检索增强生成和聚类技术构建和评估数千种专业化模型，并有望扩展到数百万个模型。 

---
# Multiview graph dual-attention deep learning and contrastive learning for multi-criteria recommender systems 

**Title (ZH)**: 多视图图双向注意力深度学习与对比学习在多准则推荐系统中的应用 

**Authors**: Saman Forouzandeh, Pavel N. Krivitsky, Rohitash Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2502.19271)  

**Abstract**: Recommender systems leveraging deep learning models have been crucial for assisting users in selecting items aligned with their preferences and interests. However, a significant challenge persists in single-criteria recommender systems, which often overlook the diverse attributes of items that have been addressed by Multi-Criteria Recommender Systems (MCRS). Shared embedding vector for multi-criteria item ratings but have struggled to capture the nuanced relationships between users and items based on specific criteria. In this study, we present a novel representation for Multi-Criteria Recommender Systems (MCRS) based on a multi-edge bipartite graph, where each edge represents one criterion rating of items by users, and Multiview Dual Graph Attention Networks (MDGAT). Employing MDGAT is beneficial and important for adequately considering all relations between users and items, given the presence of both local (criterion-based) and global (multi-criteria) relations. Additionally, we define anchor points in each view based on similarity and employ local and global contrastive learning to distinguish between positive and negative samples across each view and the entire graph. We evaluate our method on two real-world datasets and assess its performance based on item rating predictions. The results demonstrate that our method achieves higher accuracy compared to the baseline method for predicting item ratings on the same datasets. MDGAT effectively capture the local and global impact of neighbours and the similarity between nodes. 

**Abstract (ZH)**: 基于深度学习模型的推荐系统对于协助用户选择与其偏好和兴趣相符的项目至关重要。然而，单一标准推荐系统仍存在显著挑战，这些系统往往忽略了多标准推荐系统(Multi-Criteria Recommender Systems, MCRS)所涉及的项目多样化属性。尽管MCRS能够为项目评分建立共享的嵌入向量，但在基于特定标准的用户与项目关系中仍难以捕捉到复杂的关系。在这项研究中，我们基于多边二分图提出了一个新的多标准推荐系统(MCRS)表示方法，其中每条边代表用户对项目的一项评分标准，并采用多视图双图注意网络(Multiview Dual Graph Attention Networks, MDGAT)。利用MDGAT有助于充分考虑用户和项目之间的所有关系，因为这些关系既包括局部(基于标准)关系，也包括全局(多标准)关系。此外，我们根据相似性在每个视图中定义锚点，并采用局部和全局对比学习来区分每个视图及其整个图形中的正样本和负样本。我们在两个真实世界数据集上评估了我们的方法，并基于项目评分预测来评估其性能。结果表明，我们的方法在预测项目评分方面比基线方法更具准确性。MDGAT能够有效捕捉邻居的局部和全局影响以及节点之间的相似性。 

---
# UQABench: Evaluating User Embedding for Prompting LLMs in Personalized Question Answering 

**Title (ZH)**: UQABench：评估用户嵌入在个性化问答中提示LLMs的效果 

**Authors**: Langming Liu, Shilei Liu, Yujin Yuan, Yizhen Zhang, Bencheng Yan, Zhiyuan Zeng, Zihao Wang, Jiaqi Liu, Di Wang, Wenbo Su, Pengjie Wang, Jian Xu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.19178)  

**Abstract**: Large language models (LLMs) achieve remarkable success in natural language processing (NLP). In practical scenarios like recommendations, as users increasingly seek personalized experiences, it becomes crucial to incorporate user interaction history into the context of LLMs to enhance personalization. However, from a practical utility perspective, user interactions' extensive length and noise present challenges when used directly as text prompts. A promising solution is to compress and distill interactions into compact embeddings, serving as soft prompts to assist LLMs in generating personalized responses. Although this approach brings efficiency, a critical concern emerges: Can user embeddings adequately capture valuable information and prompt LLMs? To address this concern, we propose \name, a benchmark designed to evaluate the effectiveness of user embeddings in prompting LLMs for personalization. We establish a fair and standardized evaluation process, encompassing pre-training, fine-tuning, and evaluation stages. To thoroughly evaluate user embeddings, we design three dimensions of tasks: sequence understanding, action prediction, and interest perception. These evaluation tasks cover the industry's demands in traditional recommendation tasks, such as improving prediction accuracy, and its aspirations for LLM-based methods, such as accurately understanding user interests and enhancing the user experience. We conduct extensive experiments on various state-of-the-art methods for modeling user embeddings. Additionally, we reveal the scaling laws of leveraging user embeddings to prompt LLMs. The benchmark is available online. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在自然语言处理（NLP）中取得了显著的成功。在推荐等实际场景中，随着用户越来越多地寻求个性化体验，将用户的交互历史融入LLMs的上下文中以增强个性化变得至关重要。然而，从实用角度来看，用户交互的长度和噪音在直接作为文本提示使用时带来了挑战。一种有前景的解决方案是将用户的交互压缩和提取为紧凑的嵌入，这些嵌入可以作为软提示帮助LLMs生成个性化响应。尽管这种方法带来了效率提升，但一个关键问题也随之浮现：用户的嵌入能否充分捕捉有价值的信息并有效地提示LLMs？为了应对这一问题，我们提出了\name这一基准，旨在评估用户嵌入在提示LLMs实现个性化方面的有效性。我们建立了一个公平且标准化的评估流程，包括预训练、微调和评估阶段。为了全面评估用户嵌入，我们设计了三个任务维度：序列理解、行为预测和兴趣感知。这些评估任务涵盖了传统推荐任务对准确预测的需求，同时也反映了基于LLMs方法的期望，如准确理解用户兴趣并提升用户体验。我们对各种最先进的用户嵌入建模方法进行了广泛实验，并揭示了利用用户嵌入提示LLMs的扩展规律。该基准已在线发布供公众使用。 

---
# A 106K Multi-Topic Multilingual Conversational User Dataset with Emoticons 

**Title (ZH)**: 一个包含表情符号的106K多主题多语言对话用户数据集 

**Authors**: Heng Er Metilda Chee, Jiayin Wang, Zhiqiang Guo, Weizhi Ma, Qinglang Guo, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.19108)  

**Abstract**: Instant messaging has become a predominant form of communication, with texts and emoticons enabling users to express emotions and ideas efficiently. Emoticons, in particular, have gained significant traction as a medium for conveying sentiments and information, leading to the growing importance of emoticon retrieval and recommendation systems. However, one of the key challenges in this area has been the absence of datasets that capture both the temporal dynamics and user-specific interactions with emoticons, limiting the progress of personalized user modeling and recommendation approaches. To address this, we introduce the emoticon dataset, a comprehensive resource that includes time-based data along with anonymous user identifiers across different conversations. As the largest publicly accessible emoticon dataset to date, it comprises 22K unique users, 370K emoticons, and 8.3M messages. The data was collected from a widely-used messaging platform across 67 conversations and 720 hours of crawling. Strict privacy and safety checks were applied to ensure the integrity of both text and image data. Spanning across 10 distinct domains, the emoticon dataset provides rich insights into temporal, multilingual, and cross-domain behaviors, which were previously unavailable in other emoticon-based datasets. Our in-depth experiments, both quantitative and qualitative, demonstrate the dataset's potential in modeling user behavior and personalized recommendation systems, opening up new possibilities for research in personalized retrieval and conversational AI. The dataset is freely accessible. 

**Abstract (ZH)**: 即时通讯已成为主要的交流方式，文本和表情符号使用户能够高效地表达情感和思想。特别是，表情符号已经成为传达情感和信息的重要媒介，从而促进了表情符号检索和推荐系统的日益重要。然而，该领域的一个主要挑战是没有能够捕捉表情符号使用时间和用户特定交互的语料库，这限制了个性化用户建模和推荐方法的发展。为解决这一问题，我们提出了表情符号数据集，这是一个综合资源，包含基于时间的数据以及不同对话中的匿名用户标识符。作为迄今为止最大的公开可用表情符号数据集，它包含了22,000个独特的用户、370,000个表情符号及830万条消息。数据从一个广泛使用的消息平台的67个对话中收集而来，并爬取了720小时的信息。我们实施了严格的数据隐私和安全检查，以确保文本和图像数据的完整性。该数据集覆盖了10个不同的领域，提供了关于时间动态、多语言和跨领域行为的丰富见解，而这些见解在其他基于表情符号的数据集中是难以获得的。我们深入的定量和定性实验表明，该数据集在建模用户行为和个性化推荐系统方面具有潜在价值，为个性化检索和对话式人工智能的研究打开了新的可能性。该数据集将免费提供。 

---
# OntologyRAG: Better and Faster Biomedical Code Mapping with Retrieval-Augmented Generation (RAG) Leveraging Ontology Knowledge Graphs and Large Language Models 

**Title (ZH)**: OntologyRAG：利用本体知识图谱和大规模语言模型增强的检索增强生成（RAG）方法在生物医学代码映射中的表现与效率提升 

**Authors**: Hui Feng, Yuntzu Yin, Emiliano Reynares, Jay Nanavati  

**Link**: [PDF](https://arxiv.org/pdf/2502.18992)  

**Abstract**: Biomedical ontologies, which comprehensively define concepts and relations for biomedical entities, are crucial for structuring and formalizing domain-specific information representations. Biomedical code mapping identifies similarity or equivalence between concepts from different ontologies. Obtaining high-quality mapping usually relies on automatic generation of unrefined mapping with ontology domain fine-tuned language models (LMs), followed by manual selections or corrections by coding experts who have extensive domain expertise and familiarity with ontology schemas. The LMs usually provide unrefined code mapping suggestions as a list of candidates without reasoning or supporting evidence, hence coding experts still need to verify each suggested candidate against ontology sources to pick the best matches. This is also a recurring task as ontology sources are updated regularly to incorporate new research findings. Consequently, the need of regular LM retraining and manual refinement make code mapping time-consuming and labour intensive. In this work, we created OntologyRAG, an ontology-enhanced retrieval-augmented generation (RAG) method that leverages the inductive biases from ontological knowledge graphs for in-context-learning (ICL) in large language models (LLMs). Our solution grounds LLMs to knowledge graphs with unrefined mappings between ontologies and processes questions by generating an interpretable set of results that include prediction rational with mapping proximity assessment. Our solution doesn't require re-training LMs, as all ontology updates could be reflected by updating the knowledge graphs with a standard process. Evaluation results on a self-curated gold dataset show promises of using our method to enable coding experts to achieve better and faster code mapping. The code is available at this https URL. 

**Abstract (ZH)**: 生物医学本体是以全面定义生物医学实体的概念及其关系而著称，对于结构化和形式化专业领域信息表示至关重要。生物医学代码映射识别不同本体概念之间的相似性或等价性。获得高质量的映射通常依赖于使用细调过的本体领域语言模型（LMs）自动生成未经精炼的映射，然后由具有丰富领域专业知识和本体 schema 熟悉度的编码专家进行手工选择或修正。这些语言模型通常只是提供未经精炼的代码映射建议列表，而不进行推理或提供支持证据，因此编码专家仍需验证每个建议的候选人以挑选最佳匹配。由于本体源定期更新以纳入新的研究发现，这一任务是重复执行的。因此，定期重新训练 LM 和手动精炼使得代码映射耗时且劳动密集。在本项工作中，我们创建了 OntologyRAG，这是一种利用本体知识图谱的归纳偏倚增强了检索增强生成（RAG）方法，在大型语言模型（LLMs）的上下文学习（ICL）中发挥作用。我们的解决方案将 LLMs 接地于知识图谱，并通过生成包括预测理由和映射相似度评估的可解释结果集来处理问题。我们的解决方案无需重新训练 LM，因为所有本体更新都可以通过标准化过程更新知识图谱来反映。在自制作的黄金数据集上的评估结果表明，使用我们的方法可以帮助编码专家更快、更好地进行代码映射。代码可在以下 URL 获取：[提供的URL]。 

---
# OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment 

**Title (ZH)**: OneRec：生成式推荐器与迭代偏好对齐的统一检索与排名模型 

**Authors**: Jiaxin Deng, Shiyao Wang, Kuo Cai, Lejian Ren, Qigen Hu, Weifeng Ding, Qiang Luo, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.18965)  

**Abstract**: Recently, generative retrieval-based recommendation systems have emerged as a promising paradigm. However, most modern recommender systems adopt a retrieve-and-rank strategy, where the generative model functions only as a selector during the retrieval stage. In this paper, we propose OneRec, which replaces the cascaded learning framework with a unified generative model. To the best of our knowledge, this is the first end-to-end generative model that significantly surpasses current complex and well-designed recommender systems in real-world scenarios. Specifically, OneRec includes: 1) an encoder-decoder structure, which encodes the user's historical behavior sequences and gradually decodes the videos that the user may be interested in. We adopt sparse Mixture-of-Experts (MoE) to scale model capacity without proportionally increasing computational FLOPs. 2) a session-wise generation approach. In contrast to traditional next-item prediction, we propose a session-wise generation, which is more elegant and contextually coherent than point-by-point generation that relies on hand-crafted rules to properly combine the generated results. 3) an Iterative Preference Alignment module combined with Direct Preference Optimization (DPO) to enhance the quality of the generated results. Unlike DPO in NLP, a recommendation system typically has only one opportunity to display results for each user's browsing request, making it impossible to obtain positive and negative samples simultaneously. To address this limitation, We design a reward model to simulate user generation and customize the sampling strategy. Extensive experiments have demonstrated that a limited number of DPO samples can align user interest preferences and significantly improve the quality of generated results. We deployed OneRec in the main scene of Kuaishou, achieving a 1.6\% increase in watch-time, which is a substantial improvement. 

**Abstract (ZH)**: 近年来，生成式检索推荐系统已经展现出一种有前景的范式。然而，大多数现代推荐系统采用检索-排序策略，其中生成模型仅在检索阶段作为选择器发挥作用。本文中，我们提出了一种新型的推荐模型OneRec，它用统一的生成模型替代了多阶段的学习框架。据我们所知，OneRec是首个可以在实际应用中显著超越当前复杂且设计良好的推荐系统的方法，并且是一个端到端的生成模型。具体而言，OneRec包括以下内容：1）编码-解码结构，编码用户的 histórico 行为序列，并逐步解码用户可能感兴趣的视频。我们采用稀疏 Mixture-of-Experts (MoE) 模型，以扩大模型容量而不需成比例增加计算量。2）会话级生成方法。与传统的下一个项目预测相比，我们提出了会话级生成，这是一种比点对点生成更具优雅性和语境连贯性的方法，会话级生成不需要依赖手工设计的规则来正确结合生成结果。3）结合直接偏好优化（DPO）的迭代偏好对齐模块，以增强生成结果的质量。不同于自然语言处理中的DPO，推荐系统通常对每个用户的浏览请求只能展示一次结果，无法同时获取正负样本。为克服这一限制，我们设计了一个奖励模型来模拟用户生成并定制采样策略。大量实验表明，有限数量的DPO样本可以对齐用户兴趣偏好，显著提高生成结果的质量。我们在快手的主要场景中部署了OneRec，实现了一次观看时间增加了1.6%，这是一个显著的提升。 

---
# A Multifacet Hierarchical Sentiment-Topic Model with Application to Multi-Brand Online Review Analysis 

**Title (ZH)**: 一种多维度层次情感主题模型及其在多品牌在线评论分析中的应用 

**Authors**: Qiao Liang, Xinwei Deng  

**Link**: [PDF](https://arxiv.org/pdf/2502.18927)  

**Abstract**: Multi-brand analysis based on review comments and ratings is a commonly used strategy to compare different brands in marketing. It can help consumers make more informed decisions and help marketers understand their brand's position in the market. In this work, we propose a multifacet hierarchical sentiment-topic model (MH-STM) to detect brand-associated sentiment polarities towards multiple comparative aspects from online customer reviews. The proposed method is built on a unified generative framework that explains review words with a hierarchical brand-associated topic model and the overall polarity score with a regression model on the empirical topic distribution. Moreover, a novel hierarchical Polya urn (HPU) scheme is proposed to enhance the topic-word association among topic hierarchy, such that the general topics shared by all brands are separated effectively from the unique topics specific to individual brands. The performance of the proposed method is evaluated on both synthetic data and two real-world review corpora. Experimental studies demonstrate that the proposed method can be effective in detecting reasonable topic hierarchy and deriving accurate brand-associated rankings on multi-aspects. 

**Abstract (ZH)**: 基于评论和评分的多品牌分析是一种常用的市场比较策略。它可以帮助消费者做出更加明智的决策，并帮助营销人员了解其品牌的市场地位。在本研究中，我们提出了一种多维度层次情感-主题模型（MH-STM），以从在线客户评论中检测多个比较方面与品牌相关的极性。该提出的模型基于一个统一生成框架，通过层次品牌相关的主题模型解释评论词汇，并通过经验主题分布上的回归模型解释总体极性分数。此外，我们提出了一种新的层次Polya urn（HPU）方案，以增强主题层次结构中的主题词关联，从而使所有品牌共有的主题与各品牌独有的主题能够得到有效分离。我们在合成数据和两个真实世界的评论语料库上评估了所提出方法的性能。实验研究表明，所提出的方法在检测合理的主题层次结构和多维度品牌关联排名方面具有有效性。 

---
# Hierarchical corpus encoder: Fusing generative retrieval and dense indices 

**Title (ZH)**: 层次式语料库编码器：融合生成型检索和密集索引 

**Authors**: Tongfei Chen, Ankita Sharma, Adam Pauls, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2502.18877)  

**Abstract**: Generative retrieval employs sequence models for conditional generation of document IDs based on a query (DSI (Tay et al., 2022); NCI (Wang et al., 2022); inter alia). While this has led to improved performance in zero-shot retrieval, it is a challenge to support documents not seen during training. We identify the performance of generative retrieval lies in contrastive training between sibling nodes in a document hierarchy. This motivates our proposal, the hierarchical corpus encoder (HCE), which can be supported by traditional dense encoders. Our experiments show that HCE achieves superior results than generative retrieval models under both unsupervised zero-shot and supervised settings, while also allowing the easy addition and removal of documents to the index. 

**Abstract (ZH)**: 生成式检索利用序列模型根据查询条件生成文档ID（DSI（Tay等人，2022）；NCI（Wang等人，2022）等）。尽管这在零样本检索中提高了性能，但在支持未在训练中遇到的文档方面仍面临挑战。我们发现生成式检索的性能在于文档层次结构中的兄弟节点之间的对比训练。这促使我们提出层次语料库编码器（HCE），该编码器可以由传统的密集编码器支持。我们的实验表明，在无监督零样本和监督设置下，HCE在生成式检索模型中取得了更优的成绩，同时还能轻松地向索引中添加和删除文档。 

---
# Training Large Recommendation Models via Graph-Language Token Alignment 

**Title (ZH)**: 通过图-语言令牌对齐训练大型推荐模型 

**Authors**: Mingdai Yang, Zhiwei Liu, Liangwei Yang, Xiaolong Liu, Chen Wang, Hao Peng, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.18757)  

**Abstract**: Recommender systems (RS) have become essential tools for helping users efficiently navigate the overwhelming amount of information on e-commerce and social platforms. However, traditional RS relying on Collaborative Filtering (CF) struggles to integrate the rich semantic information from textual data. Meanwhile, large language models (LLMs) have shown promising results in natural language processing, but directly using LLMs for recommendation introduces challenges, such as ambiguity in generating item predictions and inefficiencies in scalability. In this paper, we propose a novel framework to train Large Recommendation models via Graph-Language Token Alignment. By aligning item and user nodes from the interaction graph with pretrained LLM tokens, GLTA effectively leverages the reasoning abilities of LLMs. Furthermore, we introduce Graph-Language Logits Matching (GLLM) to optimize token alignment for end-to-end item prediction, eliminating ambiguity in the free-form text as recommendation results. Extensive experiments on three benchmark datasets demonstrate the effectiveness of GLTA, with ablation studies validating each component. 

**Abstract (ZH)**: 推荐系统（RS）已成为帮助用户有效导航电子商务和社交平台上的海量信息的重要工具。然而，传统依赖于协同过滤（CF）的RS难以整合文本数据中的丰富语义信息。与此同时，大型语言模型（LLMs）在自然语言处理方面表现出色，但直接使用LLMs进行推荐也面临挑战，如项目预测中的歧义性和可扩展性的效率问题。在本文中，我们提出了一种新的框架，通过图-语言标记对齐来训练大型推荐模型。通过将交互图中的项目和用户节点与预训练的LLM标记对齐，GLTA有效地利用了LLMs的推理能力。此外，我们引入了图-语言对数匹配（GLLM）来优化标记对齐，以实现端到端的项目预测，并消除推荐结果中自由形式文本带来的歧义。在三个基准数据集上的广泛实验表明，GLTA的有效性得到了验证，消融研究也验证了每个组件的效果。 

---
# AgentSociety Challenge: Designing LLM Agents for User Modeling and Recommendation on Web Platforms 

**Title (ZH)**: AgentSociety挑战：设计针对网络平台用户建模与推荐的大型语言模型代理 

**Authors**: Yuwei Yan, Yu Shang, Qingbin Zeng, Yu Li, Keyu Zhao, Zhiheng Zheng, Xuefei Ning, Tianji Wu, Shengen Yan, Yu Wang, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.18754)  

**Abstract**: The AgentSociety Challenge is the first competition in the Web Conference that aims to explore the potential of Large Language Model (LLM) agents in modeling user behavior and enhancing recommender systems on web platforms. The Challenge consists of two tracks: the User Modeling Track and the Recommendation Track. Participants are tasked to utilize a combined dataset from Yelp, Amazon, and Goodreads, along with an interactive environment simulator, to develop innovative LLM agents. The Challenge has attracted 295 teams across the globe and received over 1,400 submissions in total over the course of 37 official competition days. The participants have achieved 21.9% and 20.3% performance improvement for Track 1 and Track 2 in the Development Phase, and 9.1% and 15.9% in the Final Phase, representing a significant accomplishment. This paper discusses the detailed designs of the Challenge, analyzes the outcomes, and highlights the most successful LLM agent designs. To support further research and development, we have open-sourced the benchmark environment at this https URL. 

**Abstract (ZH)**: 《AgentSociety挑战赛》是Web Conference中的首个比赛，旨在探索大型语言模型（LLM）代理在建模用户行为和提升网络平台推荐系统方面的潜力。挑战赛包含两个赛道：用户建模赛道和推荐赛道。参赛者需利用来自Yelp、Amazon和Goodreads的数据集以及互动环境模拟器，开发创新的LLM代理。比赛吸引了全球295支队伍的参与，并在为期37天的正式竞赛日内收到了超过1400份提交作品。参赛者在开发阶段和最终阶段分别实现了1轨道21.9%和2轨道20.3%的性能提升，而在最终阶段分别为1轨道9.1%和2轨道15.9%的提升，这是一项令人瞩目的成就。本文详细介绍了挑战的设计、分析了比赛结果，并重点介绍了最成功的LLM代理设计方案。为了支持后续的研究与开发，我们已在此URL处开源了基准环境。 

---
# A Cooperative Multi-Agent Framework for Zero-Shot Named Entity Recognition 

**Title (ZH)**: 零样本命名实体识别的协作多agent框架 

**Authors**: Zihan Wang, Ziqi Zhao, Yougang Lyu, Zhumin Chen, Maarten de Rijke, Zhaochun Ren  

**Link**: [PDF](https://arxiv.org/pdf/2502.18702)  

**Abstract**: Zero-shot named entity recognition (NER) aims to develop entity recognition systems from unannotated text corpora. This task presents substantial challenges due to minimal human intervention. Recent work has adapted large language models (LLMs) for zero-shot NER by crafting specialized prompt templates. It advances model self-learning abilities by incorporating self-annotated demonstrations. However, two important challenges persist: (i) Correlations between contexts surrounding entities are overlooked, leading to wrong type predictions or entity omissions. (ii) The indiscriminate use of task demonstrations, retrieved through shallow similarity-based strategies, severely misleads LLMs during inference.
In this paper, we introduce the cooperative multi-agent system (CMAS), a novel framework for zero-shot NER that uses the collective intelligence of multiple agents to address the challenges outlined above. CMAS has four main agents: (i) a self-annotator, (ii) a type-related feature (TRF) extractor, (iii) a demonstration discriminator, and (iv) an overall predictor. To explicitly capture correlations between contexts surrounding entities, CMAS reformulates NER into two subtasks: recognizing named entities and identifying entity type-related features within the target sentence. To enable controllable utilization of demonstrations, a demonstration discriminator is established to incorporate the self-reflection mechanism, automatically evaluating helpfulness scores for the target sentence. Experimental results show that CMAS significantly improves zero-shot NER performance across six benchmarks, including both domain-specific and general-domain scenarios. Furthermore, CMAS demonstrates its effectiveness in few-shot settings and with various LLM backbones. 

**Abstract (ZH)**: 零样本命名实体识别（NER）旨在从未标注文本语料库中开发实体识别系统。这一任务由于人类干预极低而面临巨大挑战。近期的工作通过设计特定的提示模板，将大型语言模型（LLMs）应用于零样本NER中，提升了模型的自我学习能力，引入了自我标注的示范。然而，仍存在两个关键挑战：(i) 忽视了实体上下文之间的关联性，导致错误的实体类型预测或实体遗漏；(ii) 通过浅层相似性策略检索的示范的无差别使用，在推理过程中严重误导了LLMs。

本文提出了合作多智能体系统（CMAS），这是一种新颖的零样本命名实体识别框架，利用多个智能体的集体智能来应对上述挑战。CMAS包括四个主要智能体：(i) 自我标注器，(ii) 类型相关特征（TRF）提取器，(iii) 示范鉴别器，以及(iv) 综合预测器。为明确捕获实体上下文之间的关联性，CMAS将NER重新表述为两个子任务：识别命名实体和在目标句子中识别实体类型相关特征。为实现示范的可控利用，建立了一个示范鉴别器，引入了自助反思机制，自动评估目标句子的示范有用性评分。实验结果表明，CMAS在包括特定领域和通用领域在内的六个基准测试中显著提高了零样本命名实体识别性能。此外，CMAS在少量样本设置和不同大型语言模型（LLM）基础架构中均展示了其有效性。 

---
# AI Enhanced Ontology Driven NLP for Intelligent Cloud Resource Query Processing Using Knowledge Graphs 

**Title (ZH)**: 基于知识图谱的智能云资源查询处理中的AI增强本体驱动自然语言处理 

**Authors**: Krishna Chaitanya Sunkara, Krishnaiah Narukulla  

**Link**: [PDF](https://arxiv.org/pdf/2502.18484)  

**Abstract**: The conventional resource search in cloud infrastructure relies on keyword-based searches or GUIDs, which demand exact matches and significant user effort to locate resources. These conventional search approaches often fail to interpret the intent behind natural language queries, making resource discovery inefficient and inaccessible to users. Though there exists some form of NLP based search engines, they are limited and focused more on analyzing the NLP query itself and extracting identifiers to find the resources. But they fail to search resources based on their behavior or operations or their capabilities or relationships or features or business relevance or the dynamic changing state or the knowledge these resources have. The search criteria has been changing with the inundation of AI based services which involved discovering not just the requested resources and identifiers but seeking insights. The real intent of a search has never been to just to list the resources but with some actual context such as to understand causes of some behavior in the system, compliance checks, capacity estimations, network constraints, or troubleshooting or business insights. This paper proposes an advanced Natural Language Processing (NLP) enhanced by ontology-based semantics to enable intuitive, human-readable queries which allows users to actually discover the intent-of-search itself. By constructing an ontology of cloud resources, their interactions, and behaviors, the proposed framework enables dynamic intent extraction and relevance ranking using Latent Semantic Indexing (LSI) and AI models. It introduces an automated pipeline which integrates ontology extraction by AI powered data crawlers, building a semantic knowledge base for context aware resource discovery. 

**Abstract (ZH)**: 传统的云基础设施资源搜索依赖于基于关键词或全局唯一标识符（GUID）的搜索，这需要精确匹配和用户付出显著的努力才能定位资源。这些传统的搜索方法往往无法理解自然语言查询背后的意图，从而使资源发现过程低效且对用户不友好。尽管存在一些基于自然语言处理（NLP）的搜索引擎，但它们大多集中在分析NLP查询本身和提取标识符以定位资源上。然而，这些搜索引擎无法根据资源的行为、操作、功能、关系、特性、业务相关性或这些资源所具备的知识来进行搜索。随着基于AI的服务的普及，搜索标准也在发生变化，不仅仅是发现请求的资源和标识符，还涉及寻求洞察。搜索的真实意图从来不仅仅是为了列出资源，而是为了在某些上下文中有实际意义的信息，例如了解系统中某些行为的原因、合规检查、容量估算、网络限制、故障排除或业务洞察。本文提出了一种增强自然语言处理（NLP）的方法，结合基于本体的语义，以使用户能够进行直观、人机可读的查询，并能够发现搜索的意图本身。通过构建云资源及其交互和行为的本体，所提出的框架利用潜在语义索引（LSI）和AI模型实现了动态意图提取和相关性排名。该方法引入了一种自动处理管道，通过AI驱动的数据爬虫提取本体，并构建一个基于上下文的认知资源发现的语义知识库。 

---
# Modeling Churn in Recommender Systems with Aggregated Preferences 

**Title (ZH)**: 基于聚合偏好建模推荐系统中的流失模型 

**Authors**: Gur Keinan, Omer Ben-Porat  

**Link**: [PDF](https://arxiv.org/pdf/2502.18483)  

**Abstract**: While recommender systems (RSs) traditionally rely on extensive individual user data, regulatory and technological shifts necessitate reliance on aggregated user information. This shift significantly impacts the recommendation process, requiring RSs to engage in intensive exploration to identify user preferences. However, this approach risks user churn due to potentially unsatisfactory recommendations. In this paper, we propose a model that addresses the dual challenges of leveraging aggregated user information and mitigating churn risk. Our model assumes that the RS operates with a probabilistic prior over user types and aggregated satisfaction levels for various content types. We demonstrate that optimal policies naturally transition from exploration to exploitation in finite time, develop a branch-and-bound algorithm for computing these policies, and empirically validate its effectiveness. 

**Abstract (ZH)**: 在传统上，推荐系统（RS）依赖于大量的个体用户数据。然而，监管和技术的变化促使推荐系统转向依赖聚合的用户信息。这种转变显著地影响了推荐过程，要求RS们进行大量的探索以识别用户偏好。然而，这种探索方法可能会因为推荐结果不尽如人意而导致用户流失。在本文中，我们提出了一种模型，旨在应对利用聚合用户信息和减轻用户流失风险的双重挑战。我们的模型假设推荐系统在内容类型的各种聚合满意度水平上具备用户类型的概率先验知识。我们证明了最优策略在有限时间内自然地从探索过渡到利用，并开发了一种分支定界算法来计算这些策略，并通过实证验证了其有效性。 

---
# MDE: Modality Discrimination Enhancement for Multi-modal Recommendation 

**Title (ZH)**: MDE：模态鉴别能力增强在多模态推荐中的应用 

**Authors**: Hang Zhou, Yucheng Wang, Huijing Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2502.18481)  

**Abstract**: Multi-modal recommendation systems aim to enhance performance by integrating an item's content features across various modalities with user behavior data. Effective utilization of features from different modalities requires addressing two challenges: preserving semantic commonality across modalities (modality-shared) and capturing unique characteristics for each modality (modality-specific). Most existing approaches focus on aligning feature spaces across modalities, which helps represent modality-shared features. However, modality-specific distinctions are often neglected, especially when there are significant semantic variations between modalities. To address this, we propose a Modality Distinctiveness Enhancement (MDE) framework that prioritizes extracting modality-specific information to improve recommendation accuracy while maintaining shared features. MDE enhances differences across modalities through a novel multi-modal fusion module and introduces a node-level trade-off mechanism to balance cross-modal alignment and differentiation. Extensive experiments on three public datasets show that our approach significantly outperforms other state-of-the-art methods, demonstrating the effectiveness of jointly considering modality-shared and modality-specific features. 

**Abstract (ZH)**: 多模态推荐系统旨在通过整合项目内容特征与其他模态下的用户行为数据来提高性能。有效利用不同模态的特征需要解决两个挑战：保持模态之间的语义一致性（模态共享）和捕获每个模态的独特特征（模态特定）。大多数现有方法集中在对齐不同模态的特征空间，这有助于表示模态共享特征。然而，模态特有的差异往往被忽视，尤其是在不同模态之间存在显著的语义差异时。为了解决这一问题，我们提出了一种模态独特性增强（MDE）框架，该框架优先提取模态特定信息以改进推荐准确性，同时保持共享特征。MDE通过一个新颖的多模态融合模块增强了模态间的差异，并引入了一个节点级权衡机制，以平衡跨模态对齐和差异化。在三个公开数据集上的广泛实验表明，我们的方法显著优于其他最先进的方法，证明了同时考虑模态共享和模态特定特征的有效性。 

---
# QExplorer: Large Language Model Based Query Extraction for Toxic Content Exploration 

**Title (ZH)**: QExplorer：基于大规模语言模型的有毒内容查询提取 

**Authors**: Shaola Ren, Li Ke, Longtao Huang, Dehong Gao, Hui Xue  

**Link**: [PDF](https://arxiv.org/pdf/2502.18480)  

**Abstract**: Automatically extracting effective queries is challenging in information retrieval, especially in toxic content exploration, as such content is likely to be disguised. With the recent achievements in generative Large Language Model (LLM), we are able to leverage the capabilities of LLMs to extract effective queries for similar content exploration directly. This study proposes QExplorer, an approach of large language model based Query Extraction for toxic content Exploration. The QExplorer approach involves a 2-stage training process: instruction Supervised FineTuning (SFT) and preference alignment using Direct Preference Optimization (DPO), as well as the datasets construction with feedback of search system. To verify the effectiveness of QExplorer, a series of offline and online experiments are conducted on our real-world system. The offline empirical results demonstrate that the performance of our automatic query extraction outperforms that of several LLMs and humans. The online deployment shows a significant increase in the detection of toxic items. 

**Abstract (ZH)**: 在信息检索中，自动提取有效的查询是一项挑战，尤其是在有毒内容探索中，因为此类内容很可能被隐藏或伪装。得益于近期生成型大语言模型（LLM）的进展，我们可以利用LLM的能力直接提取类似内容探索的有效查询。本研究提出了一种基于大语言模型的查询提取方法，QExplorer，该方法用于有毒内容探索。QExplorer 方法包括两阶段的训练过程：指令监督微调（SFT）和使用直接偏好优化（DPO）的偏好对齐，以及通过搜索系统的反馈构建数据集。为了验证QExplorer的有效性，我们在实际系统中进行了离线和在线实验。离线实验的结果表明，我们的自动查询提取性能优于几种LLM和人类的表现。在线部署结果显示有毒物品的检测显著增加。 

---
# Disrupt Your Research Using Generative AI Powered ScienceSage 

**Title (ZH)**: 使用生成式AI驱动的ScienceSage颠覆您的研究 

**Authors**: Yong Zhang, Eric Herrison Gyamfi, Kelly Anderson, Sasha Roberts, Matt Barker  

**Link**: [PDF](https://arxiv.org/pdf/2502.18479)  

**Abstract**: Large Language Models (LLM) are disrupting science and research in different subjects and industries. Here we report a minimum-viable-product (MVP) web application called $\textbf{ScienceSage}$. It leverages generative artificial intelligence (GenAI) to help researchers disrupt the speed, magnitude and scope of product innovation. $\textbf{ScienceSage}$ enables researchers to build, store, update and query a knowledge base (KB). A KB codifies user's knowledge/information of a given domain in both vector index and knowledge graph (KG) index for efficient information retrieval and query. The knowledge/information can be extracted from user's textual documents, images, videos, audios and/or the research reports generated based on a research question and the latest relevant information on internet. The same set of KBs interconnect three functions on $\textbf{ScienceSage}$: 'Generate Research Report', 'Chat With Your Documents' and 'Chat With Anything'. We share our learning to encourage discussion and improvement of GenAI's role in scientific research. 

**Abstract (ZH)**: 大型语言模型（LLM）正在以不同学科和行业的方式颠覆科学研究。在此，我们报告一个最小可行产品（MVP）的网络应用程序——**ScienceSage**。该应用程序利用生成型人工智能（GenAI）来帮助研究人员提高产品创新的速度、规模和范围。**ScienceSage** 使研究人员能够构建、存储、更新和查询知识库（KB）。知识库通过向量索引和知识图谱（KG）索引来编码用户的领域知识/信息，以便高效地检索和查询。知识/信息可以从用户的文本文件、图像、视频、音频或基于研究问题生成的研究报告中提取，也可以从互联网上最新的相关信息中提取。同一套知识库连接**ScienceSage** 上的三个功能：“生成研究报告”、“与您的文档对话”和“与任何事物对话”。我们分享了我们的学习经验，以促进讨论和改进GenAI在科学研究中的作用。 

---
# Beyond Self-Consistency: Loss-Balanced Perturbation-Based Regularization Improves Industrial-Scale Ads Ranking 

**Title (ZH)**: 超越自一致性：基于损失平衡扰动的正则化方法能够改进工业规模广告排名 

**Authors**: Ilqar Ramazanli, Hamid Eghbalzadeh, Xiaoyi Liu, Yang Wang, Jiaxiang Fu, Kaushik Rangadurai, Sem Park, Bo Long, Xue Feng  

**Link**: [PDF](https://arxiv.org/pdf/2502.18478)  

**Abstract**: Perturbation-based regularization techniques address many challenges in industrial-scale large models, particularly with sparse labels, and emphasize consistency and invariance for perturbation in model predictions. One of the popular regularization techniques has been various forms of self-consistency, which involve making small modifications to input data while preserving contextual information and enforcing similar predictions through auxiliary loss functions. In this work, we explore the first successful application of perturbation-based regularization algorithms in large-scale ads ranking models, and further propose a novel regularization algorithm, namely, Loss-Balanced Small Perturbation Regularization (LSPR) that can be used in potentially any deep learning model. We have successfully demonstrate that both Self-Consistency Regularization approaches (SCR) and LSPR are scalable and can improve ads delivery systems. By conducting industrial-scale experiments, and numerical analysis, we additionally show that our proposed LSPR, performs consistently better compared to SCR, across various groups and signal availability setups. Finally, we report a successful application of the proposed LSPR in a billion-scale industrial ranking system, which to the best of our knowledge, is the first of its kind, and it is specially designed to address the various scalability challenges (e.g, various surfaces, geological locations, clients and so on) as we will mention in this paper. 

**Abstract (ZH)**: 基于扰动的正则化技术在处理工业规模大型模型中的诸多挑战，特别是稀疏标签问题时表现出色，强调了扰动的一致性和不变性在模型预测中的重要性。一种流行的正则化技术是各种形式的自一致性，这类技术涉及对输入数据进行微小修改以保留上下文信息，并通过辅助损失函数使预测结果保持相似性。在此项研究中，我们探讨了基于扰动的正则化算法在大规模广告排名模型中的首次成功应用，并进一步提出了一种新颖的正则化算法，即损失均衡小扰动正则化（Loss-Balanced Small Perturbation Regularization，简称LSPR），该算法可以应用于任何深度学习模型。我们已经成功地证明，自一致性正则化方法（Self-Consistency Regularization, SCR）和LSPR都是可扩展的，并能够改进广告推送系统。通过进行工业规模的实验和数值分析，我们还展示了我们的提出的LSPR算法在不同群体和信号可用性设置下表现得更加稳定且效果更优。最后，我们描述了一个成功应用LSPR算法的工业规模排序系统，在我们所知的情况下，这是首个此类应用，且特别设计以解决各种扩展性挑战（例如，不同的表面、地质位置、客户等），这些内容将在本文中详细说明。 

---
# Recommendations Beyond Catalogs: Diffusion Models for Personalized Generation 

**Title (ZH)**: 超越目录的推荐：面向个性化生成的扩散模型 

**Authors**: Gabriel Patron, Zhiwei Xu, Ishan Kapnadak, Felipe Maia Polo  

**Link**: [PDF](https://arxiv.org/pdf/2502.18477)  

**Abstract**: Modern recommender systems follow the guiding principle of serving the right user, the right item at the right time. One of their main limitations is that they are typically limited to items already in the catalog. We propose REcommendations BEyond CAtalogs, REBECA, a new class of probabilistic diffusion-based recommender systems that synthesize new items tailored to individual tastes rather than retrieve items from the catalog. REBECA combines efficient training in embedding space with a novel diffusion prior that only requires users' past ratings of items. We evaluate REBECA on real-world data and propose novel personalization metrics for generative recommender systems. Extensive experiments demonstrate that REBECA produces high-quality, personalized recommendations, generating images that align with users' unique preferences. 

**Abstract (ZH)**: 现代推荐系统遵循“为正确用户，在正确时间推荐正确物品”的指导原则。它们的主要局限之一是通常仅限于目录中的物品。我们提出了超越目录的推荐系统（REBECA），这是一种新型的概率扩散基推荐系统，能够合成符合个人口味的新物品，而不是从目录中检索物品。REBECA 结合了在嵌入空间中的高效训练方法，并引入了一种新颖的扩散先验，仅需使用用户对物品的历史评分。我们在真实数据上评估了 REBECA，并为生成推荐系统提出了新的个性化度量方法。广泛的实验表明，REBECA 能生成高质量且个性化的推荐，生成的图像能够与用户的独特偏好相匹配。 

---
# FinBloom: Knowledge Grounding Large Language Model with Real-time Financial Data 

**Title (ZH)**: FinBloom: 基于实时金融数据的知识约束大规模语言模型 

**Authors**: Ankur Sinha, Chaitanya Agarwal, Pekka Malo  

**Link**: [PDF](https://arxiv.org/pdf/2502.18471)  

**Abstract**: Large language models (LLMs) excel at generating human-like responses but often struggle with interactive tasks that require access to real-time information. This limitation poses challenges in finance, where models must access up-to-date information, such as recent news or price movements, to support decision-making. To address this, we introduce Financial Agent, a knowledge-grounding approach for LLMs to handle financial queries using real-time text and tabular data. Our contributions are threefold: First, we develop a Financial Context Dataset of over 50,000 financial queries paired with the required context. Second, we train FinBloom 7B, a custom 7 billion parameter LLM, on 14 million financial news articles from Reuters and Deutsche Presse-Agentur, alongside 12 million Securities and Exchange Commission (SEC) filings. Third, we fine-tune FinBloom 7B using the Financial Context Dataset to serve as a Financial Agent. This agent generates relevant financial context, enabling efficient real-time data retrieval to answer user queries. By reducing latency and eliminating the need for users to manually provide accurate data, our approach significantly enhances the capability of LLMs to handle dynamic financial tasks. Our proposed approach makes real-time financial decisions, algorithmic trading and other related tasks streamlined, and is valuable in contexts with high-velocity data flows. 

**Abstract (ZH)**: 大型语言模型（LLMs）在生成类人类响应方面表现出色，但在处理需要实时信息访问的交互任务时常常遇到困难。这一限制在金融领域尤为明显，因为模型必须访问最新的信息，如最近的新闻或价格变动，以支持决策制定。为了解决这一问题，我们引入了Financial Agent，这是一种知识接地的方法，用于使LLMs能够使用实时文本和表格数据处理金融查询。我们的贡献主要有三个方面：

首先，我们开发了一个包含超过50,000个金融查询及其所需背景信息的金融语境数据集。
其次，我们训练了一个自定义的70亿参数LLM FinBloom 7B，使用来自路透社和德国新闻社的1400万篇财务新闻文章，以及来自证券交易委员会（SEC）的1200万份文件。
第三，我们使用金融语境数据集对FinBloom 7B进行微调，将其作为Financial Agent使用。该代理可以生成相关金融背景信息，从而实现高效地实时数据检索，以回答用户查询。通过减少延迟并消除用户手动提供准确数据的需要，我们的方法大幅增强了LLMs处理动态金融任务的能力。我们提出的方法使实时金融决策、算法交易及其他相关任务变得更为顺畅，特别是在数据流高速度的环境中具有很高的价值。 

---
# Spatial-RAG: Spatial Retrieval Augmented Generation for Real-World Spatial Reasoning Questions 

**Title (ZH)**: 空间RAG：扩充生成模型在实际空间推理问题中的空间检索方法 

**Authors**: Dazhou Yu, Riyang Bao, Gengchen Mai, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.18470)  

**Abstract**: Spatial reasoning remains a challenge for Large Language Models (LLMs), which struggle with spatial data retrieval and reasoning. We propose Spatial Retrieval-Augmented Generation (Spatial-RAG), a framework that extends RAG to spatial tasks by integrating sparse spatial retrieval (spatial databases) and dense semantic retrieval (LLM-based similarity). A multi-objective ranking strategy balances spatial constraints and semantic relevance, while an LLM-guided generator ensures coherent responses. Experiments on a real-world tourism dataset show that Spatial-RAG significantly improves spatial question answering, bridging the gap between LLMs and spatial intelligence. 

**Abstract (ZH)**: 空间推理仍然是大型语言模型（LLMs）的一个挑战，它们在处理空间数据检索和推理方面存在困难。我们提出了空间检索增强生成（Spatial-RAG）框架，该框架通过结合稀疏空间检索（空间数据库）和密集语义检索（基于LLM的相似性），将RAG扩展到空间任务中。多目标排名策略平衡了空间约束和语义相关性，而LLM引导的生成器确保了连贯的响应。在现实世界的旅游数据集上的实验表明，Spatial-RAG 显著提高了空间问题的回答能力，弥补了LLMs在空间智能方面的差距。 

---
# Using LLM-Based Approaches to Enhance and Automate Topic Labeling 

**Title (ZH)**: 使用基于大语言模型的方法增强并自动化主题标注 

**Authors**: Trishia Khandelwal  

**Link**: [PDF](https://arxiv.org/pdf/2502.18469)  

**Abstract**: Topic modeling has become a crucial method for analyzing text data, particularly for extracting meaningful insights from large collections of documents. However, the output of these models typically consists of lists of keywords that require manual interpretation for precise labeling. This study explores the use of Large Language Models (LLMs) to automate and enhance topic labeling by generating more meaningful and contextually appropriate labels. After applying BERTopic for topic modeling, we explore different approaches to select keywords and document summaries within each topic, which are then fed into an LLM to generate labels. Each approach prioritizes different aspects, such as dominant themes or diversity, to assess their impact on label quality. Additionally, recognizing the lack of quantitative methods for evaluating topic labels, we propose a novel metric that measures how semantically representative a label is of all documents within a topic. 

**Abstract (ZH)**: 主题建模已成为分析文本数据的一种关键方法，特别是从大量文档集合中提取有意义的洞察。然而，这些模型的输出通常是一系列关键词，需要人工解释以进行精确的标签化。本研究探索了大型语言模型（LLMs）在自动化和增强主题标签化方面的应用，通过生成更加有意义且上下文相关性强的标签来提高其效果。在应用BERTopic进行主题建模之后，我们探索了不同的方法来选择每个主题内的关键词和文档摘要，然后将其输入LLM以生成标签。每种方法都侧重于不同的方面，如主导主题或多样性，以评估其对标签质量的影响。此外，鉴于缺乏可用于评估主题标签的定量方法，我们提出了一个新的评价指标，用于衡量标签在代表主题内所有文档的语义方面的程度。 

---
# Efficient Federated Search for Retrieval-Augmented Generation 

**Title (ZH)**: 高效的联邦搜索以增强生成检索 

**Authors**: Rachid Guerraoui, Anne-Marie Kermarrec, Diana Petrescu, Rafael Pires, Mathis Randl, Martijn de Vos  

**Link**: [PDF](https://arxiv.org/pdf/2502.19280)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across various domains but remain susceptible to hallucinations and inconsistencies, limiting their reliability. Retrieval-augmented generation (RAG) mitigates these issues by grounding model responses in external knowledge sources. Existing RAG workflows often leverage a single vector database, which is impractical in the common setting where information is distributed across multiple repositories. We introduce RAGRoute, a novel mechanism for federated RAG search. RAGRoute dynamically selects relevant data sources at query time using a lightweight neural network classifier. By not querying every data source, this approach significantly reduces query overhead, improves retrieval efficiency, and minimizes the retrieval of irrelevant information. We evaluate RAGRoute using the MIRAGE and MMLU benchmarks and demonstrate its effectiveness in retrieving relevant documents while reducing the number of queries. RAGRoute reduces the total number of queries up to 77.5% and communication volume up to 76.2%. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各个领域展现了卓越的能力，但仍然容易产生幻觉和不一致性，这限制了其可靠性。检索增强生成（RAG）通过将模型响应与外部知识源相结合，缓解了这些问题。现有的RAG工作流程通常依赖单一向量数据库，但在信息分散存储于多个存储库的情况下，这种方式往往是不切实际的。我们提出了RAGRoute，这是一种新颖的联邦RAG搜索机制。RAGRoute使用轻量级的神经网络分类器在查询时动态选择相关数据源。通过不查询每个数据源，这种方法显著减少了查询开销，提高了检索效率，并最小化了无关信息的检索。我们使用MIRAGE和MMLU基准评估了RAGRoute，并展示了其在获取相关文档的同时减少查询次数的效果。RAGRoute将总查询次数减少多达77.5%，通信量减少多达76.2%。 

---
# TestNUC: Enhancing Test-Time Computing Approaches through Neighboring Unlabeled Data Consistency 

**Title (ZH)**: TestNUC：通过未标记邻居数据一致性增强测试时计算方法 

**Authors**: Henry Peng Zou, Zhengyao Gu, Yue Zhou, Yankai Chen, Weizhi Zhang, Liancheng Fang, Yibo Wang, Yangning Li, Kay Liu, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.19163)  

**Abstract**: Test-time computing approaches, which leverage additional computational resources during inference, have been proven effective in enhancing large language model performance. This work introduces a novel, linearly scaling approach, TestNUC, that improves test-time predictions by leveraging the local consistency of neighboring unlabeled data-it classifies an input instance by considering not only the model's prediction on that instance but also on neighboring unlabeled instances. We evaluate TestNUC across eight diverse datasets, spanning intent classification, topic mining, domain discovery, and emotion detection, demonstrating its consistent superiority over baseline methods such as standard prompting and self-consistency. Furthermore, TestNUC can be seamlessly integrated with existing test-time computing approaches, substantially boosting their performance. Our analysis reveals that TestNUC scales effectively with increasing amounts of unlabeled data and performs robustly across different embedding models, making it practical for real-world applications. Our code is available at this https URL. 

**Abstract (ZH)**: 在推理过程中利用额外计算资源的测试时计算方法已被证明能够有效提升大型语言模型的性能。本文介绍了一种新颖的线性扩展方法——TestNUC，该方法通过利用周围未标记数据的一致性来改进测试时的预测结果——它通过考虑模型对该输入实例及其周围未标记实例的预测来对输入实例进行分类。我们利用八个具有代表性的数据集，涵盖了意图分类、主题挖掘、领域发现和情绪检测等多个领域，评估了TestNUC的表现，并展示了它优于诸如标准提示和自我一致性等基线方法的一致优势。此外，TestNUC可以无缝集成到现有的测试时计算方法中，显著提升了它们的性能。我们的分析显示，TestNUC在增加未标记数据量时能够有效扩展，并且在不同的嵌入模型上表现稳定，使其适用于实际应用。我们的代码可以在这里访问：this https URL。 

---
# On Aggregation Queries over Predicted Nearest Neighbors 

**Title (ZH)**: 关于预测最近邻的聚合查询 

**Authors**: Carrie Wang, Sihem Amer-Yahia, Laks V. S. Lakshmanan, Reynold Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.18803)  

**Abstract**: We introduce Aggregation Queries over Nearest Neighbors (AQNNs), a novel type of aggregation queries over the predicted neighborhood of a designated object. AQNNs are prevalent in modern applications where, for instance, a medical professional may want to compute "the average systolic blood pressure of patients whose predicted condition is similar to a given insomnia patient". Since prediction typically involves an expensive deep learning model or a human expert, we formulate query processing as the problem of returning an approximate aggregate by combining an expensive oracle and a cheaper model (e.g, a simple ML model) to compute the predictions. We design the Sampler with Precision-Recall in Target (SPRinT) framework for answering AQNNs. SPRinT consists of sampling, nearest neighbor refinement, and aggregation, and is tailored for various aggregation functions. It enjoys provable theoretical guarantees, including bounds on sample size and on error in approximate aggregates. Our extensive experiments on medical, e-commerce, and video datasets demonstrate that SPRinT consistently achieves the lowest aggregation error with minimal computation cost compared to its baselines. Scalability results show that SPRinT's execution time and aggregation error remain stable as the dataset size increases, confirming its suitability for large-scale applications. 

**Abstract (ZH)**: 我们引入了一种新型的聚合查询，即邻近查询聚合（Aggregation Queries over Nearest Neighbors, AQNNs），这是一种针对指定对象预测邻域进行聚合查询的新类型。AQNNs 在现代应用中非常普遍，例如，一名医疗专业人士可能需要计算“具有预测病症与给定失眠患者相似的患者的平均收缩压”。由于预测通常涉及昂贵的深度学习模型或人类专家，我们通过结合一个昂贵的Oracle（大数据源或权威数据集合）和一个更便宜的模型（例如简单的机器学习模型）来计算预测，从而将查询处理问题转化为求解近似聚合。为此，我们设计了针对AQNNs的Sample with Precision-Recall in Target（SPRinT）框架。SPRinT 框架包括抽样、最近邻精化和聚合等步骤，并且可以根据不同的聚合函数进行调整。它具有可证明的理论保证，包括样本大小的上限以及近似聚合误差的上限。在医疗、电子商务和视频等多个数据集上的大量实验表明，与基线方法相比，SPRinT 在最低聚合误差和最小计算成本方面表现出色。扩展性结果表明，随着数据集规模的增加，SPRinT 的执行时间和聚合误差保持稳定，这证明了它适用于大规模应用的适用性。 

---
# PII-Bench: Evaluating Query-Aware Privacy Protection Systems 

**Title (ZH)**: PII-Bench: 评估查询感知的隐私保护系统 

**Authors**: Hao Shen, Zhouhong Gu, Haokai Hong, Weili Han  

**Link**: [PDF](https://arxiv.org/pdf/2502.18545)  

**Abstract**: The widespread adoption of Large Language Models (LLMs) has raised significant privacy concerns regarding the exposure of personally identifiable information (PII) in user prompts. To address this challenge, we propose a query-unrelated PII masking strategy and introduce PII-Bench, the first comprehensive evaluation framework for assessing privacy protection systems. PII-Bench comprises 2,842 test samples across 55 fine-grained PII categories, featuring diverse scenarios from single-subject descriptions to complex multi-party interactions. Each sample is carefully crafted with a user query, context description, and standard answer indicating query-relevant PII. Our empirical evaluation reveals that while current models perform adequately in basic PII detection, they show significant limitations in determining PII query relevance. Even state-of-the-art LLMs struggle with this task, particularly in handling complex multi-subject scenarios, indicating substantial room for improvement in achieving intelligent PII masking. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的广泛应用引发了对用户提示中个人可识别信息（PII）曝光的重大隐私担忧。为应对这一挑战，我们提出了一种与查询无关的PII遮掩策略，并引入了PII-Bench，这是第一个全面评估隐私保护系统的评价框架。PII-Bench 包含来自55个细粒度PII类别的2,842个测试样本，涵盖了从单一主体描述到复杂多方交互的各种场景。每个样本都精心设计了用户查询、上下文描述以及反映查询相关PII的标准答案。我们的实证评估表明，当前模型在基本PII检测方面表现尚可，但在确定查询相关PII方面显示出显著的局限性。即使是最先进的LLMs，在处理复杂多主体场景时也存在困难，这表明在实现智能PII遮掩方面有较大的改进空间。 

---
# FilterRAG: Zero-Shot Informed Retrieval-Augmented Generation to Mitigate Hallucinations in VQA 

**Title (ZH)**: FilterRAG：减轻VQA中幻觉现象的零样本引导检索增强生成方法 

**Authors**: S M Sarwar  

**Link**: [PDF](https://arxiv.org/pdf/2502.18536)  

**Abstract**: Visual Question Answering requires models to generate accurate answers by integrating visual and textual understanding. However, VQA models still struggle with hallucinations, producing convincing but incorrect answers, particularly in knowledge-driven and Out-of-Distribution scenarios. We introduce FilterRAG, a retrieval-augmented framework that combines BLIP-VQA with Retrieval-Augmented Generation to ground answers in external knowledge sources like Wikipedia and DBpedia. FilterRAG achieves 36.5% accuracy on the OK-VQA dataset, demonstrating its effectiveness in reducing hallucinations and improving robustness in both in-domain and Out-of-Distribution settings. These findings highlight the potential of FilterRAG to improve Visual Question Answering systems for real-world deployment. 

**Abstract (ZH)**: 视觉问答要求模型通过结合视觉和文本理解来生成准确的答案。然而，视觉问答（VQA）模型在应对幻觉问题时仍然存在挑战，特别是在知识驱动和分布外（Out-of-Distribution, OOD）场景中生成富有说服力但错误的答案。我们引入了FilterRAG，这是一种检索增强框架，将BLIP-VQA与检索增强生成相结合，利用外部知识源（如维基百科和DBpedia）来验证答案。在OK-VQA数据集上，FilterRAG实现了36.5%的准确率，证明了其在减少幻觉和提高系统鲁棒性方面的有效性能，无论是对于领域内还是分布外的场景。这些发现突显了FilterRAG在提升实际部署中视觉问答系统性能方面的潜力。 

---
# A Comprehensive Survey on Composed Image Retrieval 

**Title (ZH)**: 全面综述合成图像检索 

**Authors**: Xuemeng Song, Haoqiang Lin, Haokun Wen, Bohan Hou, Mingzhu Xu, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2502.18495)  

**Abstract**: Composed Image Retrieval (CIR) is an emerging yet challenging task that allows users to search for target images using a multimodal query, comprising a reference image and a modification text specifying the user's desired changes to the reference image. Given its significant academic and practical value, CIR has become a rapidly growing area of interest in the computer vision and machine learning communities, particularly with the advances in deep learning. To the best of our knowledge, there is currently no comprehensive review of CIR to provide a timely overview of this field. Therefore, we synthesize insights from over 120 publications in top conferences and journals, including ACM TOIS, SIGIR, and CVPR In particular, we systematically categorize existing supervised CIR and zero-shot CIR models using a fine-grained taxonomy. For a comprehensive review, we also briefly discuss approaches for tasks closely related to CIR, such as attribute-based CIR and dialog-based CIR. Additionally, we summarize benchmark datasets for evaluation and analyze existing supervised and zero-shot CIR methods by comparing experimental results across multiple datasets. Furthermore, we present promising future directions in this field, offering practical insights for researchers interested in further exploration. 

**Abstract (ZH)**: 合成图像检索（Composed Image Retrieval, CIR）是一项新兴且具有挑战性的任务，允许用户使用包含参考图像和指定修改内容的文字说明的多模态查询来查找目标图像。鉴于其重要的学术和实际价值，CIR 成为了计算机视觉和机器学习社区中的一个快速发展的研究领域，尤其是在深度学习技术不断进步的背景下。据我们所知，目前尚无全面的综述来提供该领域的及时概述。因此，我们综合了120余篇在顶级会议和期刊（包括ACM TOIS、SIGIR和CVPR）上发表的论文的见解，并从细粒度分类的角度系统地归类了现有的监督式CIR和零样本CIR模型。为了进行全面的综述，我们还简要讨论了一些与CIR密切相关的任务，例如基于属性的CIR和基于对话的CIR。此外，我们总结了用于评估的基准数据集，并通过对比多个数据集上的实验结果分析了现有的监督式和零样本CIR方法。最后，我们还提出了该领域的未来研究方向，为希望进一步探索这一领域的研究人员提供了实用的见解。 

---
# MixLLM: Dynamic Routing in Mixed Large Language Models 

**Title (ZH)**: 当然，以下是该标题的学术规范中文翻译：

MixLLM：混合大型语言模型中的动态路由 

**Authors**: Xinyuan Wang, Yanchi Liu, Wei Cheng, Xujiang Zhao, Zhengzhang Chen, Wenchao Yu, Yanjie Fu, Haifeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.18482)  

**Abstract**: Large Language Models (LLMs) exhibit potential artificial generic intelligence recently, however, their usage is costly with high response latency. Given mixed LLMs with their own strengths and weaknesses, LLM routing aims to identify the most suitable model for each query in the stream to maximize response quality and minimize cost and latency. However, the challenges involve: (1) dynamic trade-offs among quality, cost, and latency; (2) enabling continual learning in deployed systems; and (3) navigating a varying (e.g., new LLM addition or old LLM removal) set of LLM candidates over time. To bridge these gaps, we develop MixLLM, a dynamic contextual-bandit-based routing system for query-LLM assignment. Specifically, we first leverage query tags to enhance query embeddings for the routing task. Next, we design lightweight prediction models to estimate the response qualities and costs of queries over LLMs. We then devise a meta-decision maker to choose the query-LLM assignments to best tradeoff response quality, cost, and latency. Finally, the system benefits from continual training, allowing it to adapt to evolving queries and user feedback over time. Our extensive experiments show that MixLLM achieves the best trade-offs in response quality, cost, and latency (97.25% of GPT-4's quality at 24.18% of the cost under the time constraint). 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）展示了潜在的人工通用智能能力，然而其使用成本高昂且响应延迟较高。鉴于LLMs各自具备各自的优点与缺点，LLM路由旨在识别每条查询在流中的最优模型，以最大化响应质量并最小化成本和延迟。然而，面对的挑战包括：（1）质量、成本和延迟之间的动态权衡；（2）在部署系统中启用持续学习；以及（3）随着时间推移导航变动的LLM候选项集（例如，新的LLM加入或旧的LLM移除）。为了弥合这些差距，我们开发了MixLLM，这是一种基于动态上下文-上止博弈的路由系统，专门用于查询-LLM分配。具体而言，我们首先利用查询标签来增强查询嵌入，以提高路由任务的效果。接下来，我们设计了轻量级预测模型来估计查询在不同LLM上的响应质量和成本。然后，我们设计了一个元决策者来选择最佳的查询-LLM分配，以权衡响应质量、成本和延迟。最后，该系统通过持续训练受益，使其能够随着时间推移适应不断变化的查询和用户反馈。我们的 extensive 实验表明，MixLLM 在响应质量和延迟方面实现了最佳的权衡（在时间约束条件下，其质量达到了GPT-4的97.25%，成本仅为GPT-4的24.18%）。 

---
