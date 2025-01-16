# MMDocIR: Benchmarking Multi-Modal Retrieval for Long Documents 

**Title (ZH)**: MMDocIR：长文档多模态检索的基准测试 

**Authors**: Kuicai Dong, Yujing Chang, Xin Deik Goh, Dexun Li, Ruiming Tang, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.08828)  

**Abstract**: Multi-modal document retrieval is designed to identify and retrieve various forms of multi-modal content, such as figures, tables, charts, and layout information from extensive documents. Despite its significance, there is a notable lack of a robust benchmark to effectively evaluate the performance of systems in multi-modal document retrieval. To address this gap, this work introduces a new benchmark, named as MMDocIR, encompassing two distinct tasks: page-level and layout-level retrieval. The former focuses on localizing the most relevant pages within a long document, while the latter targets the detection of specific layouts, offering a more fine-grained granularity than whole-page analysis. A layout can refer to a variety of elements such as textual paragraphs, equations, figures, tables, or charts. The MMDocIR benchmark comprises a rich dataset featuring expertly annotated labels for 1,685 questions and bootstrapped labels for 173,843 questions, making it a pivotal resource for advancing multi-modal document retrieval for both training and evaluation. Through rigorous experiments, we reveal that (i) visual retrievers significantly outperform their text counterparts, (ii) MMDocIR train set can effectively benefit the training process of multi-modal document retrieval and (iii) text retrievers leveraging on VLM-text perform much better than those using OCR-text. These findings underscores the potential advantages of integrating visual elements for multi-modal document retrieval. 

**Abstract (ZH)**: 多模态文档检索旨在识别和检索各种多模态内容形式，如图表、表格、图表和布局信息，从广泛的文档中提取。尽管其具有重要意义，但缺乏一个稳健的基准来有效评估多模态文档检索系统的表现。为解决这一问题，本工作提出一个新的基准，名为MMDocIR，包括两个不同的任务：页面级检索和布局级检索。前者专注于在长文档中定位最相关的页面，而后者则专注于检测特定布局，提供比整个页面分析更精细的粒度。一个布局可以包括文本段落、公式、图表、表格或图表等多种元素。MMDocIR基准包括一个丰富数据集，包含1,685个问题的手动注释标签和173,843个问题的种子注释标签，使其成为推动多模态文档检索训练和评估的关键资源。通过严格的实验，我们揭示了以下几点：(i) 视觉检索器显著优于文本检索器，(ii) MMDocIR的训练集可以有效地提高多模态文档检索的训练过程，(iii) 基于VLM文本的文本检索器的表现明显优于基于OCR文本的文本检索器。这些发现强调了在多模态文档检索中集成视觉元素的潜在优势。 

---
# $\texttt{InfoHier}$: Hierarchical Information Extraction via Encoding and Embedding 

**Title (ZH)**: $\texttt{InfoHier}$：基于编码与嵌入的层次信息提取 

**Authors**: Tianru Zhang, Li Ju, Prashant Singh, Salman Toor  

**Link**: [PDF](https://arxiv.org/pdf/2501.08717)  

**Abstract**: Analyzing large-scale datasets, especially involving complex and high-dimensional data like images, is particularly challenging. While self-supervised learning (SSL) has proven effective for learning representations from unlabelled data, it typically focuses on flat, non-hierarchical structures, missing the multi-level relationships present in many real-world datasets. Hierarchical clustering (HC) can uncover these relationships by organizing data into a tree-like structure, but it often relies on rigid similarity metrics that struggle to capture the complexity of diverse data types. To address these we envision $\texttt{InfoHier}$, a framework that combines SSL with HC to jointly learn robust latent representations and hierarchical structures. This approach leverages SSL to provide adaptive representations, enhancing HC's ability to capture complex patterns. Simultaneously, it integrates HC loss to refine SSL training, resulting in representations that are more attuned to the underlying information hierarchy. $\texttt{InfoHier}$ has the potential to improve the expressiveness and performance of both clustering and representation learning, offering significant benefits for data analysis, management, and information retrieval. 

**Abstract (ZH)**: 分析大规模数据集，特别是在涉及复杂且高维的数据，如图像时，尤其具有挑战性。虽然自监督学习（SSL）已被证明在从未标记数据中学习表示方面有效，但其通常侧重于平坦而非层次化的结构，未能捕捉到许多现实世界数据集中的多层次关系。层次聚类（HC）可以通过将数据组织成树状结构来揭示这些关系，但它往往依赖于刚性的相似性度量，难以捕捉多种数据类型之间的复杂性。为了解决这些问题，我们构想了一个结合SSL与HC的框架$\texttt{InfoHier}$，旨在联合学习稳健的潜在表示和层次结构。该方法利用SSL提供适应性表示，增强HC捕捉复杂模式的能力。同时，它整合了HC损失以优化SSL训练，从而产生更符合潜在信息层次的表示。$\texttt{InfoHier}$有望提升聚类和表示学习的表达能力和性能，为数据的分析、管理和信息检索提供显著优势。 

---
# Real-time Indexing for Large-scale Recommendation by Streaming Vector Quantization Retriever 

**Title (ZH)**: 大规模推荐中的实时索引方法：基于流式向量量化检索 

**Authors**: Xingyan Bin, Jianfei Cui, Wujie Yan, Zhichen Zhao, Xintian Han, Chongyang Yan, Feng Zhang, Xun Zhou, Qi Wu, Zuotao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.08695)  

**Abstract**: Retrievers, which form one of the most important recommendation stages, are responsible for efficiently selecting possible positive samples to the later stages under strict latency limitations. Because of this, large-scale systems always rely on approximate calculations and indexes to roughly shrink candidate scale, with a simple ranking model. Considering simple models lack the ability to produce precise predictions, most of the existing methods mainly focus on incorporating complicated ranking models. However, another fundamental problem of index effectiveness remains unresolved, which also bottlenecks complication. In this paper, we propose a novel index structure: streaming Vector Quantization model, as a new generation of retrieval paradigm. Streaming VQ attaches items with indexes in real time, granting it immediacy. Moreover, through meticulous verification of possible variants, it achieves additional benefits like index balancing and reparability, enabling it to support complicated ranking models as existing approaches. As a lightweight and implementation-friendly architecture, streaming VQ has been deployed and replaced all major retrievers in Douyin and Douyin Lite, resulting in remarkable user engagement gain. 

**Abstract (ZH)**: 检索器是推荐流程中至关重要的一环，负责在严格的时间延迟限制下高效地从候选池中挑选出可能的正面样本传递给后续阶段。因此，在大规模系统中，通常依赖近似计算和索引来粗略缩小候选样本规模，并采用简单的排名模型。由于简单的模型缺乏生成精确预测的能力，现有大多数方法主要集中在整合复杂的排名模型。然而，索引有效性的一个基本问题仍未解决，这也限制了复杂模型的发展。本文提出了一种新的索引结构：流式向量量化模型（Streaming Vector Quantization, SVQ），这是一种新的检索范式。流式向量量化模型能够实时为项目分配索引，使其具有即时性。此外，通过细致验证可能的变体，它还实现了索引平衡和可修复性，能够支持现有的复杂排名模型。作为一种轻量级且易于实现的架构，流式向量量化模型已被部署并取代了抖音（Douyin）和抖音轻应用（Douyin Lite）中的所有主要检索器，显著提升了用户参与度。 

---
# Continuous Approach to Phase (Norm) Retrieval Frames 

**Title (ZH)**: 连续方法在相位（范数）恢复框架中的应用 

**Authors**: Ramin Farshchian, Rajab Ali Kamyabi-Gol, Fahimeh Arabyani-Neyshaburi, Fatemeh Esmaeelzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2501.08927)  

**Abstract**: This paper investigates the properties of continuous frames, with a particular focus on phase retrieval and norm retrieval in the context of Hilbert spaces. We introduce the concept of continuous near-Riesz bases and prove their invariance under invertible operators. Some equivalent conditions for phase and norm retrieval property of continuous frames are presented. We study the stability of phase retrieval under perturbations. Furthermore, tensor product frames for separable Hilbert spaces are studied, and we establish the equivalence of phase retrieval and norm retrieval properties between components and their tensor products. 

**Abstract (ZH)**: 本文研究了连续框架的性质，特别关注在希尔伯特空间中相位检索和范数检索的问题。我们引入了连续接近Riesz基的概念，并证明了其在可逆算子下的不变性。我们提出了相位检索和范数检索性质的若干等价条件。此外，我们研究了相位检索在扰动下的稳定性。进一步地，我们研究了可分希尔伯特空间中的张量积框架，并建立了各组成部分与其张量积之间的相位检索和范数检索性质的等价性。 

---
# Knowledge Graph-based Retrieval-Augmented Generation for Schema Matching 

**Title (ZH)**: 基于知识图谱的检索增强生成方法在模式匹配中的应用 

**Authors**: Chuangtao Ma, Sriom Chakrabarti, Arijit Khan, Bálint Molnár  

**Link**: [PDF](https://arxiv.org/pdf/2501.08686)  

**Abstract**: Traditional similarity-based schema matching methods are incapable of resolving semantic ambiguities and conflicts in domain-specific complex mapping scenarios due to missing commonsense and domain-specific knowledge. The hallucination problem of large language models (LLMs) also makes it challenging for LLM-based schema matching to address the above issues. Therefore, we propose a Knowledge Graph-based Retrieval-Augmented Generation model for Schema Matching, referred to as the KG-RAG4SM. In particular, KG-RAG4SM introduces novel vector-based, graph traversal-based, and query-based graph retrievals, as well as a hybrid approach and ranking schemes that identify the most relevant subgraphs from external large knowledge graphs (KGs). We showcase that KG-based retrieval-augmented LLMs are capable of generating more accurate results for complex matching cases without any re-training. Our experimental results show that KG-RAG4SM outperforms the LLM-based state-of-the-art (SOTA) methods (e.g., Jellyfish-8B) by 35.89% and 30.50% in terms of precision and F1 score on the MIMIC dataset, respectively; KG-RAG4SM with GPT-4o-mini outperforms the pre-trained language model (PLM)-based SOTA methods (e.g., SMAT) by 69.20% and 21.97% in terms of precision and F1 score on the Synthea dataset, respectively. The results also demonstrate that our approach is more efficient in end-to-end schema matching, and scales to retrieve from large KGs. Our case studies on the dataset from the real-world schema matching scenario exhibit that the hallucination problem of LLMs for schema matching is well mitigated by our solution. 

**Abstract (ZH)**: 传统基于相似性的模式匹配方法在处理领域特定复杂的映射场景时难以解决语义歧义和冲突，因为这些方法缺乏常识和领域特定的知识。大型语言模型（LLMs）的幻觉问题也使其难以解决上述问题。因此，我们提出了一种基于知识图谱的检索增强生成模型，称之为KG-RAG4SM。特别地，KG-RAG4SM 引入了基于向量的、基于图遍历的和基于查询的图检索方法，以及一种混合方法和排名方案，从而从外部大型知识图谱（KGs）中识别出最相关的子图。我们展示了基于知识图谱的检索增强LLMs能够在无需重新训练的情况下生成更准确的结果。实验结果表明，KG-RAG4SM 在 MIMIC 数据集上的准确率和F1 分数分别比基于LLMs的最先进的（SOTA）方法（例如Jellyfish-8B）高出35.89%和30.50%；使用GPT-4o-mini的KG-RAG4SM在Synthea数据集上的准确率和F1 分数分别比基于预训练语言模型（PLM）的SOTA 方法（例如SMAT）高69.20%和21.97%。实验结果还表明，我们的方法在端到端模式匹配中更有效率，并且能够扩展到从大规模知识图谱中检索。我们在实际模式匹配场景中的数据集上进行的案例研究显示，通过我们的解决方案可以很好地缓解LLMs在模式匹配中的幻觉问题。 

---
# DNMDR: Dynamic Networks and Multi-view Drug Representations for Safe Medication Recommendation 

**Title (ZH)**: DNMDR：动态网络和多视图药物表示的安全药物推荐 

**Authors**: Guanlin Liu, Xiaomei Yu, Zihao Liu, Xue Li, Xingxu Fan, Xiangwei Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2501.08572)  

**Abstract**: Medication Recommendation (MR) is a promising research topic which booms diverse applications in the healthcare and clinical domains. However, existing methods mainly rely on sequential modeling and static graphs for representation learning, which ignore the dynamic correlations in diverse medical events of a patient's temporal visits, leading to insufficient global structural exploration on nodes. Additionally, mitigating drug-drug interactions (DDIs) is another issue determining the utility of the MR systems. To address the challenges mentioned above, this paper proposes a novel MR method with the integration of dynamic networks and multi-view drug representations (DNMDR). Specifically, weighted snapshot sequences for dynamic heterogeneous networks are constructed based on discrete visits in temporal EHRs, and all the dynamic networks are jointly trained to gain both structural correlations in diverse medical events and temporal dependency in historical health conditions, for achieving comprehensive patient representations with both semantic features and structural relationships. Moreover, combining the drug co-occurrences and adverse drug-drug interactions (DDIs) in internal view of drug molecule structure and interactive view of drug pairs, the safe drug representations are available to obtain high-quality medication combination recommendation. Finally, extensive experiments on real world datasets are conducted for performance evaluation, and the experimental results demonstrate that the proposed DNMDR method outperforms the state-of-the-art baseline models with a large margin on various metrics such as PRAUC, Jaccard, DDI rates and so on. 

**Abstract (ZH)**: 药物推荐（Medication Recommendation, MR）是医疗和临床领域中具有广阔应用前景的研究课题。然而，现有的方法主要依赖于顺序模型和静态图来进行表示学习，这忽略了患者随访过程中多样化医学事件之间的动态相关性，导致节点的全局结构探索不足。此外，降低药物-药物相互作用（Drug-Drug Interactions, DDIs）是决定MR系统实用性的另一个问题。为了解决上述挑战，本文提出了一种名为动态网络与多视图药物表示集成（Dynamic Networks and Multi-view Drug Representations, DNMDR）的新型MR方法。具体而言，基于随时间变化的电子健康记录（EHRs）中的离散访问，构建加权快照序列以构建动态非同质网络；并将所有动态网络联合训练，以捕捉不同医学事件之间的结构相关性和历史健康条件的时间依赖性，从而实现具有语义特征和结构关系的综合患者表示。此外，通过结合药物共现及其交互视角中的内部视图药物分子结构和交互视图药物对中的负面DDI，得到安全的药物表示，以提供高质量的药物组合推荐。最后，在实际数据集上进行了广泛实验以评估性能，并通过多种度量指标（如PRAUC、Jaccard、DDI比率等）的实验结果表明，提出的DNMDR方法显著优于现有最佳基线模型。 

---
