# From Interets to Insights: An LLM Approach to Course Recommendations Using Natural Language Queries 

**Title (ZH)**: 从兴趣到洞见：基于自然语言查询的LLM课程推荐方法 

**Authors**: Hugh Van Deventer, Mark Mills, August Evrard  

**Link**: [PDF](https://arxiv.org/pdf/2412.19312)  

**Abstract**: Most universities in the United States encourage their students to explore academic areas before declaring a major and to acquire academic breadth by satisfying a variety of requirements. Each term, students must choose among many thousands of offerings, spanning dozens of subject areas, a handful of courses to take. The curricular environment is also dynamic, and poor communication and search functions on campus can limit a student's ability to discover new courses of interest. To support both students and their advisers in such a setting, we explore a novel Large Language Model (LLM) course recommendation system that applies a Retrieval Augmented Generation (RAG) method to the corpus of course descriptions. The system first generates an 'ideal' course description based on the user's query. This description is converted into a search vector using embeddings, which is then used to find actual courses with similar content by comparing embedding similarities. We describe the method and assess the quality and fairness of some example prompts. Steps to deploy a pilot system on campus are discussed. 

**Abstract (ZH)**: 美国大多数大学鼓励学生在正式选定专业之前探索学术领域，并通过满足各种要求来获得学科广度。每一学期，学生都需要从数千门课程中选择若干门课程，这些课程涵盖几十个学科领域。课程结构环境是动态变化的，而校园内的不良沟通和搜索功能可能限制学生发现新课程的能力。为了在这样一个环境中支持学生及其导师，我们探讨了一种新颖的大型语言模型（LLM）课程推荐系统，该系统采用检索增强生成（RAG）方法对课程描述语料库进行处理。该系统首先根据用户的查询生成一个“理想”的课程描述。然后，将该描述转换为嵌入向量，通过比较嵌入相似度来查找具有类似内容的实际课程。我们描述了这种方法，并评估了一些示例提示的质量和公平性。还讨论了在校园中部署试点系统的步骤。 

---
# Attack-in-the-Chain: Bootstrapping Large Language Models for Attacks Against Black-box Neural Ranking Models 

**Title (ZH)**: 链式攻击：用于黑盒神经排序模型攻击的大型语言模型的自我提升攻击方法 

**Authors**: Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2412.18770)  

**Abstract**: Neural ranking models (NRMs) have been shown to be highly effective in terms of retrieval performance. Unfortunately, they have also displayed a higher degree of sensitivity to attacks than previous generation models. To help expose and address this lack of robustness, we introduce a novel ranking attack framework named Attack-in-the-Chain, which tracks interactions between large language models (LLMs) and NRMs based on chain-of-thought (CoT) prompting to generate adversarial examples under black-box settings. Our approach starts by identifying anchor documents with higher ranking positions than the target document as nodes in the reasoning chain. We then dynamically assign the number of perturbation words to each node and prompt LLMs to execute attacks. Finally, we verify the attack performance of all nodes at each reasoning step and proceed to generate the next reasoning step. Empirical results on two web search benchmarks show the effectiveness of our method. 

**Abstract (ZH)**: 神经排序模型（NRMs）在检索性能方面已被证明非常有效。然而，它们对攻击的敏感性也较前一代模型更高。为了解决这一鲁棒性不足的问题，我们提出了一种名为“链中攻击”（Attack-in-the-Chain）的新型攻击框架，该框架基于链式思维（CoT）提示跟踪大规模语言模型（LLMs）与NRMs之间的交互，以生成对抗样本。我们的方法首先识别出排名高于目标文档的锚文档，并将其作为推理链中的节点。然后，动态分配每个节点的扰动词数量，并提示LLMs执行攻击。最后，我们在每一步推理中验证所有节点的攻击性能，并继续生成下一步推理。在两个网页搜索基准数据集上的实证结果表明了我们方法的有效性。 

---
# Bootstrap Your Own Context Length 

**Title (ZH)**: 自我生成情境长度 

**Authors**: Liang Wang, Nan Yang, Xingxing Zhang, Xiaolong Huang, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2412.18860)  

**Abstract**: We introduce a bootstrapping approach to train long-context language models by exploiting their short-context capabilities only. Our method utilizes a simple agent workflow to synthesize diverse long-context instruction tuning data, thereby eliminating the necessity for manual data collection and annotation. The proposed data synthesis workflow requires only a short-context language model, a text retriever, and a document collection, all of which are readily accessible within the open-source ecosystem. Subsequently, language models are fine-tuned using the synthesized data to extend their context lengths. In this manner, we effectively transfer the short-context capabilities of language models to long-context scenarios through a bootstrapping process. We conduct experiments with the open-source Llama-3 family of models and demonstrate that our method can successfully extend the context length to up to 1M tokens, achieving superior performance across various benchmarks. 

**Abstract (ZH)**: 我们提出了一种通过利用短语境语言模型的能力来训练长语境语言模型的bootstrapping方法。该方法利用简单的代理工作流合成长语境指令调整数据，从而消除了手动数据收集和标注的需求。所提出的数据合成工作流仅需要一个短语境语言模型、一个文本检索器和一个文档集合，这些资源在开源生态系统中都很容易获得。随后，使用合成数据对语言模型进行微调，以扩展其上下文长度。通过这种方式，我们有效地通过bootstrapping过程将语言模型的短语境能力转移到长语境场景中。我们使用开源Llama-3模型家族进行了实验，并展示了我们的方法能够将上下文长度扩展至多达100万词，且在各种基准测试中表现出更优的性能。 

---
# LLM-assisted vector similarity search 

**Title (ZH)**: LLM辅助向量相似性搜索 

**Authors**: Md Riyadh, Muqi Li, Felix Haryanto Lie, Jia Long Loh, Haotian Mi, Sayam Bohra  

**Link**: [PDF](https://arxiv.org/pdf/2412.18819)  

**Abstract**: As data retrieval demands become increasingly complex, traditional search methods often fall short in addressing nuanced and conceptual queries. Vector similarity search has emerged as a promising technique for finding semantically similar information efficiently. However, its effectiveness diminishes when handling intricate queries with contextual nuances. This paper explores a hybrid approach combining vector similarity search with Large Language Models (LLMs) to enhance search accuracy and relevance. The proposed two-step solution first employs vector similarity search to shortlist potential matches, followed by an LLM for context-aware ranking of the results. Experiments on structured datasets demonstrate that while vector similarity search alone performs well for straightforward queries, the LLM-assisted approach excels in processing complex queries involving constraints, negations, or conceptual requirements. By leveraging the natural language understanding capabilities of LLMs, this method improves the accuracy of search results for complex tasks without sacrificing efficiency. We also discuss real-world applications and propose directions for future research to refine and scale this technique for diverse datasets and use cases.
Original article: this https URL 

**Abstract (ZH)**: 随着数据检索需求变得越来越复杂，传统的搜索方法往往难以处理精练且概念上的查询。向量相似性搜索已经作为一种有希望的技术，能够高效地找到语义上相似的信息。然而，当处理包含上下文细微差别的复杂查询时，其有效性会下降。本文探讨了一种结合向量相似性搜索与大规模语言模型（LLMs）的混合方法，以提高搜索的准确性和相关性。提出的两步解决方案首先使用向量相似性搜索来筛选出潜在匹配项，然后使用LLM进行上下文感知的结果排名。实验表明，在结构化数据集上，仅依靠向量相似性搜索能够很好地处理简单的查询，而LLM辅助的方法在处理包含约束、否定或概念要求的复杂查询方面表现出色。通过利用LLMs的自然语言理解能力，这种方法能够提高复杂任务的搜索结果准确性，而不会牺牲效率。我们还讨论了该方法在实际应用中的应用，并提出了未来研究的方向，以进一步细化和扩展该技术以适用于不同的数据集和应用场景。
原始文章：https://your-link-here 

---
# Enhanced Recommendation Combining Collaborative Filtering and Large Language Models 

**Title (ZH)**: 结合协同过滤和大规模语言模型的增强推荐方法 

**Authors**: Xueting Lin, Zhan Cheng, Longfei Yun, Qingyi Lu, Yuanshuai Luo  

**Link**: [PDF](https://arxiv.org/pdf/2412.18713)  

**Abstract**: With the advent of the information explosion era, the importance of recommendation systems in various applications is increasingly significant. Traditional collaborative filtering algorithms are widely used due to their effectiveness in capturing user behavior patterns, but they encounter limitations when dealing with cold start problems and data sparsity. Large Language Models (LLMs), with their strong natural language understanding and generation capabilities, provide a new breakthrough for recommendation systems. This study proposes an enhanced recommendation method that combines collaborative filtering and LLMs, aiming to leverage collaborative filtering's advantage in modeling user preferences while enhancing the understanding of textual information about users and items through LLMs to improve recommendation accuracy and diversity. This paper first introduces the fundamental theories of collaborative filtering and LLMs, then designs a recommendation system architecture that integrates both, and validates the system's effectiveness through experiments. The results show that the hybrid model based on collaborative filtering and LLMs significantly improves precision, recall, and user satisfaction, demonstrating its potential in complex recommendation scenarios. 

**Abstract (ZH)**: 随着信息爆炸时代的到来，推荐系统在各种应用中的重要性越来越显著。传统的协同过滤算法因其能够有效捕捉用户行为模式而得到广泛应用，但在处理冷启动问题和数据稀疏性时遇到了限制。大型语言模型（LLMs）凭借其强大的自然语言理解和生成能力，为推荐系统提供了新的突破。本研究提出了一种结合协同过滤和LLMs的增强推荐方法，旨在利用协同过滤模型在用户偏好建模方面的优势，通过LLMs增强对用户和项目文本信息的理解，从而提高推荐准确性和多样性。本文首先介绍了协同过滤和LLMs的基本理论，然后设计了一个结合两者功能的推荐系统架构，并通过实验验证了该系统的有效性。结果表明，基于协同过滤和LLMs的混合模型在精确度、召回率和用户满意度方面显著提高，显示出其在复杂推荐场景中的潜力。 

---
# Can AI Help with Your Personal Finances? 

**Title (ZH)**: 人工智能能帮助您的个人财务管理吗？ 

**Authors**: Oudom Hean, Utsha Saha, Binita Saha  

**Link**: [PDF](https://arxiv.org/pdf/2412.19784)  

**Abstract**: In recent years, Large Language Models (LLMs) have emerged as a transformative development in artificial intelligence (AI), drawing significant attention from industry and academia. Trained on vast datasets, these sophisticated AI systems exhibit impressive natural language processing and content generation capabilities. This paper explores the potential of LLMs to address key challenges in personal finance, focusing on the United States. We evaluate several leading LLMs, including OpenAI's ChatGPT, Google's Gemini, Anthropic's Claude, and Meta's Llama, to assess their effectiveness in providing accurate financial advice on topics such as mortgages, taxes, loans, and investments. Our findings show that while these models achieve an average accuracy rate of approximately 70%, they also display notable limitations in certain areas. Specifically, LLMs struggle to provide accurate responses for complex financial queries, with performance varying significantly across different topics. Despite these limitations, the analysis reveals notable improvements in newer versions of these models, highlighting their growing utility for individuals and financial advisors. As these AI systems continue to evolve, their potential for advancing AI-driven applications in personal finance becomes increasingly promising. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）已成为人工智能（AI）领域的一个变革性发展，引起了业界和学术界的广泛关注。这些模型通过训练大量数据集，展现了卓越的自然语言处理和内容生成能力。本文探讨了LLMs在个人金融领域中解决关键挑战的潜力，重点关注美国。我们评估了几种领先的LLM模型，包括OpenAI的ChatGPT、Google的Gemini、Anthropic的Claude以及Meta的Llama，并评估其在提供关于按揭、税收、贷款和投资等财务建议方面的准确性和有效性。研究结果表明，尽管这些模型的平均准确率为约70%，但在某些领域仍表现出明显的局限性。具体而言，LLMs在处理复杂的财务查询时存在困难，不同主题的性能差异显著。尽管存在这些局限性，分析结果表明，这些模型的新版本在某些方面有了显著改进，突显了其在个人和财务顾问中的逐步实用性。随着这些人工智能系统的不断进化，它们在个人金融领域中推动AI驱动应用的潜力越来越具有前景。 

---
# Can Large Language Models Adapt to Other Agents In-Context? 

**Title (ZH)**: 大型语言模型能否在上下文中适应其他代理？ 

**Authors**: Matthew Riemer, Zahra Ashktorab, Djallel Bouneffouf, Payel Das, Miao Liu, Justin D. Weisz, Murray Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2412.19726)  

**Abstract**: As the research community aims to build better AI assistants that are more dynamic and personalized to the diversity of humans that they interact with, there is increased interest in evaluating the theory of mind capabilities of large language models (LLMs). Indeed, several recent studies suggest that LLM theory of mind capabilities are quite impressive, approximating human-level performance. Our paper aims to rebuke this narrative and argues instead that past studies were not directly measuring agent performance, potentially leading to findings that are illusory in nature as a result. We draw a strong distinction between what we call literal theory of mind i.e. measuring the agent's ability to predict the behavior of others and functional theory of mind i.e. adapting to agents in-context based on a rational response to predictions of their behavior. We find that top performing open source LLMs may display strong capabilities in literal theory of mind, depending on how they are prompted, but seem to struggle with functional theory of mind -- even when partner policies are exceedingly simple. Our work serves to highlight the double sided nature of inductive bias in LLMs when adapting to new situations. While this bias can lead to strong performance over limited horizons, it often hinders convergence to optimal long-term behavior. 

**Abstract (ZH)**: 随着研究社区致力于构建更具动态性和个性化的AI助手，以更好地适应与之交互的人类多样性，对大型语言模型（LLMs）的理论共情能力的评估变得越来越受欢迎。实际上，多项近期研究显示，LLMs的理论共情能力表现出色，接近人类水平。本文旨在反驳这一观点，并认为过去的许多研究并未直接衡量代理的实际表现，有可能导致一些虚幻的研究发现。我们严格区分了所谓的字面意义上的理论共情与功能性的理论共情。字面意义上的理论共情指的是测评代理预测他人行为的能力，而功能性理论共情则基于预测他人行为的理性响应来进行即境适应。我们发现，开源顶级的LLMs在字面意义上的理论共情方面可能会表现出很强的能力，但似乎在功能性理论共情方面遇到困难——即使伙伴策略非常简单也是如此。我们的工作突显了LLMs在适应新情况时的两面性引致偏见的特性。虽然这种偏见可以在短期内促进强大表现，但它往往阻碍了对最优长期行为的收敛。 

---
# Toward Adaptive Reasoning in Large Language Models with Thought Rollback 

**Title (ZH)**: 面向大型语言模型的自适应推理机制研究：反向思考方法探索 

**Authors**: Sijia Chen, Baochun Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.19707)  

**Abstract**: Large language models (LLMs) have been routinely used to solve various tasks using step-by-step reasoning. However, the structure of intermediate reasoning steps, or thoughts, is rigid and unidirectional, such as chains, trees, or acyclic-directed graphs. Consequently, the resulting inflexible and forward-only reasoning may not address challenging tasks and fail when the LLM frequently gives false responses, i.e., ``hallucinations''. This paper proposes a new reasoning framework, called Thought Rollback (TR), allowing LLMs to adaptively build thought structure while maintaining effective reasoning toward problem-solving under ``hallucinations''. The core mechanism of TR is rolling back thoughts, which allows LLMs to perform error analysis on thoughts, and thus roll back to any previously mistaken thought for revision. Subsequently, by including such trial-and-error in the prompt to guide the LLM, each rollback leads to one more reliable reasoning path. Therefore, starting with a simple prompt without human annotations, LLM with TR adaptively and gradually explores thoughts for a correct solution. Comprehensive experiments on mathematical problems and multi-task reasoning demonstrate the state-of-the-art performance of TR in terms of problem-solving rate and interaction cost. For instance, the solving rate of GPT-4 with TR outperforms the current best by $9\%$ on the MATH dataset. 

**Abstract (ZH)**: 大规模语言模型（LLMs）通常被用于通过逐步推理来解决各种任务。然而，中间推理步骤的结构是刚性的且单向的，例如链条、树结构或无环有向图。因此，这种僵化且只能向前的推理可能无法应对复杂的任务，并且当LLM频繁给出错误响应（即“幻觉”）时，会失效。本文提出了一种新的推理框架，称为“思维回滚”（Thought Rollback, TR），允许LLMs在遇到“幻觉”时能够适应性地构建推理结构，并在解决问题过程中进行有效的推理。TR的核心机制是回滚思维，这使得LLMs能够对思维进行错误分析，并回滚到之前的任何错误思维进行修正。在问题解决过程中，通过将这种试错过程包含在提示中以引导LLM，每次回滚都会产生一条更可靠的推理路径。因此，从简单的初始提示且无需人工注释开始，带有TR的LLM能够适配性地逐步探索思维，以找到正确的解决方案。在数学问题和多任务推理的综合实验中，TR在问题解决率和交互成本方面表现出最先进的性能。例如，带有TR的GPT-4在MATH数据集上的解决率为9%优于当前最佳性能。 

---
# Xmodel-2 Technical Report 

**Title (ZH)**: Xmodel-2 技术报告 

**Authors**: Wang Qun, Liu Yang, Lin Qingquan, Qu Zhijiu, Jiang Ling  

**Link**: [PDF](https://arxiv.org/pdf/2412.19638)  

**Abstract**: Xmodel-2 is a 1.2-billion-parameter large language model designed specifically for reasoning tasks. Its architecture enables different model scales to share a unified set of hyperparameters, allowing for extensive experimentation on smaller models and seamless transfer of optimal configurations to larger models. To maximize training efficiency and stability, Xmodel-2 employs the WSD learning rate scheduler from MiniCPM. Pretrained on 1.5 trillion tokens from diverse sources, Xmodel-2 achieves state-of-the-art performance in complex reasoning and agent-based tasks, while maintaining low training costs. These results highlight the potential of efficient model design and training strategies in advancing reasoning capabilities. Model checkpoints and code are publicly available on GitHub at this https URL 

**Abstract (ZH)**: Xmodel-2 是一个专门为推理任务设计的拥有 1.2 亿参数的大型语言模型。其架构允许不同规模的模型共享统一的超参数集，这既促进了小型模型的大规模实验，也实现了最优配置在大模型上的无缝转移。为了提高训练效率和稳定性，Xmodel-2 使用了 MiniCPM 中的 WSD 学习率调度器。Xmodel-2 采用来自多样化数据源的 1.5 万亿 tokens 进行预训练，在复杂推理和基于代理的任务中达到了最先进的性能，同时保持了较低的训练成本。这些结果突显了高效模型设计和训练策略在提升推理能力方面的重要潜力。Xmodel-2 的模型检查点和代码已公开发布在 GitHub 上，访问链接为 [该 https URL]。 

---
# Find the Intention of Instruction: Comprehensive Evaluation of Instruction Understanding for Large Language Models 

**Title (ZH)**: 探索指令意图：大型语言模型指令理解的综合评估 

**Authors**: Hyeonseok Moon, Jaehyung Seo, Seungyoon Lee, Chanjun Park, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2412.19450)  

**Abstract**: One of the key strengths of Large Language Models (LLMs) is their ability to interact with humans by generating appropriate responses to given instructions. This ability, known as instruction-following capability, has established a foundation for the use of LLMs across various fields and serves as a crucial metric for evaluating their performance. While numerous evaluation benchmarks have been developed, most focus solely on clear and coherent instructions. However, we have noted that LLMs can become easily distracted by instruction-formatted statements, which may lead to an oversight of their instruction comprehension skills. To address this issue, we introduce the Intention of Instruction (IoInst) benchmark. This benchmark evaluates LLMs' capacity to remain focused and understand instructions without being misled by extraneous instructions. The primary objective of this benchmark is to identify the appropriate instruction that accurately guides the generation of a given context. Our findings suggest that even recently introduced state-of-the-art models still lack instruction understanding capability. Along with the proposition of IoInst in this study, we also present broad analyses of the several strategies potentially applicable to IoInst. 

**Abstract (ZH)**: 大型语言模型（LLMs）的一个关键优势在于它们能够根据给定的指示生成适当的响应，这一能力被称为指令跟随能力。这种能力为LLMs在各个领域的应用奠定了基础，并且是评估其性能的重要指标。虽然已经开发出了许多评估基准，但大多数基准主要关注明确且连贯的指示。然而，我们注意到LLMs容易被格式化的指示陈述所分散，这可能导致它们忽略理解指示的能力。为了解决这一问题，我们提出了Intention of Instruction（IoInst）基准。此基准旨在评估LLMs保持专注并理解指示而不受无关指示误导的能力。该基准的主要目标是识别适当的指示，这些指示能够准确引导给定背景的生成。我们的研究发现即使是最新的先进模型在指示理解能力上仍然有所欠缺。除了在本研究中提出IoInst基准外，我们还讨论了几种潜在适用于IoInst的方法策略。 

---
# Multi-Attribute Constraint Satisfaction via Language Model Rewriting 

**Title (ZH)**: 基于语言模型重写实现多属性约束满足 

**Authors**: Ashutosh Baheti, Debanjana Chakraborty, Faeze Brahman, Ronan Le Bras, Ximing Lu, Nouha Dziri, Yejin Choi, Mark Riedl, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2412.19198)  

**Abstract**: Obeying precise constraints on top of multiple external attributes is a common computational problem underlying seemingly different domains, from controlled text generation to protein engineering. Existing language model (LM) controllability methods for multi-attribute constraint satisfaction often rely on specialized architectures or gradient-based classifiers, limiting their flexibility to work with arbitrary black-box evaluators and pretrained models. Current general-purpose large language models, while capable, cannot achieve fine-grained multi-attribute control over external attributes. Thus, we create Multi-Attribute Constraint Satisfaction (MACS), a generalized method capable of finetuning language models on any sequential domain to satisfy user-specified constraints on multiple external real-value attributes. Our method trains LMs as editors by sampling diverse multi-attribute edit pairs from an initial set of paraphrased outputs. During inference, LM iteratively improves upon its previous solution to satisfy constraints for all attributes by leveraging our designed constraint satisfaction reward. We additionally experiment with reward-weighted behavior cloning to further improve the constraint satisfaction rate of LMs. To evaluate our approach, we present a new Fine-grained Constraint Satisfaction (FineCS) benchmark, featuring two challenging tasks: (1) Text Style Transfer, where the goal is to simultaneously modify the sentiment and complexity of reviews, and (2) Protein Design, focusing on modulating fluorescence and stability of Green Fluorescent Proteins (GFP). Our empirical results show that MACS achieves the highest threshold satisfaction in both FineCS tasks, outperforming strong domain-specific baselines. Our work opens new avenues for generalized and real-value multi-attribute control, with implications for diverse applications spanning NLP and bioinformatics. 

**Abstract (ZH)**: 在多个外部属性的精确约束下进行计算，是不同领域中潜在看似不同的任务下的一个常见计算问题，从受控文本生成到蛋白质工程都是如此。现有的面向多属性约束的语言模型（LM）可控方法通常依赖于专门的架构或基于梯度的分类器，这限制了它们与任意的黑盒评估器和预训练模型的灵活性。尽管通用的大规模语言模型具备强大的能力，但在实现对外部属性的精细多属性控制方面仍然存在不足。因此，我们提出了多属性约束满足方法（MACS），这是一种泛化的框架，能够在任何序列表现领域中对语言模型进行微调，以满足用户指定的多个外部实值约束。该方法通过从初始的同义版本输出集中采样多样化的多属性编辑配对来训练LM作为编辑器。在推理过程中，LM通过利用我们设计的约束满足奖励逐步改进其先前的解决方案，以满足所有属性的约束。此外，我们还尝试使用奖励加权的行为克隆来进一步提高LM的约束满足率。为了评估该方法，我们提出了一种新的细粒度约束满足基准（FineCS），其包含两个具有挑战性的任务：（1）文本样式转换，目标是同时修改评论的情感和复杂性；（2）蛋白质设计，关注调整绿色荧光蛋白（GFP）的荧光强度和稳定性。我们的实验证据表明，MACS在两个FineCS任务中都实现了最高的约束满足阈值，超越了强大的领域特定基准。我们的工作为通用和实值多属性控制开辟了新的途径，具有跨自然语言处理和生物信息学等不同应用的广泛影响。 

---
# GAI: Generative Agents for Innovation 

**Title (ZH)**: GAI：生成式代理促进创新 

**Authors**: Masahiro Sato  

**Link**: [PDF](https://arxiv.org/pdf/2412.18899)  

**Abstract**: This study examines whether collective reasoning among generative agents can facilitate novel and coherent thinking that leads to innovation. To achieve this, it proposes GAI, a new LLM-empowered framework designed for reflection and interaction among multiple generative agents to replicate the process of innovation. The core of the GAI framework lies in an architecture that dynamically processes the internal states of agents and a dialogue scheme specifically tailored to facilitate analogy-driven innovation. The framework's functionality is evaluated using Dyson's invention of the bladeless fan as a case study, assessing the extent to which the core ideas of the innovation can be replicated through a set of fictional technical documents. The experimental results demonstrate that models with internal states significantly outperformed those without, achieving higher average scores and lower variance. Notably, the model with five heterogeneous agents equipped with internal states successfully replicated the key ideas underlying the Dyson's invention. This indicates that the internal state enables agents to refine their ideas, resulting in the construction and sharing of more coherent and comprehensive concepts. 

**Abstract (ZH)**: 本研究探讨集体生成代理间的集体推理是否能促进新颖且连贯的思考，进而推动创新。为此，本文提出了GAI（Generative Agent Innovation）框架，这是一种以大型语言模型（LLM）为基础的新框架，旨在促进多个生成代理间的反思和交互，复制创新过程。GAI框架的核心在于一种动态处理代理内部状态的架构，以及一种专门为促进类比驱动的创新而精心设计的对话方案。框架的功能性通过采用戴森公司发明的无叶风扇作为案例研究进行评估，评估在一组虚构的技术文件中复制创新核心理念的程度。实验结果表明，具有内部状态的模型显著优于没有内部状态的模型，获得了更高的平均评分和更低的变异性。值得注意的是，配备有内部状态的五个异构代理成功复制了戴森发明的关键理念。这表明内部状态使代理能够细化和精炼其理念，从而在构建和分享更多连贯和全面的概念方面取得成效。 

---
# CoEvo: Continual Evolution of Symbolic Solutions Using Large Language Models 

**Title (ZH)**: CoEvo：使用大型语言模型的符号解决方案持续进化 

**Authors**: Ping Guo, Qingfu Zhang, Xi Lin  

**Link**: [PDF](https://arxiv.org/pdf/2412.18890)  

**Abstract**: Large Language Models (LLMs) have emerged as transformative tools in artificial intelligence, capable of processing and understanding extensive human knowledge to enhance problem-solving across various domains. This paper explores the potential of LLMs to drive the discovery of symbolic solutions within scientific and engineering disciplines, where such solutions are crucial for advancing theoretical and practical applications. We propose a novel framework that utilizes LLMs in an evolutionary search methodology, augmented by a dynamic knowledge library that integrates and refines insights in an \textit{open-ended manner}. This approach aims to tackle the dual challenges of efficiently navigating complex symbolic representation spaces and leveraging both existing and newly generated knowledge to foster open-ended innovation. By enabling LLMs to interact with and expand upon a knowledge library, we facilitate the continuous generation of novel solutions in diverse forms such as language, code, and mathematical expressions. Our experimental results demonstrate that this method not only enhances the efficiency of searching for symbolic solutions but also supports the ongoing discovery process, akin to human scientific endeavors. This study represents a first effort in conceptualizing the search for symbolic solutions as a lifelong, iterative process, marking a significant step towards harnessing AI in the perpetual pursuit of scientific and engineering breakthroughs. We have open-sourced our code and data, please visit \url{this https URL} for more information. 

**Abstract (ZH)**: 大型语言模型（LLMs）已成为人工智能领域的变革性工具，能够处理和理解广泛的人类知识，从而在各个领域增强问题解决能力。本文探讨了LLMs在推动科学和工程学科中符号解决方案发现方面的潜在应用，这些解决方案对于推进理论和实践应用至关重要。我们提出了一种新颖的框架，该框架利用LLMs在进化搜索方法中的应用，并结合了一个动态知识库，以截然开放的方式集成和精炼见解。该方法旨在应对高效导航复杂的符号表示空间及利用现有和新生成的知识来促进开放创新的双重挑战。通过使LLMs与知识库发生交互，并扩展其中的知识，我们促进了多样化的新型解决方案——如语言、代码和数学表达式——的持续生成。实验结果表明，这种方法不仅提高了寻找符号解决方案的效率，还支持了持续的发现过程，类似于人类科学研究。这项研究代表了将符号解决方案的搜索概念化为终身迭代过程的一种初步尝试，标志着在利用AI追求科学和工程突破方面取得的重要一步。我们已开源了我们的代码和数据，请访问 \url{this https URL} 获取更多信息。 

---
# Agents on the Bench: Large Language Model Based Multi Agent Framework for Trustworthy Digital Justice 

**Title (ZH)**: 《审判席上的代理人：基于大型语言模型的多代理系统框架，以实现可信赖的数字正义》

此标题翻译遵循了学术规范，保持了原文的核心意思，并用中文学术界常用的表达方式进行了适当调整。 

**Authors**: Cong Jiang, Xiaolei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2412.18697)  

**Abstract**: The justice system has increasingly employed AI techniques to enhance efficiency, yet limitations remain in improving the quality of decision-making, particularly regarding transparency and explainability needed to uphold public trust in legal AI. To address these challenges, we propose a large language model based multi-agent framework named AgentsBench, which aims to simultaneously improve both efficiency and quality in judicial decision-making. Our approach leverages multiple LLM-driven agents that simulate the collaborative deliberation and decision making process of a judicial bench. We conducted experiments on legal judgment prediction task, and the results show that our framework outperforms existing LLM based methods in terms of performance and decision quality. By incorporating these elements, our framework reflects real-world judicial processes more closely, enhancing accuracy, fairness, and society consideration. AgentsBench provides a more nuanced and realistic methods of trustworthy AI decision-making, with strong potential for application across various case types and legal scenarios. 

**Abstract (ZH)**: 司法系统正越来越多地采用人工智能技术以提高效率，然而在提高决策质量方面仍存在局限，尤其是在透明性和可解释性方面，这是维护公众对法律人工智能的信任所需的重要方面。为应对这些挑战，我们提出了一种基于大型语言模型的多代理框架，名为AgentsBench，旨在同时提高司法决策的效率和质量。我们的方法利用多个由大型语言模型驱动的代理，模拟司法庭的协作讨论和决策过程。我们进行了法律判决预测任务的实验，结果表明，与现有的基于大型语言模型的方法相比，我们的框架在性能和决策质量上表现更优。通过这些元素，我们的框架更接近于真实的司法流程，提高了准确性、公平性和社会考量。AgentsBench 提供了一种更为细致和现实的可信赖人工智能决策方法，适用于各种案件类型和法律情境，具有较大的应用潜力。 

---
# CAD-GPT: Synthesising CAD Construction Sequence with Spatial Reasoning-Enhanced Multimodal LLMs 

**Title (ZH)**: CAD-GPT：增强空间推理的多模态LLM生成建筑施工序列 

**Authors**: Siyu Wang, Cailian Chen, Xinyi Le, Qimin Xu, Lei Xu, Yanzhou Zhang, Jie Yang  

**Link**: [PDF](https://arxiv.org/pdf/2412.19663)  

**Abstract**: Computer-aided design (CAD) significantly enhances the efficiency, accuracy, and innovation of design processes by enabling precise 2D and 3D modeling, extensive analysis, and optimization. Existing methods for creating CAD models rely on latent vectors or point clouds, which are difficult to obtain and costly to store. Recent advances in Multimodal Large Language Models (MLLMs) have inspired researchers to use natural language instructions and images for CAD model construction. However, these models still struggle with inferring accurate 3D spatial location and orientation, leading to inaccuracies in determining the spatial 3D starting points and extrusion directions for constructing geometries. This work introduces CAD-GPT, a CAD synthesis method with spatial reasoning-enhanced MLLM that takes either a single image or a textual description as input. To achieve precise spatial inference, our approach introduces a 3D Modeling Spatial Mechanism. This method maps 3D spatial positions and 3D sketch plane rotation angles into a 1D linguistic feature space using a specialized spatial unfolding mechanism, while discretizing 2D sketch coordinates into an appropriate planar space to enable precise determination of spatial starting position, sketch orientation, and 2D sketch coordinate translations. Extensive experiments demonstrate that CAD-GPT consistently outperforms existing state-of-the-art methods in CAD model synthesis, both quantitatively and qualitatively. 

**Abstract (ZH)**: 计算机辅助设计（CAD）显著提升了设计过程的效率、准确性和创新性，通过实现精确的二维和三维建模、广泛分析和优化。现有的CAD模型创建方法依赖于潜在向量或点云，这些方法的获取过程复杂且储存成本高昂。最近，多模态大型语言模型（MLLMs）的进步激发了研究人员使用自然语言指令和图像进行CAD模型构建。然而，这些模型仍然难以准确推断三维空间位置和方向，导致在构建几何图形时难以精确确定三维空间的起始点和拉伸方向。本项研究引入了CAD-GPT，这是一种具有增强空间推理的MLLM的CAD合成方法，可以接受单张图像或文本描述作为输入。为了实现精确的空间推断，我们的方法引入了三维建模空间机制。该方法利用专门的空间展开机制将三维空间位置和三维素描面旋转角度映射到一维语言特征空间，同时将二维素描坐标离散化到合适的平面空间，以实现对空间起始位置、素描方向和二维素描坐标转换的精确确定。大量实验结果显示，CAD-GPT在CAD模型合成方面的性能在定性和定量上都显著优于现有最先进的方法。 

---
# An Engorgio Prompt Makes Large Language Model Babble on 

**Title (ZH)**: 一个充血性的提示使大型语言模型胡言乱语 

**Authors**: Jianshuo Dong, Ziyuan Zhang, Qingjie Zhang, Han Qiu, Tianwei Zhang, Hao Wang, Hewu Li, Qi Li, Chao Zhang, Ke Xu  

**Link**: [PDF](https://arxiv.org/pdf/2412.19394)  

**Abstract**: Auto-regressive large language models (LLMs) have yielded impressive performance in many real-world tasks. However, the new paradigm of these LLMs also exposes novel threats. In this paper, we explore their vulnerability to inference cost attacks, where a malicious user crafts Engorgio prompts to intentionally increase the computation cost and latency of the inference process. We design Engorgio, a novel methodology, to efficiently generate adversarial Engorgio prompts to affect the target LLM's service availability. Engorgio has the following two technical contributions. (1) We employ a parameterized distribution to track LLMs' prediction trajectory. (2) Targeting the auto-regressive nature of LLMs' inference process, we propose novel loss functions to stably suppress the appearance of the <EOS> token, whose occurrence will interrupt the LLM's generation process. We conduct extensive experiments on 13 open-sourced LLMs with parameters ranging from 125M to 30B. The results show that Engorgio prompts can successfully induce LLMs to generate abnormally long outputs (i.e., roughly 2-13$\times$ longer to reach 90%+ of the output length limit) in a white-box scenario and our real-world experiment demonstrates Engergio's threat to LLM service with limited computing resources. The code is accessible at this https URL. 

**Abstract (ZH)**: 自回归大型语言模型（LLMs）在许多实际任务中展现了显著的性能。然而，这些LLMs的新范式也暴露出了新型威胁。本文探讨了它们在推理成本攻击下的脆弱性，其中恶意用户精心设计Engorgio提示，以故意增加推理过程的计算成本和延迟。我们设计了Engorgio这一新颖的方法论，以高效地生成对抗性Engorgio提示，影响目标LLM的服务可用性。Engorgio具有以下两个技术贡献：

(1) 我们使用参数化的分布追踪LLMs的预测轨迹。
(2) 针对LLMs推理过程的自回归性质，我们提出了新颖的损失函数，以稳定地抑制<EOS>标记的出现，该标记的出现会中断LLM的生成过程。

我们在参数从125M到30B的13个开源LLMs上进行了广泛实验。结果表明，在白盒场景下，Engorgio提示可以成功诱导LLMs生成异常长的输出（即，约为原长度的2-13倍，以达到90%以上的输出长度限制）。我们的现实世界实验进一步证明了Engorgio对LLM服务的威胁，即使资源有限。代码可在此处访问：https://github.com/your-repo-name。

请注意将`https://github.com/your-repo-name`替换为实际的代码库链接。 

---
# PlanLLM: Video Procedure Planning with Refinable Large Language Models 

**Title (ZH)**: PlanLLM：具有可细化大型语言模型的视频程序规划 

**Authors**: Dejie Yang, Zijing Zhao, YangLiu  

**Link**: [PDF](https://arxiv.org/pdf/2412.19139)  

**Abstract**: Video procedure planning, i.e., planning a sequence of action steps given the video frames of start and goal states, is an essential ability for embodied AI. Recent works utilize Large Language Models (LLMs) to generate enriched action step description texts to guide action step decoding. Although LLMs are introduced, these methods decode the action steps into a closed-set of one-hot vectors, limiting the model's capability of generalizing to new steps or tasks. Additionally, fixed action step descriptions based on world-level commonsense may contain noise in specific instances of visual states. In this paper, we propose PlanLLM, a cross-modal joint learning framework with LLMs for video procedure planning. We propose an LLM-Enhanced Planning module which fully uses the generalization ability of LLMs to produce free-form planning output and to enhance action step decoding. We also propose Mutual Information Maximization module to connect world-level commonsense of step descriptions and sample-specific information of visual states, enabling LLMs to employ the reasoning ability to generate step sequences. With the assistance of LLMs, our method can both closed-set and open vocabulary procedure planning tasks. Our PlanLLM achieves superior performance on three benchmarks, demonstrating the effectiveness of our designs. 

**Abstract (ZH)**: 视频操作规划，即根据起始状态和目标状态的视频帧规划一系列操作步骤，是具身人工智能的一项基本能力。最近的研究利用大型语言模型（LLMs）生成丰富的操作步骤描述文本，以指导操作步骤解码。尽管引入了LLMs，但这些方法将操作步骤解码为封闭集合中的一个热向量，限制了模型泛化到新步骤或任务的能力。此外，基于世界级常识固定的操作步骤描述在特定的视觉状态示例中可能包含噪声。在本文中，我们提出了一种名为PlanLLM的跨模态联合学习框架，该框架利用LLMs进行视频操作规划。我们提出了一种增强的规划模块，该模块充分利用了LLMs的泛化能力，生成自由形式的规划输出，并增强操作步骤解码。我们还提出了信息互信息最大化模块，将步骤描述的世界级常识与视觉状态的特定样本信息连接起来，使LLMs能够利用推理能力生成步骤序列。借助LLMs的帮助，我们的方法可以同时完成封闭集和开放词汇的操作规划任务。我们的PlanLLM在三个基准测试中表现出色，证明了我们设计的有效性。 

---
# Relation-aware Hierarchical Prompt for Open-vocabulary Scene Graph Generation 

**Title (ZH)**: 基于关系的分级提示在开放词汇场景图生成中的应用 

**Authors**: Tao Liu, Rongjie Li, Chongyu Wang, Xuming He  

**Link**: [PDF](https://arxiv.org/pdf/2412.19021)  

**Abstract**: Open-vocabulary Scene Graph Generation (OV-SGG) overcomes the limitations of the closed-set assumption by aligning visual relationship representations with open-vocabulary textual representations. This enables the identification of novel visual relationships, making it applicable to real-world scenarios with diverse relationships. However, existing OV-SGG methods are constrained by fixed text representations, limiting diversity and accuracy in image-text alignment. To address these challenges, we propose the Relation-Aware Hierarchical Prompting (RAHP) framework, which enhances text representation by integrating subject-object and region-specific relation information. Our approach utilizes entity clustering to address the complexity of relation triplet categories, enabling the effective integration of subject-object information. Additionally, we utilize a large language model (LLM) to generate detailed region-aware prompts, capturing fine-grained visual interactions and improving alignment between visual and textual modalities. RAHP also introduces a dynamic selection mechanism within Vision-Language Models (VLMs), which adaptively selects relevant text prompts based on the visual content, reducing noise from irrelevant prompts. Extensive experiments on the Visual Genome and Open Images v6 datasets demonstrate that our framework consistently achieves state-of-the-art performance, demonstrating its effectiveness in addressing the challenges of open-vocabulary scene graph generation. 

**Abstract (ZH)**: 开放词汇场景图生成（OV-SGG）通过将视觉关系表示与开放词汇的文本表示对齐，克服了封闭集假设的限制，从而能够识别新型视觉关系，使该方法适用于包含多种关系的现实世界场景。然而，现有的OV-SGG方法受限于固定的文字表示，这限制了图像-文本对齐的多样性和准确性。为了解决这些挑战，我们提出了关系感知分层提示（RAHP）框架，通过整合主语-宾语和区域特定的关系信息来增强文字表示。我们的方法利用实体聚类处理关系三元组类别的复杂性，从而能够有效整合主语-宾语信息。此外，我们利用大型语言模型（LLM）生成详细的区域感知提示，捕获细微的视觉交互，从而提高视觉与文本模态之间的对齐。RAHP还在视觉语言模型（VLMs）中引入了动态选择机制，该机制根据视觉内容自适应地选择相关文字提示，减少无关提示的噪音。在Visual Genome和Open Images v6数据集上的大量实验表明，我们的框架始终能够达到最佳性能，证明了其在解决开放词汇场景图生成挑战方面的有效性。 

---
# How Propense Are Large Language Models at Producing Code Smells? A Benchmarking Study 

**Title (ZH)**: 大型语言模型生成代码异味的概率有多大？一项基准研究 

**Authors**: Alejandro Velasco, Daniel Rodriguez-Cardenas, David N. Palacio, Luftar Rahman Alif, Denys Poshyvanyk  

**Link**: [PDF](https://arxiv.org/pdf/2412.18989)  

**Abstract**: Large Language Models (LLMs) have shown significant potential in automating software engineering tasks, particularly in code generation. However, current evaluation benchmarks, which primarily focus on accuracy, fall short in assessing the quality of the code generated by these models, specifically their tendency to produce code smells. To address this limitation, we introduce CodeSmellEval, a benchmark designed to evaluate the propensity of LLMs for generating code smells. Our benchmark includes a novel metric: Propensity Smelly Score (PSC), and a curated dataset of method-level code smells: CodeSmellData. To demonstrate the use of CodeSmellEval, we conducted a case study with two state-of-the-art LLMs, CodeLlama and Mistral. The results reveal that both models tend to generate code smells, such as simplifiable-condition and consider-merging-isinstance. These findings highlight the effectiveness of our benchmark in evaluating LLMs, providing valuable insights into their reliability and their propensity to introduce code smells in code generation tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自动化软件工程任务，特别是代码生成方面展现了显著的潜力。然而，当前主要基于准确性的评估基准在评估这些模型生成代码的质量时存在不足，尤其是它们生成代码异味的倾向。为解决这一局限性，我们引入了CodeSmellEval基准，旨在评估LLMs生成代码异味的倾向。该基准包括一个新颖的度量标准：代码异味倾向评分（PSC），以及一个精心收集的方法级代码异味数据集：CodeSmellData。为了展示CodeSmellEval的使用，我们对两种最先进的LLMs——CodeLlama和Mistral——进行了案例研究。结果显示，这两种模型都倾向于生成代码异味，如可简化条件和考虑合并(isinstance)等问题。这些发现突显了该基准在评估LLMs方面的有效性，提供了有关其可靠性和代码生成任务中引入代码异味倾向的重要见解。 

---
# HuatuoGPT-o1, Towards Medical Complex Reasoning with LLMs 

**Title (ZH)**: HuatuoGPT-o1：面向医学复杂推理的大型语言模型探究 

**Authors**: Junying Chen, Zhenyang Cai, Ke Ji, Xidong Wang, Wanlong Liu, Rongsheng Wang, Jianye Hou, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.18925)  

**Abstract**: The breakthrough of OpenAI o1 highlights the potential of enhancing reasoning to improve LLM. Yet, most research in reasoning has focused on mathematical tasks, leaving domains like medicine underexplored. The medical domain, though distinct from mathematics, also demands robust reasoning to provide reliable answers, given the high standards of healthcare. However, verifying medical reasoning is challenging, unlike those in mathematics. To address this, we propose verifiable medical problems with a medical verifier to check the correctness of model outputs. This verifiable nature enables advancements in medical reasoning through a two-stage approach: (1) using the verifier to guide the search for a complex reasoning trajectory for fine-tuning LLMs, (2) applying reinforcement learning (RL) with verifier-based rewards to enhance complex reasoning further. Finally, we introduce HuatuoGPT-o1, a medical LLM capable of complex reasoning, which outperforms general and medical-specific baselines using only 40K verifiable problems. Experiments show complex reasoning improves medical problem-solving and benefits more from RL. We hope our approach inspires advancements in reasoning across medical and other specialized domains. 

**Abstract (ZH)**: OpenAI o1的突破凸显了增强推理能力以提高大语言模型性能的潜力。然而，大多数关于推理的研究主要集中在数学任务上，而医学领域等其他领域则未得到充分探索。尽管医学与数学领域不同，但医学同样需要强大的推理能力以提供可靠的答案，因为医疗服务的标准非常高。然而，医学推理的验证比数学领域的验证更具挑战性。为解决这一问题，我们提出了一种带有医学验证器的可验证医学问题，以检查模型输出的正确性。这种可验证性通过两阶段方法促进了医学推理的进步：（1）使用验证器指导复杂的推理轨迹搜索，以微调大语言模型；（2）利用基于验证器的奖励强化学习（RL）来进一步增强复杂的推理能力。最后，我们介绍了HuatuoGPT-o1，这是一种能够进行复杂推理的医学大语言模型，仅使用40,000个可验证问题便超越了一般和专门针对医学问题的基准模型。实验结果显示，复杂的推理能够提高医学问题的解决能力，并且更受益于强化学习。我们希望我们的方法能够激励跨医学和其他专门领域中的推理进步。 

---
# Whose Morality Do They Speak? Unraveling Cultural Bias in Multilingual Language Models 

**Title (ZH)**: 他们代言的是哪种道德观？探究多语言语言模型中的文化偏见 

**Authors**: Meltem Aksoy  

**Link**: [PDF](https://arxiv.org/pdf/2412.18863)  

**Abstract**: Large language models (LLMs) have become integral tools in diverse domains, yet their moral reasoning capabilities across cultural and linguistic contexts remain underexplored. This study investigates whether multilingual LLMs, such as GPT-3.5-Turbo, GPT-4o-mini, Llama 3.1, and MistralNeMo, reflect culturally specific moral values or impose dominant moral norms, particularly those rooted in English. Using the updated Moral Foundations Questionnaire (MFQ-2) in eight languages, Arabic, Farsi, English, Spanish, Japanese, Chinese, French, and Russian, the study analyzes the models' adherence to six core moral foundations: care, equality, proportionality, loyalty, authority, and purity. The results reveal significant cultural and linguistic variability, challenging the assumption of universal moral consistency in LLMs. Although some models demonstrate adaptability to diverse contexts, others exhibit biases influenced by the composition of the training data. These findings underscore the need for culturally inclusive model development to improve fairness and trust in multilingual AI systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）已成为各个领域不可或缺的工具，但它们在跨文化与语言背景下进行道德推理的能力尚未得到充分探索。本研究考察了多语言LLMs，如GPT-3.5-Turbo、GPT-4o-mini、Llama 3.1和MistralNeMo，是否反映出了特定文化下的道德价值观，还是倾向于传播根植于英语的主导道德规范。研究使用阿拉伯语、波斯语、英语、西班牙语、日语、汉语、法语和俄语版本的更新版《道德基础问卷》（MFQ-2），分析了这六个核心道德基础：关爱、平等、适中、忠诚、权威和纯洁，以评估模型的道德倾向。研究结果揭示了显著的文化和语言差异，挑战了LLMs普遍道德一致性这一假设。尽管有些模型显示出适应不同背景的能力，但也有其他模型因其训练数据的构成而表现出偏见。这些发现强调了在多语言AI系统中进行文化包容性模型开发的必要性，以提高公平性和信任度。 

---
# PhyloGen: Language Model-Enhanced Phylogenetic Inference via Graph Structure Generation 

**Title (ZH)**: PhyloGen：通过图结构生成增强的语言模型辅助系统演化推断 

**Authors**: ChenRui Duan, Zelin Zang, Siyuan Li, Yongjie Xu, Stan Z. Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.18827)  

**Abstract**: Phylogenetic trees elucidate evolutionary relationships among species, but phylogenetic inference remains challenging due to the complexity of combining continuous (branch lengths) and discrete parameters (tree topology). Traditional Markov Chain Monte Carlo methods face slow convergence and computational burdens. Existing Variational Inference methods, which require pre-generated topologies and typically treat tree structures and branch lengths independently, may overlook critical sequence features, limiting their accuracy and flexibility. We propose PhyloGen, a novel method leveraging a pre-trained genomic language model to generate and optimize phylogenetic trees without dependence on evolutionary models or aligned sequence constraints. PhyloGen views phylogenetic inference as a conditionally constrained tree structure generation problem, jointly optimizing tree topology and branch lengths through three core modules: (i) Feature Extraction, (ii) PhyloTree Construction, and (iii) PhyloTree Structure Modeling. Meanwhile, we introduce a Scoring Function to guide the model towards a more stable gradient descent. We demonstrate the effectiveness and robustness of PhyloGen on eight real-world benchmark datasets. Visualization results confirm PhyloGen provides deeper insights into phylogenetic relationships. 

**Abstract (ZH)**: 系统树能够阐明物种之间的进化关系，但进化树推理由于连续参数（分支长度）和离散参数（树拓扑结构）的结合复杂性而仍然具有挑战性。传统马尔可夫链蒙特卡洛方法面临收敛缓慢和计算负担重的问题。现有的一些变分推断方法需要预先生成的拓扑结构，并通常独立处理树结构和分支长度，可能会忽略关键序列特征，限制了它们的准确性和灵活性。我们提出了一种名为PhyloGen的新方法，利用预训练的基因组语言模型生成和优化树，无需依赖进化模型或对齐序列约束。PhyloGen将进化树推理视为有条件约束的树结构生成问题，通过三个核心模块联合优化树拓扑结构和分支长度：(i) 特征提取，(ii) 进化树构建，和(iii) 进化树结构建模。此外，我们引入了一个评分函数来引导模型向更稳定的梯度下降方向发展。我们通过八个实际基准数据集验证了PhyloGen的有效性和鲁棒性。可视化结果表明，PhyloGen能够更深入地揭示进化关系。 

---
# Torque-Aware Momentum 

**Title (ZH)**: 扭矩感知动量 

**Authors**: Pranshu Malviya, Goncalo Mordido, Aristide Baratin, Reza Babanezhad Harikandeh, Gintare Karolina Dziugaite, Razvan Pascanu, Sarath Chandar  

**Link**: [PDF](https://arxiv.org/pdf/2412.18790)  

**Abstract**: Efficiently exploring complex loss landscapes is key to the performance of deep neural networks. While momentum-based optimizers are widely used in state-of-the-art setups, classical momentum can still struggle with large, misaligned gradients, leading to oscillations. To address this, we propose Torque-Aware Momentum (TAM), which introduces a damping factor based on the angle between the new gradients and previous momentum, stabilizing the update direction during training. Empirical results show that TAM, which can be combined with both SGD and Adam, enhances exploration, handles distribution shifts more effectively, and improves generalization performance across various tasks, including image classification and large language model fine-tuning, when compared to classical momentum-based optimizers. 

**Abstract (ZH)**: 高效探索复杂的损失景观是深度神经网络性能的关键。尽管基于动量的优化器在当今的先进设置中广泛使用，但经典的动量仍然难以处理大的、方向错位的梯度，导致振荡。为了解决这一问题，我们提出了一种基于扭矩的动量（Torque-Aware Momentum，TAM），它通过引入基于新梯度与先前动量之间夹角的阻尼因子来稳定训练中的更新方向。实验结果表明，TAM 可以与 SGD 和 Adam 等方法结合使用，能够增强探索能力，更好地处理分布偏移，并在包括图像分类和大规模语言模型微调在内的各种任务上提高泛化性能，优于传统的基于动量的优化器。 

---
# SAFLITE: Fuzzing Autonomous Systems via Large Language Models 

**Title (ZH)**: SAFLITE：通过大规模语言模型对自主系统进行模糊测试 

**Authors**: Taohong Zhu, Adrians Skapars, Fardeen Mackenzie, Declan Kehoe, William Newton, Suzanne Embury, Youcheng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2412.18727)  

**Abstract**: Fuzz testing effectively uncovers software vulnerabilities; however, it faces challenges with Autonomous Systems (AS) due to their vast search spaces and complex state spaces, which reflect the unpredictability and complexity of real-world environments. This paper presents a universal framework aimed at improving the efficiency of fuzz testing for AS. At its core is SaFliTe, a predictive component that evaluates whether a test case meets predefined safety criteria. By leveraging the large language model (LLM) with information about the test objective and the AS state, SaFliTe assesses the relevance of each test case. We evaluated SaFliTe by instantiating it with various LLMs, including GPT-3.5, Mistral-7B, and Llama2-7B, and integrating it into four fuzz testing tools: PGFuzz, DeepHyperion-UAV, CAMBA, and TUMB. These tools are designed specifically for testing autonomous drone control systems, such as ArduPilot, PX4, and PX4-Avoidance. The experimental results demonstrate that, compared to PGFuzz, SaFliTe increased the likelihood of selecting operations that triggered bug occurrences in each fuzzing iteration by an average of 93.1\%. Additionally, after integrating SaFliTe, the ability of DeepHyperion-UAV, CAMBA, and TUMB to generate test cases that caused system violations increased by 234.5\%, 33.3\%, and 17.8\%, respectively. The benchmark for this evaluation was sourced from a UAV Testing Competition. 

**Abstract (ZH)**: 模糊测试有效地揭示了软件漏洞，但在自主系统（AS）中面临挑战，因为AS具有庞大的搜索空间和复杂的状态空间，这反映了现实环境中的不可预测性和复杂性。本文提出了一种通用框架，旨在提高AS模糊测试的效率。其核心是SaFliTe，这是一种预测组件，用于评估测试案例是否满足预定义的安全标准。通过利用大型语言模型（LLM）中的测试目标和AS状态信息，SaFliTe评估每个测试案例的相关性。我们通过实例化不同的LLM，包括GPT-3.5、Mistral-7B和Llama2-7B，并将其集成到四种模糊测试工具：PGFuzz、DeepHyperion-UAV、CAMBA和TUMB中进行了评估。这些工具专门用于测试自主无人机控制系统，如ArduPilot、PX4和PX4-Avoidance。实验结果表明，与PGFuzz相比，SaFliTe在每次模糊测试迭代中选择触发错误的选项的可能性平均提高了93.1%。另外，集成SaFliTe后，DeepHyperion-UAV、CAMBA和TUMB生成导致系统违规的测试案例的能力分别提高了234.5%、33.3%和17.8%。此次评估的基准数据来自于一项无人机测试竞赛。 

---
# Diverse and Effective Red Teaming with Auto-generated Rewards and Multi-step Reinforcement Learning 

**Title (ZH)**: 使用自动生成奖励和多步强化学习的多样化和有效的红队演练 

**Authors**: Alex Beutel, Kai Xiao, Johannes Heidecke, Lilian Weng  

**Link**: [PDF](https://arxiv.org/pdf/2412.18693)  

**Abstract**: Automated red teaming can discover rare model failures and generate challenging examples that can be used for training or evaluation. However, a core challenge in automated red teaming is ensuring that the attacks are both diverse and effective. Prior methods typically succeed in optimizing either for diversity or for effectiveness, but rarely both. In this paper, we provide methods that enable automated red teaming to generate a large number of diverse and successful attacks.
Our approach decomposes the task into two steps: (1) automated methods for generating diverse attack goals and (2) generating effective attacks for those goals. While we provide multiple straightforward methods for generating diverse goals, our key contributions are to train an RL attacker that both follows those goals and generates diverse attacks for those goals. First, we demonstrate that it is easy to use a large language model (LLM) to generate diverse attacker goals with per-goal prompts and rewards, including rule-based rewards (RBRs) to grade whether the attacks are successful for the particular goal. Second, we demonstrate how training the attacker model with multi-step RL, where the model is rewarded for generating attacks that are different from past attempts further increases diversity while remaining effective. We use our approach to generate both prompt injection attacks and prompts that elicit unsafe responses. In both cases, we find that our approach is able to generate highly-effective and considerably more diverse attacks than past general red-teaming approaches. 

**Abstract (ZH)**: 自动化红队演练可以发现罕见的模型故障并生成具有挑战性的实例，这些实例可用于训练或评估。然而，自动化红队演练的核心挑战在于确保攻击既多样化又有效。以往的方法通常在优化多样性和有效性之间取得成功，但很少两者兼顾。在本文中，我们提供了一种方法，使自动化红队演练能够生成大量多样且成功的攻击。

我们的方法将任务分解为两个步骤：（1）生成多样化的攻击目标的自动化方法；（2）为这些目标生成有效的攻击。尽管我们提供了多种简单的方法来生成多样化的目标，但我们的关键贡献在于训练一个深度强化学习（RL）攻击者，该攻击者不仅遵循这些目标，还为每个目标生成多样化的攻击。首先，我们证明使用大型语言模型（LLMs）通过每个目标的提示和奖励（包括基于规则的奖励RBR，以评估攻击是否针对特定目标成功）来生成多样化的攻击者目标是很容易的。其次，我们展示通过使用多步强化学习训练攻击模型，其中模型因其生成的攻击与过去尝试不同而得到奖励，在保持有效性的同时还能增加多样性。我们使用我们的方法生成了注入提示攻击以及引发不安全响应的提示。在两种情况下，我们发现我们的方法生成的攻击不仅更为有效，且在多样性方面也明显优于以往的一般红队方法。 

---
# A Survey of NL2SQL with Large Language Models: Where are we, and where are we going? 

**Title (ZH)**: 大规模语言模型下的自然语言到结构化查询转换综述：我们在哪里，以及将要去向何处？ 

**Authors**: Xinyu Liu, Shuyu Shen, Boyan Li, Peixian Ma, Runzhi Jiang, Yuxin Zhang, Ju Fan, Guoliang Li, Nan Tang, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2408.05109)  

**Abstract**: Translating users' natural language queries (NL) into SQL queries (i.e., NL2SQL, a.k.a., Text-to-SQL) can significantly reduce barriers to accessing relational databases and support various commercial applications. The performance of NL2SQL has been greatly enhanced with the emergence of Large Language Models (LLMs). In this survey, we provide a comprehensive review of NL2SQL techniques powered by LLMs, covering its entire lifecycle from the following four aspects: (1) Model: NL2SQL translation techniques that tackle not only NL ambiguity and under-specification, but also properly map NL with database schema and instances; (2) Data: From the collection of training data, data synthesis due to training data scarcity, to NL2SQL benchmarks; (3) Evaluation: Evaluating NL2SQL methods from multiple angles using different metrics and granularities; and (4) Error Analysis: analyzing NL2SQL errors to find the root cause and guiding NL2SQL models to evolve. Moreover, we provide a rule of thumb for developing NL2SQL solutions. Finally, we discuss the research challenges and open problems of NL2SQL in the LLMs era. 

**Abstract (ZH)**: 将用户自然语言查询（NL）转换为SQL查询（即NL2SQL，又称Text-to-SQL）可以显著降低访问关系数据库的壁垒，并支持各种商业应用。随着大型语言模型（LLMs）的出现，NL2SQL性能得到了显著提升。本文综述了由LLMs赋能的NL2SQL技术，从以下四个方面全面回顾了其生命周期：（1）模型：NL2SQL转换技术不仅解决了NL的歧义性和欠具体性问题，还在合理的将NL映射到数据库模式和实例方面也加以处理；（2）数据：从训练数据的收集，因数据稀缺而进行的数据合成，到NL2SQL基准测试；（3）评估：从多个角度、使用不同指标和粒度对NL2SQL方法进行评估；（4）错误分析：分析NL2SQL的错误，找出根本原因，并指导NL2SQL模型的进化。此外，我们提出了开发NL2SQL解决方案的经验法则。最后，我们讨论了LLMs时代下NL2SQL研究中的挑战和开放问题。 

---
# Confidence v.s. Critique: A Decomposition of Self-Correction Capability for LLMs 

**Title (ZH)**: 自信与批评：大型语言模型自我修正能力的分解 

**Authors**: Zhe Yang, Yichang Zhang, Yudong Wang, Ziyao Xu, Junyang Lin, Zhifang Sui  

**Link**: [PDF](https://arxiv.org/pdf/2412.19513)  

**Abstract**: Large Language Models (LLMs) can correct their self-generated responses, but a decline in accuracy after self-correction is also witnessed. To have a deeper understanding of self-correction, we endeavor to decompose, evaluate, and analyze the self-correction behaviors of LLMs. By enumerating and analyzing answer correctness before and after self-correction, we decompose the self-correction capability into confidence (being confident to correct answers) and critique (turning wrong answers to correct) capabilities, and propose two metrics from a probabilistic perspective to measure these 2 capabilities, along with another metric for overall self-correction capability evaluation. Based on our decomposition and evaluation metrics, we conduct extensive experiments and draw some empirical conclusions. For example, we find different models can exhibit distinct behaviors: some models are confident while others are more critical. We also find the trade-off between the two capabilities (i.e. improving one can lead to a decline in the other) when manipulating model self-correction behavior by prompts or in-context learning. Further, we find a simple yet efficient strategy to improve self-correction capability by transforming Supervision Fine-Tuning (SFT) data format, and our strategy outperforms vanilla SFT in both capabilities and achieves much higher accuracy after self-correction. Our code will be publicly available on GitHub. 

**Abstract (ZH)**: 大型语言模型（LLMs）可以纠正它们自动生成的答案，但同时也观察到自我纠正后准确度会下降的现象。为了更深入地理解自我纠正机制，我们尝试分解、评估和分析LLMs的自我纠正行为。通过对比和分析自我纠正前后答案的正确性，我们将自我纠正能力分解为信心（敢于修正答案）和批判力（将错误的答案转变为正确答案）两个能力，并从概率论的角度提出了两个用于衡量这两种能力的度量标准，同时还提出了一种用于总体自我纠正能力评估的度量标准。基于我们的分解和评估指标，我们进行了广泛的实验并得出了若干实证结论。例如，我们发现不同模型可能表现出不同的行为：有些模型更自信，而另一些模型则更严谨。我们还发现，在通过提示或上下文学习操纵模型自我纠正行为时，这两种能力之间的权衡关系（即提高一种能力可能会导致另一种能力下降）。此外，我们发现了一种简单有效的策略，通过更改Supervision Fine-Tuning（SFT）数据格式来提高自我纠正能力，并且该策略在两种能力上都优于传统的SFT，且自我纠正后的准确度也显著提高。我们的代码将在GitHub上公开发布。 

---
# Safeguard Fine-Tuned LLMs Through Pre- and Post-Tuning Model Merging 

**Title (ZH)**: 通过预调和后调模型合并保障微调后的大型语言模型安全 

**Authors**: Hua Farn, Hsuan Su, Shachi H Kumar, Saurav Sahay, Shang-Tse Chen, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2412.19512)  

**Abstract**: Fine-tuning large language models (LLMs) for downstream tasks is a widely adopted approach, but it often leads to safety degradation in safety-aligned LLMs. Currently, many solutions address this issue by incorporating additional safety data, which can be impractical in many cases. In this paper, we address the question: How can we improve downstream task performance while preserving safety in LLMs without relying on additional safety data? We propose a simple and effective method that maintains the inherent safety of LLMs while enhancing their downstream task performance: merging the weights of pre- and post-fine-tuned safety-aligned models. Experimental results across various downstream tasks, models, and merging methods demonstrate that this approach effectively mitigates safety degradation while improving downstream task performance, offering a practical solution for adapting safety-aligned LLMs. 

**Abstract (ZH)**: 将下面的论文内容或标题翻译成中文，符合学术规范：

对大型语言模型（LLM）进行微调以适应下游任务是一种广泛采用的方法，但在安全对齐的LLM中，这往往会导致安全性下降。目前，许多解决方案通过引入额外的安全数据来应对这一问题，但在许多情况下这并不实用。在本文中，我们探讨的问题是：如何在不依赖额外安全数据的情况下提高下游任务性能同时保持LLM的安全性？我们提出了一种简单且有效的方法，该方法保持了LLM的内在安全性并增强了其下游任务性能：合并预制和后微调的安全对齐模型的权重。在各种下游任务、模型和合并方法下的实验结果表明，这种做法有效地减轻了安全性下降的问题并提升了下游任务性能，提供了一种实用的方法来适应安全对齐的LLM。 

---
# Dynamic Skill Adaptation for Large Language Models 

**Title (ZH)**: 大型语言模型的动态技能适应性 

**Authors**: Jiaao Chen, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2412.19361)  

**Abstract**: We present Dynamic Skill Adaptation (DSA), an adaptive and dynamic framework to adapt novel and complex skills to Large Language Models (LLMs). Compared with previous work which learns from human-curated and static data in random orders, we propose to first automatically generate and organize the training data by mimicking the learning pathways of human and then dynamically tailor the training data based on the training dynamics. Specifically, inspired by the learning structures and teaching strategies in the human education system, we first construct a skill graph by decomposing complex skills into sub-skills and arranging them based on their dependencies in human syllables. For every skill, we utilize LLMs to generate both textbook-like data which contains detailed descriptions of skills for pre-training and exercise-like data which targets at explicitly utilizing the skills to solve problems for instruction-tuning. Furthermore, during the instruction-tuning, we dynamically update the training data which down-weight easy-to-learn examples, generate more complex examples, and filter out data with errors. Experiments on large language models such as LLAMA and Mistral demonstrate the effectiveness of our proposed methods in adapting math reasoning skills and social study skills. 

**Abstract (ZH)**: 我们提出了动态技能适应（DSA），这是一种适应性和动态框架，用于将新颖且复杂的技能适应大型语言模型（LLMs）。与之前的工作相比，这些工作依赖于随机顺序的人工策划和静态数据进行学习，我们提出首先通过模仿人类的学习路径自动生成和组织训练数据，然后根据训练动态动态调整训练数据。具体而言，借鉴人类教育系统中的学习结构和教学策略，我们首先通过将复杂技能分解为子技能，并根据人类教科书中的依赖关系进行排列，构建了一个技能图。对于每个技能，我们利用LLMs生成类似教科书的数据，包含技能的详细描述以进行预训练，并生成类似于练习题的数据，以明确利用这些技能解决具体问题，用于指令调优。此外，在指令调优过程中，我们动态更新训练数据，降低容易学习的示例权重，生成更复杂的示例，并过滤掉错误数据。实验表明，在LLAMA和Mistral等大型语言模型上的实验验证了我们提出的方法在适应数学推理技能和社会研究技能方面的有效性。 

---
# "I've Heard of You!": Generate Spoken Named Entity Recognition Data for Unseen Entities 

**Title (ZH)**: 《听过你！》：为未知实体生成语音命名实体识别数据 

**Authors**: Jiawei Yu, Xiang Geng, Yuang Li, Mengxin Ren, Wei Tang, Jiahuan Li, Zhibin Lan, Min Zhang, Hao Yang, Shujian Huang, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2412.19102)  

**Abstract**: Spoken named entity recognition (NER) aims to identify named entities from speech, playing an important role in speech processing. New named entities appear every day, however, annotating their Spoken NER data is costly. In this paper, we demonstrate that existing Spoken NER systems perform poorly when dealing with previously unseen named entities. To tackle this challenge, we propose a method for generating Spoken NER data based on a named entity dictionary (NED) to reduce costs. Specifically, we first use a large language model (LLM) to generate sentences from the sampled named entities and then use a text-to-speech (TTS) system to generate the speech. Furthermore, we introduce a noise metric to filter out noisy data. To evaluate our approach, we release a novel Spoken NER benchmark along with a corresponding NED containing 8,853 entities. Experiment results show that our method achieves state-of-the-art (SOTA) performance in the in-domain, zero-shot domain adaptation, and fully zero-shot settings. Our data will be available at this https URL. 

**Abstract (ZH)**: 口语命名实体识别（Spoken Named Entity Recognition, Spoken NER）旨在从口语中识别命名实体，这在口语处理中扮演着重要角色。新的命名实体每天都在出现，然而标注其口语命名实体识别数据的成本较高。本文表明，现有的口语命名实体识别系统在处理之前未见过的命名实体时表现不佳。为应对这一挑战，我们提出了一种基于命名实体字典（Named Entity Dictionary, NED）生成口语命名实体识别数据的方法以降低标注成本。具体而言，我们首先使用大规模语言模型（Large Language Model, LLM）从采样的命名实体中生成句子，然后使用文本转语音（Text-to-Speech, TTS）系统生成口语。此外，我们引入了一个噪音度量方法以过滤掉嘈杂的数据。为了评估我们的方法，我们提供了一个新的口语命名实体识别基准数据集及其相应的包含8,853个实体的命名实体字典。实验结果表明，我们的方法在领域内、零样本领域自适应和完全零样本设置中均达到了目前最先进的性能（State-of-the-Art, SOTA）。我们的数据集将在以下网址提供：\[此链接\]。 

---
# Advancing LLM detection in the ALTA 2024 Shared Task: Techniques and Analysis 

**Title (ZH)**: ALTA 2024 共享任务中大语言模型检测的进展：技术与分析 

**Authors**: Dima Galat  

**Link**: [PDF](https://arxiv.org/pdf/2412.19076)  

**Abstract**: The recent proliferation of AI-generated content has prompted significant interest in developing reliable detection methods. This study explores techniques for identifying AI-generated text through sentence-level evaluation within hybrid articles. Our findings indicate that ChatGPT-3.5 Turbo exhibits distinct, repetitive probability patterns that enable consistent in-domain detection. Empirical tests show that minor textual modifications, such as rewording, have minimal impact on detection accuracy. These results provide valuable insights for advancing AI detection methodologies, offering a pathway toward robust solutions to address the complexities of synthetic text identification. 

**Abstract (ZH)**: 近年来，AI生成内容的激增引发了对可靠检测方法的广泛关注。本研究探讨了通过混合文章内的句子级评估来识别AI生成文本的技术。研究发现，ChatGPT-3.5 Turbo 显示出独特的、重复的概率模式，这使其可以在领域内实现一致的检测。实证测试表明，轻微的文本修改，如重新润色，对检测准确性的影响很小。这些结果为推进AI检测方法学提供了宝贵的见解，为解决合成文本识别的复杂性提供了稳健的解决方案路径。 

---
# RapGuard: Safeguarding Multimodal Large Language Models via Rationale-aware Defensive Prompting 

**Title (ZH)**: RapGuard：通过理据意识防御型提示保护多模态大规模语言模型 

**Authors**: Yilei Jiang, Yingshui Tan, Xiangyu Yue  

**Link**: [PDF](https://arxiv.org/pdf/2412.18826)  

**Abstract**: While Multimodal Large Language Models (MLLMs) have made remarkable progress in vision-language reasoning, they are also more susceptible to producing harmful content compared to models that focus solely on text. Existing defensive prompting techniques rely on a static, unified safety guideline that fails to account for the specific risks inherent in different multimodal contexts. To address these limitations, we propose RapGuard, a novel framework that uses multimodal chain-of-thought reasoning to dynamically generate scenario-specific safety prompts. RapGuard enhances safety by adapting its prompts to the unique risks of each input, effectively mitigating harmful outputs while maintaining high performance on benign tasks. Our experimental results across multiple MLLM benchmarks demonstrate that RapGuard achieves state-of-the-art safety performance, significantly reducing harmful content without degrading the quality of responses. 

**Abstract (ZH)**: 尽管多模态大型语言模型（MLLMs）在视觉语言推理方面取得了显著进展，但与仅专注于文本的模型相比，它们更容易生成有害内容。现有的防御性提示技术依靠的是静态的、统一的安全准则，无法考虑到不同多模态背景下固有的特定风险。为了解决这些限制，我们提出了一种名为RapGuard的新型框架，该框架利用多模态链式推理动态生成针对特定场景的安全提示。RapGuard通过适应每个输入的独特风险来提升安全性，有效地减轻有害输出，同时在无害任务上保持高水平的性能。我们在多个MLLM基准上的实验结果表明，RapGuard实现了最先进的安全性能，显著减少了有害内容，而不会降低响应的质量。 

---
# DCIS: Efficient Length Extrapolation of LLMs via Divide-and-Conquer Scaling Factor Search 

**Title (ZH)**: DCIS: 分而治之缩放因子搜索下的高效长度外推大规模语言模型 

**Authors**: Lei Yang, Shaoyang Xu, Deyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2412.18811)  

**Abstract**: Large language models (LLMs) based on the Transformer architecture usually have their context length limited due to the high training cost. Recent advancements extend the context window by adjusting the scaling factors of RoPE and fine-tuning. However, suboptimal initialization of these factors results in increased fine-tuning costs and reduced performance at target length. To address these challenges, we propose an innovative RoPE-based fine-tuning framework that diverges from conventional scaling factors search. Specifically, we present a Divide-and-Conquer Incremental Search (DCIS) algorithm that strategically determines the better scaling factors. Further fine-tuning with the identified scaling factors effectively extends the context window of LLMs. Empirical results demonstrate that our methodology not only mitigates performance decay at extended target lengths but also allows the model to fine-tune on short contexts and generalize to long contexts, thereby reducing the cost of fine-tuning. The scaling factors obtained through DCIS can even perform effectively without fine-tuning. Further analysis of the search space reveals that DCIS achieves twice the search efficiency compared to other methods. We also examine the impact of the non-strictly increasing scaling factors utilized in DCIS and evaluate the general capabilities of LLMs across various context lengths. 

**Abstract (ZH)**: 基于Transformer架构的大语言模型（LLMs）通常受限于上下文长度，因为训练成本较高。最近的进步通过调整RoPE（旋转位置编码）的缩放因子和微调来扩展上下文窗口。然而，这些因子的次优初始化会导致微调成本增加并降低目标长度上的性能。为应对这些挑战，我们提出了一种创新的基于RoPE的微调框架，该框架不同于传统的缩放因子搜索方式。具体而言，我们提出了一种分而治之增量搜索（DCIS）算法，该算法战略性地确定了更佳的缩放因子。使用识别出的缩放因子进行进一步微调可以有效扩展LLMs的上下文窗口。实验证明，我们的方法不仅能缓解在延长目标长度上的性能衰减，还能使模型在短上下文中进行微调并适应长上下文，从而降低微调成本。通过DCIS获得的缩放因子甚至可以在不微调的情况下有效工作。进一步对搜索空间的分析表明，DCIS的搜索效率是其他方法的两倍。我们还研究了DCIS中使用的非严格递增缩放因子的影响，并评估了LLMs在各种上下文长度下的泛化能力。 

---
# Using Large Language Models for Automated Grading of Student Writing about Science 

**Title (ZH)**: 使用大型语言模型对学生科学写作的自动化评分应用 

**Authors**: Chris Impey, Matthew Wenger, Nikhil Garuda, Shahriar Golchin, Sarah Stamer  

**Link**: [PDF](https://arxiv.org/pdf/2412.18719)  

**Abstract**: Assessing writing in large classes for formal or informal learners presents a significant challenge. Consequently, most large classes, particularly in science, rely on objective assessment tools such as multiple-choice quizzes, which have a single correct answer. The rapid development of AI has introduced the possibility of using large language models (LLMs) to evaluate student writing. An experiment was conducted using GPT-4 to determine if machine learning methods based on LLMs can match or exceed the reliability of instructor grading in evaluating short writing assignments on topics in astronomy. The audience consisted of adult learners in three massive open online courses (MOOCs) offered through Coursera. One course was on astronomy, the second was on astrobiology, and the third was on the history and philosophy of astronomy. The results should also be applicable to non-science majors in university settings, where the content and modes of evaluation are similar. The data comprised answers from 120 students to 12 questions across the three courses. GPT-4 was provided with total grades, model answers, and rubrics from an instructor for all three courses. In addition to evaluating how reliably the LLM reproduced instructor grades, the LLM was also tasked with generating its own rubrics. Overall, the LLM was more reliable than peer grading, both in aggregate and by individual student, and approximately matched instructor grades for all three online courses. The implication is that LLMs may soon be used for automated, reliable, and scalable grading of student science writing. 

**Abstract (ZH)**: 大规模班级中对正式或非正式学习者的写作进行评估是一个重大挑战。因此，大多数大规模班级，特别是在科学领域，依赖于客观评估工具，如多项选择测验，这些测验通常只有一个正确答案。随着人工智能的迅速发展，利用大型语言模型（LLMs）来评估学生写作的可能性已经出现。一项实验使用了GPT-4，旨在确定基于LLMs的机器学习方法能否在评估天文学相关主题的短篇写作作业方面与教师评分相媲美或超越教师评分的可靠性。实验对象包括通过Coursera提供的三门大规模开放在线课程（MOOCs）中的成年学习者。其中一门课程为天文学，第二门课程为天体生物学，第三门课程为天文学的历史与哲学。实验结果也可以适用于大学环境中非科学专业的学生，其中课程内容和评估方式类似。数据包括三个课程中120名学生对12个问题的回答。GPT-4被提供了所有三个课程中的总成绩、模型答案和教师评分标准。除了评估LLM在重现教师评分方面的可靠度外，LLM还被要求生成自己的评分标准。总体而言，LLM在汇总和个体学生层面都比同伴评分更加可靠，并且在所有三个在线课程中大致与教师评分相当。这一结果意味着未来LLMs可能被用于自动、可靠且可扩展地评估学生的科学写作。 

---
# AgreeMate: Teaching LLMs to Haggle 

**Title (ZH)**: AgreeMate：教学超大规模语言模型进行讨价还价 

**Authors**: Ainesh Chatterjee, Samuel Miller, Nithin Parepally  

**Link**: [PDF](https://arxiv.org/pdf/2412.18690)  

**Abstract**: We introduce AgreeMate, a framework for training Large Language Models (LLMs) to perform strategic price negotiations through natural language. We apply recent advances to a negotiation setting where two agents (i.e. buyer or seller) use natural language to bargain on goods using coarse actions. Specifically, we present the performance of Large Language Models when used as agents within a decoupled (modular) bargaining architecture. We demonstrate that using prompt engineering, fine-tuning, and chain-of-thought prompting enhances model performance, as defined by novel metrics. We use attention probing to show model attention to semantic relationships between tokens during negotiations. 

**Abstract (ZH)**: 我们将介绍AgreeMate框架，该框架旨在通过自然语言训练大规模语言模型（LLMs）进行战略性价格谈判。我们应用了最近的进展，将这种谈判设置应用于两个代理（买家或卖家）使用自然语言通过粗略行动来进行商品交易的情境。具体而言，我们展示了在解耦（模块化）的协商架构中，当大规模语言模型作为代理使用时的性能表现。我们证明了通过提示工程、微调和链式思考提示可以提升模型性能，而这种提升利用了新的性能指标进行定义。我们采用注意力探针来展示模型在谈判过程中对词汇间语义关系的注意力。 

---
# KRAIL: A Knowledge-Driven Framework for Base Human Reliability Analysis Integrating IDHEAS and Large Language Models 

**Title (ZH)**: KRAIL：一种基于知识的框架，结合IDHEAS和大规模语言模型进行基础人类可靠性分析 

**Authors**: Xingyu Xiao, Peng Chen, Ben Qi, Hongru Zhao, Jingang Liang, Jiejuan Tong, Haitao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.18627)  

**Abstract**: Human reliability analysis (HRA) is crucial for evaluating and improving the safety of complex systems. Recent efforts have focused on estimating human error probability (HEP), but existing methods often rely heavily on expert knowledge,which can be subjective and time-consuming. Inspired by the success of large language models (LLMs) in natural language processing, this paper introduces a novel two-stage framework for knowledge-driven reliability analysis, integrating IDHEAS and LLMs (KRAIL). This innovative framework enables the semi-automated computation of base HEP values. Additionally, knowledge graphs are utilized as a form of retrieval-augmented generation (RAG) for enhancing the framework' s capability to retrieve and process relevant data efficiently. Experiments are systematically conducted and evaluated on authoritative datasets of human reliability. The experimental results of the proposed methodology demonstrate its superior performance on base HEP estimation under partial information for reliability assessment. 

**Abstract (ZH)**: 人类可靠性分析（HRA）对于评估和改善复杂系统的安全性至关重要。近期的研究主要集中在估计人类错误概率（HEP），但现有方法往往依赖于专家知识，这可能会导致主观性和耗时性。受到大型语言模型（LLMs）在自然语言处理领域取得成功的影响，本文提出了一种新的基于知识驱动的可靠性分析框架——KRAIL（Knowledge-driven Reliability Analysis using Integrated IDHEAS and Large Language Models），并采用两阶段流程实现这一框架。这个创新性框架能够半自动化地计算基础HEP值。此外，本文还利用知识图谱作为一种检索增强生成（RAG）方法，以提高框架在高效检索和处理相关数据方面的能力。我们系统地在权威的人类可靠性数据集上进行了实验并进行了评估。所提出方法的实验结果表明，在可靠性评估中基于部分信息对基础HEP进行估计时，该方法表现出卓越的性能。 

---
# Why Do Large Language Models (LLMs) Struggle to Count Letters? 

**Title (ZH)**: 为什么大型语言模型（LLMs）在数字母时显得力不从心？ 

**Authors**: Tairan Fu, Raquel Ferrando, Javier Conde, Carlos Arriaga, Pedro Reviriego  

**Link**: [PDF](https://arxiv.org/pdf/2412.18626)  

**Abstract**: Large Language Models (LLMs) have achieved unprecedented performance on many complex tasks, being able, for example, to answer questions on almost any topic. However, they struggle with other simple tasks, such as counting the occurrences of letters in a word, as illustrated by the inability of many LLMs to count the number of "r" letters in "strawberry". Several works have studied this problem and linked it to the tokenization used by LLMs, to the intrinsic limitations of the attention mechanism, or to the lack of character-level training data. In this paper, we conduct an experimental study to evaluate the relations between the LLM errors when counting letters with 1) the frequency of the word and its components in the training dataset and 2) the complexity of the counting operation. We present a comprehensive analysis of the errors of LLMs when counting letter occurrences by evaluating a representative group of models over a large number of words. The results show a number of consistent trends in the models evaluated: 1) models are capable of recognizing the letters but not counting them; 2) the frequency of the word and tokens in the word does not have a significant impact on the LLM errors; 3) there is a positive correlation of letter frequency with errors, more frequent letters tend to have more counting errors, 4) the errors show a strong correlation with the number of letters or tokens in a word and 5) the strongest correlation occurs with the number of letters with counts larger than one, with most models being unable to correctly count words in which letters appear more than twice. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在许多复杂任务上取得了前所未有的性能，能够回答几乎任何主题的问题。然而，它们在一些简单的任务上表现不佳，例如计算单词中某个字母的出现次数。例如，许多LLMs无法准确计算“strawberry”中字母“r”的出现次数。已有研究将这一问题与LLMs所使用的分词、注意力机制的内在限制或字符级别训练数据的缺乏联系起来。在这项研究中，我们进行了实验性研究，评估了LLM在计算字母出现次数时的错误与1）训练数据集中单词及其组成部分的频率和2）计数操作的复杂性之间的关系。我们通过评估一组代表性模型在大量单词上的计字母出现次数，对LLM的错误进行了全面分析。研究结果表明，在评估的模型中存在几个一致的趋势：1）模型能够识别字母，但无法进行计数；2）单词及其组成部分在训练数据集中的频率对LLMs的错误影响不大；3）字母频率与错误之间存在正相关关系，更频繁出现的字母往往有更多的计数错误；4）错误与单词或其组成部分中的字母数或令牌数之间存在强烈的相关性；5）与字母数目的相关性最强的是那些计数超过一个的字母，大多数模型在字母出现次数超过两次的单词计数方面无法正确完成任务。 

---
# InfAlign: Inference-aware language model alignment 

**Title (ZH)**: InfAlign：基于推断的语言模型对齐 

**Authors**: Ananth Balashankar, Ziteng Sun, Jonathan Berant, Jacob Eisenstein, Michael Collins, Adrian Hutter, Jong Lee, Chirag Nagpal, Flavien Prost, Aradhana Sinha, and Ananda Theertha Suresh, Ahmad Beirami  

**Link**: [PDF](https://arxiv.org/pdf/2412.19792)  

**Abstract**: Language model alignment has become a critical step in training modern generative language models. The goal of alignment is to finetune a reference model such that the win rate of a sample from the aligned model over a sample from the reference model is high, subject to a KL divergence constraint. Today, we are increasingly using inference-time algorithms (e.g., Best-of-N, controlled decoding, tree search) to decode from language models rather than standard sampling. However, the alignment objective does not capture such inference-time decoding procedures. We show that the existing alignment framework is sub-optimal in view of such inference-time methods. We then modify the alignment objective and propose a framework for inference-aware alignment (IAPO). We prove that for any inference-time decoding algorithm, the optimal solution that optimizes the inference-time win rate of the aligned policy against the reference policy is the solution to the typical RLHF problem with a transformation of the reward. This motivates us to provide the KL-regularized calibrate-and-transform RL (CTRL) algorithm to solve this problem, which involves a reward calibration step and a KL-regularized reward maximization step with a transformation of the calibrated reward. We particularize our study to two important inference-time strategies: best-of-N sampling and best-of-N jailbreaking, where N responses are sampled from the model and the one with the highest or lowest reward is selected. We propose specific transformations for these strategies and demonstrate that our framework offers significant improvements over existing state-of-the-art methods for language model alignment. Empirically, we outperform baselines that are designed without taking inference-time decoding into consideration by 8-12% and 4-9% on inference-time win rates over the Anthropic helpfulness and harmlessness dialog benchmark datasets. 

**Abstract (ZH)**: 语言模型对齐已成为培训现代生成语言模型的一个关键步骤。对齐的目标是微调一个参考模型，使得对齐模型的一个样本相对于参考模型的一个样本的胜率最大化，同时要满足KL散度约束。如今，我们越来越多地在推理时间使用算法（如Best-of-N、受控解码、树搜索）进行语言模型解码，而不仅仅是使用标准采样方法。然而，现有的对齐目标并不能捕捉到这样的推理时解码过程。我们显示，在这样的推理时方法方面，现有的对齐框架是次优的。我们随后修改了对齐目标，并提出了一个考虑推理时的对齐框架（IAPO，Inference-Aware Alignment）。我们证明，对于任何推理时解码算法，优化对齐策略相对于参考策略的推理时胜率的最佳解是转换后的奖励问题的标准RLHF问题的解。这促使我们提出了KL正则化的校准和转换强化学习（CTRL，KL-regularized Calibrate-and-Transform RL）算法来解决这一问题，该算法包括一个奖励校准步骤和一个转换后的校准奖励的KL正则化奖励最大化步骤。我们将研究具体应用到了两个重要的推理时策略：Best-of-N采样和Best-of-N破解，其中模型会采样N个响应，然后选择奖励最高或最低的响应。我们为这些策略提出了具体的转换方法，并证明我们的框架在语言模型对齐方面显著优于现有最先进的方法。在实验中，我们的框架在Anthropic友好性和无害性对话基准数据集上的推理时胜率上相较于不考虑推理时解码的基线方法，提高了8%-12%和4%-9%。 

---
