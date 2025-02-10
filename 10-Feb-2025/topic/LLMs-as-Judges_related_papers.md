# Aligning Black-box Language Models with Human Judgments 

**Title (ZH)**: 将黑盒语言模型与人类判断对齐 

**Authors**: Gerrit J. J. van den Burg, Gen Suzuki, Wei Liu, Murat Sensoy  

**Link**: [PDF](https://arxiv.org/pdf/2502.04997)  

**Abstract**: Large language models (LLMs) are increasingly used as automated judges to evaluate recommendation systems, search engines, and other subjective tasks, where relying on human evaluators can be costly, time-consuming, and unscalable. LLMs offer an efficient solution for continuous, automated evaluation. However, since the systems that are built and improved with these judgments are ultimately designed for human use, it is crucial that LLM judgments align closely with human evaluators to ensure such systems remain human-centered. On the other hand, aligning LLM judgments with human evaluators is challenging due to individual variability and biases in human judgments. We propose a simple yet effective framework to align LLM judgments with individual human evaluators or their aggregated judgments, without retraining or fine-tuning the LLM. Our approach learns a linear mapping between the LLM's outputs and human judgments, achieving over 142% average improvement in agreement across 29 tasks with only a small number of calibration examples used for training. Notably, our method works in zero-shot and few-shot settings, exceeds inter-human agreement on four out of six tasks, and enables smaller LLMs to achieve performance comparable to that of larger models. 

**Abstract (ZH)**: 大规模语言模型（LLMs）越来越多地被用作自动化法官来评估推荐系统、搜索引擎和其他主观任务，此时依赖人工评估者可能会导致成本高昂、耗时且难以扩展。LLMs 提供了一种有效的解决方案，可以实现连续的自动化评估。然而，由于这些通过这些评估构建和改进的系统最终旨在为人使用，因此确保LLM的评估结果与人工评价者高度一致对于确保系统保持以人为本至关重要。另一方面，由于人工判断中的个体差异和偏见，使LLM评估结果与人工评价者相一致具有挑战性。我们提出了一种简单而有效的框架，用于将LLM的评估结果与特定的人工评价者或他们的汇总评估结果对齐，而无需对LLM进行重新训练或微调。我们的方法通过学习LLM输出与人工评估结果之间的线性映射关系，仅使用少量校准示例进行训练，在29个任务上的平均一致性改善超过142%。值得注意的是，我们的方法在零样本和少样本设置下有效，在六个任务中四个任务上超过了人与人之间的一致性和使较小的LLM能够达到与较大模型相当的性能。 

---
