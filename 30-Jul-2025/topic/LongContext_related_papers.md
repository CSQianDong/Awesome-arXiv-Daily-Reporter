# TriangleMix: A Lossless and Efficient Attention Pattern for Long Context Prefilling 

**Authors**: Zhiyuan He, Yike Zhang, Chengruidong Zhang, Huiqiang Jiang, Yuqing Yang, Lili Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2507.21526)  

**Abstract**: Large Language Models (LLMs) rely on attention mechanisms whose time complexity grows quadratically with input sequence length, creating significant computational bottlenecks during the prefilling stage. Existing static sparse attention methods typically degrade accuracy, while dynamic sparsity methods introduce additional computational overhead due to runtime sparse index estimation. To address these limitations, we propose TriangleMix, a novel training-free static attention pattern. TriangleMix employs dense attention in shallow layers and switches to a triangle-shaped sparse pattern in deeper layers. Extensive experiments demonstrate that TriangleMix reduces attention overhead by 3.7x to 15.3x in deep layers, and decreases overall Time-to-First-Token (TTFT) by 12% to 32% for sequence lengths ranging from 32K to 128K, without sacrificing model accuracy. Moreover, TriangleMix can be seamlessly integrated with dynamic sparsity methods to achieve further speedup, e.g. accelerating MInference by 19% at 128K, highlighting its potential to enhance LLM inference efficiency. 

---
