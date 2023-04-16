# Search for Concepts: Discovering Visual Concepts Using Direct Optimization

Finding an unsupervised decomposition of an image into individual objects is a key step to leverage compositionality and to perform symbolic reasoning. Traditionally, this problem is solved using amortized inference, which does not generalize beyond the scope of the training data, may sometimes miss correct decompositions, and requires large amounts of training data. We propose finding a decomposition using direct, unamortized optimization, via a combination of a gradient-based optimization for differentiable object properties and global search for non-differentiable properties. We show that using direct optimization is more generalizable, misses fewer correct decompositions, and typically requires less data than methods based on amortized inference. This highlights a weakness of the current prevalent practice of using amortized inference that can potentially be improved by integrating more direct optimization elements.

Website: https://geometry.cs.ucl.ac.uk/projects/2022/SearchForConcepts/

<img src="https://geometry.cs.ucl.ac.uk/projects/2022/SearchForConcepts/paper_docs/teaser.png">

## Citation
```
  @misc{https://doi.org/10.48550/arxiv.2210.14808,
    doi = {10.48550/ARXIV.2210.14808},
    url = {https://arxiv.org/abs/2210.14808},
    author = {Reddy, Pradyumna and Guerrero, Paul and Mitra, Niloy J.},
    title = {Search for Concepts: Discovering Visual Concepts Using Direct Optimization},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
  }
```
