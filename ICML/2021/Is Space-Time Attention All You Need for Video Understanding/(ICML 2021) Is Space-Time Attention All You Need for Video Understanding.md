# (ICML 2021) Is Space-Time Attention All You Need for Video Understanding?

## 0. Summary

## 1. Research Objective

​		Replace convolution operation with self-attention over space and time on video classification.

## 2. Background and Problems

+ The field of natural language processing has been revolutionized by the emergence of methods based on self-attention. Video understanding shares several high-level similarities with NLP:

  1. videos and sentences are both sequential.
  2. Atomic actions in short-term segments need to be contextualized with the rest of the video in order to be fully disambiguated.

+ 2D or 3D convolutions still represent the core operators for spatiotemporal feature learning across disfferent video tasks. A convolution-free video architecture has the potential to over a few inherent limitations of convolutional models for video analysis:

  1. **Strong inductive biases of convolution** are undoubedly beneficial on small training sets, which **may excessively limit the expressivity of the model** in settings where there is ample availability of data and “all” can be learned from examples. 

     **Compared to CNNs, Transformers impose less restrictive inductive biases**, which broadens the family of functions they can represent.
     
     > 卷积能够获得局部区域内的结构信息(不同元素的位置信息)，而自注意力机制会丢失这种结构信息，Transformer中使用位置编码来弥补丢失的位置信息。
     
  2. Convolution cannot model dependencies that extend beyond the receptive field. While deep stakcs of convolutions are inherently limited in capturing long-range dependencies.
  
     **Self-attention mechanism can capture both local as well as global long-range denpendencies** by directly comparing feature activations at all space-time locations.
  
  3. Training deep CNNs remains very costly, especially when applied to high-resolution and long videos. 
  
     Transormers enjoy faster traininig and inference compared to CNNs.

+ Some works use self-attention for image classification. **These works use individual pixels as queries, so it is costly to computing a similarity measure for all pairs of pixels**.

  + In order to maintain a manageable computational cost and a small memory consumption, they must:
    + restrict the scope of self-attention to local neighborhoods.
    + use global self-attention on heavily downsized versions of the image.
  + Alternative strategies for scalability to full im- ages include:
    + sparse key-value sampling
    + constraining the self-attention to be calculated along the spatial axes

  The Vision Transformer introduces a strategy decomposing the image into a sequence of patches.

+ *Problem is defined as* : Design a video architecture built exclusively on self-attention based on ViT, and  take it away from **costly computing a similarity measure for all pairs of tokens.**

## 3. Method

+ Based on ViT:

  + Decompose input clip into patches.

  + Linearly map each patch into an embedding vector, add **positional embedding to encode the spatiotemporal position** of each patch.

    > 如何初始化位置编码，并且进行学习？

  + Compute Query-Key-Value and use them compute self-attention weights.

+ Use classification embedding, The final clip embedding is obtained from the final block for the classification token.

  > 加入的分类嵌入需要阅读BERT的论文

+ Used divided attention to compute spatio and temporal attention separately.

  > 作者通过实验发现先计算时间注意力的性能最好，但是没有解释原因。由于本文仅对同一位置的patch计算时间注意力，若未先计算空间注意力，每个patch就只包含当前位置的信息，此时时间注意力计算中缺少了不同位置patch在空间维度上的依赖关系。先计算空间注意力就可以保留空间上的依赖。

## 4. Evalution

+ Divided space-time attention is more efficient than joint space-time attention when operating on higher spatial resolution, or longer videos.

+ Compared to 3D CNN, although TimeSformer has a large learning capacity, it has low inference cost. **This suggests that TimeSformer is better suited for settings that involve large-scale learning.**

+ 增大patch的大小会造成性能的下降，作者认为是由于更大的patch使得**空间粒度的减少(due to the reduced spatial granularity)**。

  > 应该指的是更粗的粒度？将patch映射成嵌入向量的过程损失了patch内的细节信息（相比将像素作为queries），更大的patch导致丢失的细节信息更多，降低网络的性能。因此将像素作为queries不会损失细节信息，不考虑开销的情况下网络性能可以达到最优？

+ ![fig_1](img/fig_1.png)

  > 增大分辨率能够增加空间上的细节特征，使网络的性能得到提升，但是随着分辨率的提升，网络的性能开始下降（如图中560Px），为什么？

## 5. Conclusion

+ TimeSformer is conceptually simple
+ Achieves state-of-art results on major action arecognition benchmarks
+ has low training and inference cost
+ cane be applied to clips of over one minute, thus enabling long-term video modeling.

## Notes

## References

