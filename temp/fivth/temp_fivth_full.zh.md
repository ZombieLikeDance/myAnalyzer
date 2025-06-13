# 基于潜空间扩散模型的高分辨率图像合成  
High-Resolution Image Synthesis with Latent Diffusion Models

Robin Rombach^1\*, Andreas Blattmann^1\*, Dominik Lorenz^1, Patrick Esser, Björn Ommer^1  
1慕尼黑路德维希-马克西米利安大学（Ludwig Maximilian University of Munich） & 海德堡大学IWR（IWR, Heidelberg University), Germany  
Runway ML  
https://github.com/CompVis[...]

---

# 摘要（Abstract）

通过将图像生成过程分解为一系列去噪自编码器（denoising autoencoder）的顺序应用，扩散模型（Diffusion Models, DMs）在图像数据及其他领域实现了最先进（state-of-the-art）的合成效果。然而，这些模型对高分辨率数据的直接扩展带来了巨大的计算负担。本文提出了一种在潜空间（latent space）中操作的扩散模型（Latent Diffusion Models, LDMs），该方法通过首先训练一个感知损失（perceptual loss）驱动的自动编码器（autoencoder），将输入映射到低维潜空间，再在该潜空间中训练扩散模型，从而大幅降低了计算复杂度。最终，生成图像通过解码器（decoder）还原到高分辨率像素空间。该方法不仅显著降低了训练和采样（sampling）的计算成本，而且在多个任务（如无条件图像生成、修复（inpainting）、超分辨率（super-resolution）等）和数据集上取得了与现有最优方法相媲美甚至更优的性能。我们还提出了一种基于跨注意力（cross-attention）的通用条件机制（conditioning mechanism），支持多模态（multi-modal）训练。我们公开了预训练的潜空间扩散模型及自动编码器，供相关领域使用。

---

# 1. 引言（Introduction）

图像合成（Image Synthesis）是计算机视觉领域发展最为迅速的方向之一，但同时也是计算资源需求最高的领域之一。特别是高分辨率图像的合成，其对硬件资源的要求极高。扩散模型（Diffusion Models, DMs）作为一种基于似然（likelihood-based）的生成模型（generative model），因其能够良好覆盖数据分布的模式（mode-covering behavior），在图像生成任务中展现了卓越的性能。但这种模式使其消耗了大量模型容量（capacity）及计算资源（compute），尤其是在高分辨率场景下。

为提升该模型类别的可及性（accessibility），同时降低其高昂的资源消耗，急需一种既能保证生成质量，又能降低训练和推理复杂度的方法。

**迁移至潜空间（Departure to Latent Space）**  
我们的方案首先分析了像素空间（pixel space）下已训练扩散模型的率失真权衡（rate-distortion trade-off）。借助常规做法[11, 23, 66, 67, 96]，我们将训练流程分为两个阶段：首先训练一个自动编码器，为图像提供低维高效的表示（representation）；随后在该潜空间中训练扩散模型。该流程的突出优势在于，自动编码器阶段仅需一次训练，便可复用到多种扩散模型训练或探索不同数据领域。

综上，我们的工作贡献如下：

1. 相较于纯Transformer[23, 66]，我们的方法能更优雅地扩展到高维数据，不仅支持更低失真率的压缩，还能在高分辨率下保持较高保真度。
2. 在多个任务（无条件图像合成、修复、随机超分辨率）和数据集上均取得了可竞争的性能，同时大幅降低了计算消耗。
3. 与以往需联合训练编码器/解码器与分数先验（score-based prior）的工作[93]不同，我们的方法无需复杂的重建与生成损失权重调整。
4. 在密集条件任务如超分辨率、修复、语义合成（semantic synthesis）中，模型可卷积式（convolutional fashion）应用，支持渲染大尺寸一致性图像。
5. 设计了一种基于跨注意力（cross-attention）的通用条件机制，支持多模态训练（如类别、文本到图像、布局到图像等）。
6. 公开了预训练潜扩散模型及自动编码器，便于各类扩散模型及相关任务复用。

---

# 2. 相关工作（Related Work）

**图像生成的生成模型（Generative Models for Image Synthesis）**  
高维图像数据给生成建模带来巨大挑战。生成对抗网络（Generative Adversarial Networks, GANs）[27]因采样高效而广泛应用，但存在训练不稳定、模式崩溃（mode collapse）等问题。近年，扩散概率模型（Diffusion Probabilistic Models, DMs）[82]在密度估计[45]和样本质量[15]方面达到最先进水平，其生成能力源自逐步去噪高斯变量的机制。

**两阶段图像合成（Two-Stage Image Synthesis）**  
为弥补单一生成方法的不足，越来越多工作[11, 23, 67, 70, 101, 103]尝试将多种方法结合。部分方法[93, 80]联合或分离训练编码/解码网络与分数先验，但联合训练需精细权重调整，难度较高。

---

# 3. 方法（Method）

为降低高分辨率扩散模型训练的计算需求，我们注意到扩散模型能够通过忽略感知不相关细节来降低任务复杂度。我们提出明确分离压缩与生成学习阶段，通过自动编码器（autoencoder）先将图像压缩到低维潜空间（latent space），再在潜空间中训练扩散模型（latent diffusion model, LDM）。该方法带来的优势有：

1. 离开高维像素空间后，采样与训练均更高效。
2. 潜空间建模更易于捕捉全局结构及长距离依赖。
3. 可灵活复用压缩模型，支持多种生成任务。

---

## 3.1 感知图像压缩（Perceptual Image Compression）

我们的感知压缩模型基于[23]，采用自动编码器（autoencoder）联合感知损失（perceptual loss）[106]和基于块的对抗损失（patch-based adversarial objective）[20, 23, 106]训练。给定RGB空间图像 $\boldsymbol{x} \in \mathbb{R}^{H \times W \times 3}$，编码器 $\mathcal{E}$ 将其映射到潜表示 $z = \mathcal{E}(x)$，解码器 $\mathcal{D}$ 负责重建。为避免潜空间方差过大，我们引入KL正则化（KL-regularization）等策略。

---

## 3.2 潜空间扩散模型（Latent Diffusion Models, LDMs）

扩散模型（Diffusion Models, DMs）[82]通过逐步去噪正态分布变量学习数据分布 $p(x)$，用如下损失优化：

$$
L_{DM} = \mathbb{E}_{x, \epsilon \sim \mathcal{N}(0,1), t} [\|\epsilon - \epsilon_\theta(x_t, t)\|_2^2],
$$

$t$ 在 $\{1, ..., T\}$ 均匀采样。  
我们在训练好的感知压缩模型 $\mathcal{E}, \mathcal{D}$ 构建的低维潜空间上进行扩散建模，主干采用时间条件UNet（time-conditional UNet）[71]。

---

## 3.3 条件机制（Conditioning Mechanisms）

类似其它生成模型[56, 83]，扩散模型可建模条件分布 $p(z|y)$。我们通过在UNet主干中引入跨注意力（cross-attention）机制[97]，实现对类别、文本、布局等多种条件的支持。具体实现为：  
$$
Q = W_Q^{(i)} \cdot \varphi_i(z_t),\quad K = W_K^{(i)} \cdot \tau_\theta(y),\quad V = W_V^{(i)} \cdot \tau_\theta(y)
$$  
其中 $\varphi_i(z_t)$ 是UNet的中间表示，$\tau_\theta(y)$ 为条件嵌入。

---

# 4. 实验（Experiments）

LDMs 支持多种图像模态的灵活高效扩散基生成。我们分别在感知压缩权衡、图像生成、条件生成、超分辨率、修复等任务上进行了系统实验。

---

## 4.1 感知压缩权衡分析（On Perceptual Compression Tradeoffs）

分析不同下采样因子$f$对样本质量、采样速度和FID分数的影响，LDM-4等在保持较低失真的同时大幅提升采样效率。

---

## 4.2 潜空间图像生成（Image Generation with Latent Diffusion）

在CelebA-HQ、FFHQ、LSUN-Churches、LSUN-Bedrooms、ImageNet等数据集上训练无条件模型，评估其样本质量（FID、Precision、Recall）与数据覆盖度，结果优于前人多数扩散模型，且资源消耗更低。

---

## 4.3 条件潜空间扩散（Conditional Latent Diffusion）

### 4.3.1 LDM中的Transformer编码器（Transformer Encoders for LDMs）

通过跨注意力条件机制，LDM支持文本到图像（text-to-image）、布局到图像（layout-to-image）等丰富任务。训练于LAION[78]和COCO[4]等数据集，性能在最新扩散方法中处于领先。

### 4.3.2 卷积采样超越256²（Convolutional Sampling Beyond 256²）

对空间对齐的条件输入进行拼接，使LDM能作为高效的通用图像到图像翻译模型（image-to-image translation model），可直接生成大分辨率一致性图像。

---

## 4.4 潜空间超分辨率（Super-Resolution with Latent Diffusion）

通过在潜空间中对低分辨率图像进行条件建模，LDM在超分辨率任务中表现出色，不仅可还原真实纹理，也能泛化到多样退化类型。

---

## 4.5 潜空间修复（Inpainting with Latent Diffusion）

在修复任务中，LDM通过条件扩散和跨注意力机制，生成的图像质量（FID、LPIPS等）均优于现有对比方法，且可高效采样多个候选结果。

---

# 5. 局限性与社会影响（Limitations & Societal Impact）

**局限性（Limitations）**  
虽然LDM大幅降低了高分辨率生成的计算需求，但其采样过程仍慢于GAN。此外，潜空间操作可能丢失极细粒度的图像细节。

**社会影响（Societal Impact）**  
生成模型既有助于创意应用和技术普及，也有可能暴露训练数据和放大数据偏见[5, 90, 91]。应关注数据隐私和伦理风险。

---

# 6. 结论（Conclusion）

本文提出了潜空间扩散模型（Latent Diffusion Models, LDMs），在不降低生成质量的前提下，大幅提升了扩散模型的训练与采样效率。实验验证了该方法在多任务和多数据集上的优越性，并展示了其广泛适用性与复用潜力。

---

# 参考文献（References）

[1] Eirikur Agustsson and Radu Timofte. NTIRE 2017 challenge on single image super-resolution: Dataset and study. In 2017 IEEE Conference on Computer Vision and Pattern Recognition Workshops, CVPR Wor[...]  
[2] Martin Arjovsky, Soumith Chintala, and Léon Bottou. Wasserstein GAN, 2017. 3   
[3] Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale GAN training for high fidelity natural image synthesis. In Int. Conf. Learn. Represent., 2019. 1, 2, 7, 8, 22, 28  
...  
（其余参考文献略，保留编号和原文格式）

---

# 附录（Appendix）

（附录内容、表格、公式及图片说明与原文一致，建议按原文结构翻译，所有专业名词沿用中英文对照）

---