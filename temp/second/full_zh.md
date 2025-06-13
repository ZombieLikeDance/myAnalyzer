# 得到你想要的，而不是你不想要的：用于文本到图像扩散模型的图像内容抑制（GET WHAT YOU WANT, NOT WHAT YOU DON’T: IMAGE CONTENT SUPPRESSION FOR TEXT-TO-IMAGE DIFFUSION MODELS）

Senmao $\mathbf{Li}^{1}$（李森茂）, Joost van de Weijer$^2$, Taihang $\mathbf{Hu}^{1}$（胡台航）, Fahad Shahbaz Khan$^{3,4}$, Qibin Hou$^1$（侯启斌）  
Yaxing Wang$^1$（王亚星）, Jian Yang$^1$（杨健）  
1. VCIP，CS，南开大学（Nankai University）；2. 巴塞罗那自治大学（Universitat Autònoma de Barcelona）；3. 穆罕默德·本·扎耶德人工智能大学（Mohamed bin Zayed University of AI）；4. 林雪平大学（Linköping University）  
senmaonk,hutaihang00 @gmail.com, joost@cvc.uab.es  
fahad.khan@liu.se, {houqb ,yaxing,csjyang}@nankai.edu.cn

# 摘要（ABSTRACT）

近期文本到图像扩散模型（Text-to-Image Diffusion Models）取得成功，主要得益于其能够通过复杂的文本提示（Text Prompt）进行引导，使用户能够精确描述期望生成的内容。然而，这些模型在抑制负面目标（Negative Target）内容的生成方面仍面临挑战。例如，当用户给出“a face without glasses（没有眼镜的脸）”这样的提示时，现有模型往往难以有效排除“眼镜”元素的生成。为了解决这一问题，本文提出了一种新的负面目标内容抑制方法。该方法无需对图像生成器进行微调（Fine-tuning），也无需收集成对的训练数据。我们的主要贡献如下：（I）通过分析发现，[EOT]嵌入（End Of Text Embedding）包含了输入提示的大量冗余和重复语义信息；（II）提出了一种基于文本嵌入的内容抑制框架，通过奇异值分解（Singular Value Decomposition, SVD）和推理时优化（Inference-time Optimization）实现对负面目标内容的有效抑制；（III）在多个数据集和任务上，实验结果表明我们的方法优于现有基线，并具有较强的泛化能力。

# 1 引言（INTRODUCTION）

基于文本的图像生成（Text-based Image Generation）旨在根据用户提供的文本提示生成高质量、语义一致的图像（Ramesh et al., 2022; Saharia et al., 2022; Rombach et al., 2021）。用户通过文本提示与模型进行交互，描述期望生成的内容。然而，现有的文本到图像模型在有效抑制负面目标（如“无眼镜”、“无胡须”等）方面存在困难。例如，给定提示“a man without glasses（没有眼镜的男人）”，主流扩散模型（如Stable Diffusion, DeepFloyd-IF）通常会错误生成带有眼镜的形象。虽然“负面提示（Negative Prompt）”技术可以引导模型排除特定元素，但往往会导致结构和风格的改变，影响图像的整体质量。

![](images/cb1742607e58fb452d43a55fa20849b637e1137b0a6562774b7944afb1724af0.jpg)  
图1：Stable Diffusion（SD）和DeepFloyd-IF的失败案例。面对提示“无眼镜的男人”，两者都未能有效抑制“眼镜”的生成。我们的方案可显著提升抑制效果。

本文提出了一种替代方案，专注于文本嵌入（Text Embedding）层面的负面目标内容抑制方法，无需微调图像生成器或成对数据。主要包括：1）基于SVD的软加权正则化（Soft-weighted Regularization）；2）推理时文本嵌入优化（Inference-time Text Embedding Optimization）。实验证明，该方法在不同模型、数据集和应用场景下均取得了领先性能。

# 2 相关工作（RELATED WORK）

**文本到图像生成（Text-to-Image Generation）**：近年来，文本到图像合成（Text-to-Image Synthesis）技术迅猛发展，目标是生成与文本描述高度语义一致的高质量图像。代表性模型包括DALL-E、Imagen和Stable Diffusion（Saharia et al., 2022; Rombach et al., 2021）。

**基于扩散的图像生成（Diffusion-Based Image Generation）**：现有研究不断探索如何通过额外条件（如标签、掩码、文本等）来控制或编辑生成图像。扩散模型（Diffusion Model）在引导图像生成和编辑中表现出色。

**扩散模型中的语义擦除（Diffusion-Based Semantic Erasion）**：最近研究（Gandikota et al., 2023; Kumari et al., 2023; Zhang et al., 2023）关注于扩散模型中的语义擦除任务，包括去除版权、艺术家风格等内容。我们的工作则专注于通过文本嵌入实现更精细的内容抑制。

# 3 方法（METHOD）

我们的目标是在扩散模型（Diffusion Model）中抑制负面目标（Negative Target）的生成。为此，我们关注于操控文本嵌入（Text Embedding），该嵌入直接影响主体内容的生成。简单地删除负面词汇无法达到预期效果，因此我们提出基于SVD的软加权正则化和推理时优化来消除负面目标信息。

## 3.1 预备知识：扩散模型（PRELIMINARY: DIFFUSION MODEL）

Stable Diffusion（SD）首先训练编码器$E$和解码器$D$，编码器将图像$\mathbf{\vec{x}}$映射到潜在空间$z_0 = E(\pmb{x})$，解码器将其还原为$\hat{\pmb{x}} = D(z_0)$。SD训练了基于UNet的去噪网络$\epsilon_\theta$用于噪声预测，优化目标为：

$$
\operatorname*{min}_\theta E_{z_0, \epsilon \sim N(0,I), t \sim [1,T]} \left\| \epsilon - \epsilon_\theta(z_t, t, c) \right\|_2^2
$$

其中，文本嵌入$c$由预训练的CLIP文本编码器$\Gamma$根据条件化提示$\pmb{p}$提取：$\pmb{c} = \Gamma(\pmb{p})$。

## 3.2 [EOT]嵌入的分析（ANALYSIS OF [EOT] EMBEDDINGS）

文本编码器$\Gamma$将输入提示$\pmb{p}$映射为文本嵌入$\pmb{c} = \Gamma(\pmb{p}) \in \mathbb{R}^{M \times N}$（如SD中$M=768, N=77$）。我们发现，[EOT]嵌入（End Of Text Embedding）携带了大量语义信息。实验表明，[EOT]嵌入具有低秩特性（Low-rank Property），包含冗余语义。利用加权奇异值核范数最小化（Weighted Nuclear Norm Minimization, WNNM），保留主奇异值可提取主要语义分量。

## 3.3 基于文本嵌入的语义抑制（TEXT EMBEDDING-BASED SEMANTIC SUPPRESSION）

为实现负面目标内容抑制，我们必须从[EOT]嵌入中去除负面信息。我们提出通过奇异值分解（SVD）分离负面目标分量，并采用软加权正则化（Soft-weighted Regularization）如下：

$$
\hat{\sigma} = e^{-\sigma} * \sigma
$$

恢复嵌入矩阵$\hat{\pmb{\chi}} = \pmb{U} \hat{\pmb{\Sigma}} \pmb{V}^T$，其中$\hat{\pmb{\Sigma}} = diag(\hat{\sigma_0}, \hat{\sigma_1}, ...)$。通过重置主奇异值或次奇异值为0，可有效去除负面目标信息。

## 3.4 推理时文本嵌入优化（INFERENCE-TIME TEXT EMBEDDING OPTIMIZATION）

在扩散过程的特定时间步$t$，我们进一步抑制负面目标生成并增强正面目标信息。提出两种注意力损失（Attention Loss），分别正则化正负目标的注意力图（Attention Map），并修改文本嵌入：

正目标损失（Positive Target Loss）：
$$
\mathcal{L}_{pl} = \left.\hat{A}_t^{PE} - A_t^{PE}\right.^2
$$

负目标损失（Negative Target Loss）：
$$
\mathcal{L}_{nl} = -\left.\hat{A}_t^{NE} - A_t^{NE}\right.^2
$$

总目标函数：
$$
\mathcal{L} = \lambda_{pl}\mathcal{L}_{pl} + \lambda_{nl}\mathcal{L}_{nl}
$$

其中，$\lambda_{pl}=1$，$\lambda_{nl}=0.5$。我们利用该损失在推理时优化文本嵌入，实现内容抑制。

# 4 实验（EXPERIMENTS）

我们与多种基线方法（如Negative Prompt、ESD、Concept-ablation、Forget-Me-Not、Inst-Inpaint等）进行比较，数据集包括合成图像编辑和真实图像编辑。评估指标包括CLIPScore（生成图像与文本一致性）、FID（Frechet Inception Distance，衡量生成与真实分布距离）、DetScore（检测得分，基于MMDetection+GLIP检测负面目标物体）、用户研究等。

![](images/27f6eaf14ec850a0016e9725b8e6fcae337373c2b9c2c7a107912554e8737ef5.jpg)  
图6：真实图像与生成图像的负面目标抑制结果。我们的方法在无需微调SD模型的情况下，有效抑制负面目标。

实验结果显示，我们的方法在Clipscore和DetScore等指标上取得最优或次优表现，并能更好地平衡图像结构与内容抑制之间的关系。

# 5 结论与局限性（CONCLUSIONS AND LIMITATIONS）

我们发现扩散模型在抑制输入提示中的负面目标信息方面经常失败。通过对[EOT]嵌入的研究，我们提出了基于文本嵌入的内容抑制新范式，实现了高效、通用且无需微调的负面目标生成抑制。未来工作可进一步提升方法的稳定性和适应性。

# 致谢（ACKNOWLEDGEMENTS）

本工作受到西班牙MCIN/AEI/10.13039/501100011033项目（TED2021-132513B-I00, PID2022-143257NBI00）、欧洲联盟NextGenerationEU/PRTR和FEDER的资助。计算资源由巴塞罗那自治大学CVC和南开大学VCIP实验室提供。

# 参考文献（REFERENCES）

（此处省略，参见原文献引用，所有英文论文均可按学术规范中文括注原文标题。）

# 附录A：实现细节（APPENDIX A: IMPLEMENTATION DETAILS）

我们在推理阶段通过优化整体文本嵌入实现语义信息抑制，优化过程仅需35秒，无需额外网络参数。实验发现，早期步骤对主体空间位置影响最大，因此我们在第20步停止优化，剩余步骤保持原模型运行。每时刻内迭代次数设置为10。

# 附录B-F（APPENDIX B-F）

（算法公式、消融实验、更多图片结果、不同抑制强度、长句行为等，均可参考英文原文。）

---

> **注释说明：**
> - 所有专业术语均采用“中文（英文）”对照形式，如“扩散模型（Diffusion Model）”、"奇异值分解（Singular Value Decomposition, SVD）"。
> - 文中涉及的具体方法、模型、指标、名词和专有名词均在首次出现时提供中英文对照，后文可择要简写。
> - 公式、图片及表格请参见原文或原文图片编号。
