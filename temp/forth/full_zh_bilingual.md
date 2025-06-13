# SINE：基于文本到图像扩散模型（Text-to-Image Diffusion Models）的单图像编辑（SINgle Image Editing，SINE）

Zhixing Zhang¹，Ligong Han¹*，Arnab Ghosh²，Dimitris Metaxas¹，Jian Ren²  
¹罗格斯大学（Rutgers University）  ²Snap Inc.

# 摘要（Abstract）

近期扩散模型（diffusion models）相关工作展现出在图像生成条件设定（conditional image generation），如文本引导的图像合成（text-guided image synthesis）方面的强大能力。这一成果激发了许多尝试利用大规模预训练扩散模型（large-scale pretrained diffusion models）进行各种下游任务，包括文本引导的图像编辑（text-guided image editing）。然而，现有方法通常需要大量训练样本，或者在仅有单张图像（single image）时出现过拟合（overfitting）与失真（artifact）问题。在本文中，我们提出了SINE，一种利用预训练文本到图像扩散模型（pretrained text-to-image diffusion models）进行单图像编辑（single image editing）与内容操作（content manipulation）的新框架。我们的方法主要包括以下两项创新：（1）通过适当修改无分类器引导（classifier-free guidance），引入模型级无分类器引导（model-based classifier-free guidance），利用扩散模型自身为内容编辑提供得分引导（score guidance）；（2）提出基于图像块（patch-based）的微调策略（fine-tuning strategy），以解除像素位置（pixel position）与内容的相关性，使任意分辨率（arbitrary resolution）的生成成为可能。通过内容描述（text descriptor）和目标语言引导（language guidance），我们的方法能够对单一独特图像实现目标域（target domain）的编辑。实验表明，SINE能够在多种编辑任务中生成高保真（high-fidelity）、语义一致（semantically consistent）的结果，并在不同分辨率下表现出强大的泛化能力（generalization ability）。

# 1. 引言（Introduction）

自动化真实图像编辑（automatic real image editing）是一个令人兴奋的研究方向，使内容生成（content generation）与创作（creation）变得更加高效省力。尽管该领域已有众多研究成果，实现高保真（high-fidelity）语义编辑（semantic editing）仍具挑战性。近期基于扩散模型（diffusion model）的文本引导编辑方法（text-guided editing）取得进展，但仍存在如下局限：首先，微调过程（fine-tuning）可能导致大规模预训练模型（large-scale pretrained model）过拟合（overfitting），尤其是在仅有单张图像的情况下；其次，现有方法难以兼顾编辑多样性（editing diversity）与源图像保真性（fidelity to the source image）；此外，编辑分辨率受限（resolution limitation），无法灵活适应不同需求。

本文提出的SINE框架，依托现有文本到图像扩散模型（text-to-image diffusion models），通过模型级无分类器引导和基于图像块的微调方法，实现了单图像（single image）的高质量编辑。具体贡献包括：（1）提出模型级无分类器引导（model-based classifier-free guidance），有效利用预训练扩散模型信息，缓解过拟合和语言漂移（language drift）；（2）设计基于图像块的微调策略（patch-based fine-tuning strategy），使编辑模型能适应任意分辨率（arbitrary resolution），提升泛化能力（generalization ability）。通过内容描述（text descriptor）和目标文本引导（target language guidance），SINE能够实现丰富的图像编辑操作，包括风格变换（style transfer）、内容添加（context addition）、目标物体修改（object modification）和细节增强（detail enhancement）等。

# 2. 相关工作（Related Work）

文本引导的图像合成（text-guided image synthesis）在生成模型（generative model）领域引起了广泛关注。扩散模型（diffusion models）的快速发展为该任务提供了灵活而强大的生成先验（generative prior）。部分工作无需训练（training-free），直接利用预训练扩散模型（pretrained diffusion models）进行图像操作（image manipulation），如ILVR等。另一些研究则通过微调（fine-tuning）提升模型的语义编辑能力（semantic editing ability），例如DiffusionCLIP结合CLIP模型进行图像编辑，能有效传递文本语义信息。然而，同时实现高保真（high-fidelity）与高度文本对齐（alignment with text）的任务仍具挑战性。本文方法能够在单图像（single image）场景下，同时实现全局与局部的高质量语义编辑，并兼顾编辑多样性（editing diversity）与源图像保真性（fidelity to the source image）。

# 3. 方法（Methods）

以一张任意真实图像（in-the-wild image）为基础，我们的目标是在最大程度保留原始细节（original details）的前提下，通过语言描述（language guidance）进行编辑。为此，我们利用大规模训练的潜在扩散模型（Latent Diffusion Models, LDM），并在其基础上进行微调（fine-tuning）以实现特定编辑任务。为解决过拟合（overfitting）和分辨率受限（resolution limitation）等问题，我们提出了测试时模型级无分类器引导（test-time model-based classifier-free guidance）以及基于图像块的微调技术（patch-based fine-tuning technique）。方法流程如图2所示，具体包括以下部分：

## 3.1 语言引导扩散模型（Language-Guided Diffusion Models）

我们选用在大规模数据集上训练的潜在扩散模型（Latent Diffusion Models, LDMs）作为基础，通过对预训练模型的微调（fine-tuning）实现单图像编辑（single image editing）。LDM是一类基于去噪扩散过程（denoising diffusion process）的生成模型，其训练目标为最小化重构误差（reconstruction error），通过编码器（encoder）将图像映射到潜空间（latent space），并在该空间内进行去噪采样与重建（denoising sampling and reconstruction）。

## 3.2 基于模型的无分类器引导（Model-Based Classifier-Free Guidance）

在上述LDM基础上，我们受无分类器引导（classifier-free guidance）思想启发，提出模型级无分类器引导（model-based classifier-free guidance），以缓解单图像微调时的过拟合（overfitting）。无分类器引导是一种广泛应用于文本到图像扩散模型（text-to-image diffusion models）的技术，通过联合条件（conditional）和无条件（unconditional）目标进行训练，提升生成质量（generation quality）。针对单张图像，直接微调会导致模型过拟合和语言漂移（language drift）。为此，我们在采样阶段（sampling stage）结合预训练模型与微调模型共同提供去噪引导（denoising guidance），通过调节权重（weight）实现编辑与保真的平衡（trade-off between editing and fidelity）。此外，我们提出在扩散过程（diffusion process）不同时刻采用不同引导方式，进一步提升生成效果。

## 3.3 基于图像块的微调（Patch-Based Fine-Tuning）

基于模型级无分类器引导（model-based classifier-free guidance），我们可进行丰富的编辑操作。为进一步提升微调效果，改善分辨率受限（resolution limitation）问题，我们提出基于图像块的微调策略（patch-based fine-tuning strategy）。具体做法是，将单张训练图像视为定义在坐标空间（coordinate space）上的函数，随机裁剪（random crop）获得不同位置与尺度（position and scale）的图像块，并在这些块上进行微调训练（fine-tuning）。微调后，模型可通过直接输入位置信息生成不同分辨率（arbitrary resolution）的图像，实现任意分辨率编辑（arbitrary resolution editing）。

# 4. 实验（Experiments）

## 实施细节（Implementation Details）

我们的方法可适用于不同框架，实验以Stable Diffusion作为基础实现。微调时，随机裁剪图像块大小在$[0.1H, H] \times [0.1W, W]$范围内，统一调整为$512\times512$分辨率。实验涵盖多种编辑任务和分辨率设置，验证了方法的有效性（effectiveness）和泛化能力（generalization ability）。

## 4.1 定性评估（Qualitative Evaluation）

我们收集了来自Flickr和Unsplash等平台的多领域高分辨率图像。微调过程中采用粗略的分类器引导（coarse classifier guidance），确保编辑多样性。实验结果显示，模型级无分类器引导（model-based classifier-free guidance）可通过文本提示（text prompts）对单张真实图片实现多样化编辑（diverse editing），包括风格变换（style transfer）、内容添加（context addition）等，并在高分辨率下保持细节和结构一致性（detail and structure consistency）。

## 4.2 方法比较（Comparisons）

我们将SINE与当前主流单图像编辑方法Textual-Inversion和DreamBooth进行了对比。由于部分方法无公开实现，实验采用官方提供或复现流程。对比结果表明，SINE在编辑效果（editing effect）、图像保真度（fidelity）和结构保持（structure preservation）方面均优于其他方法。

## 4.3 消融实验（Ablation Analysis）

我们分别分析了基于图像块的微调策略（patch-based fine-tuning strategy）和模型级无分类器引导（model-based classifier-free guidance）的作用。在不同分辨率和编辑任务下，基于图像块的微调有效提升了生成质量（generation quality），避免了重复和伪影（duplication and artifact）现象。模型级无分类器引导则在平衡编辑强度和保真度（editing strength and fidelity）方面表现突出。进一步分析了引导步数$K$（guidance step）和权重$v$（weight）的影响，结果显示合理参数设定可显著提升编辑效果（editing effect）。

## 4.4 更多编辑任务（More Editing Tasks）

我们的框架同样适用于人脸局部与整体编辑（local and global face editing）、内容移除（content removal）、风格生成（style generation）与风格迁移（style transfer）等多种任务。实验展示了对人脸表情、背景等多类型的编辑，以及内容移除与风格迁移任务的应用，进一步验证了方法的通用性（generality）与有效性（effectiveness）。

# 5. 结论（Conclusion）

本文提出了SINE，一种用于单图像编辑（single image editing）的方法。仅需一张图片及其简要描述（brief description），即可实现任意分辨率下的多样化编辑（diverse editing at arbitrary resolution）。SINE在多种编辑任务中均表现出高保真（high-fidelity）和良好的文本对齐能力（text alignment ability）。然而，当编辑引导（editing guidance）过于模糊或变化幅度过大时，方法表现有限。未来可通过缓解微调模型过拟合进一步提升编辑保真度。

**致谢（Acknowledgments）**  
本研究部分得到了罗格斯大学（Rutgers University）D. Metaxas牵头的NSF IUCRC CARTA-1747778、2235405、2212301、1951890、2003874、2310966、FA9550-23-1-0417，以及NIH-5R01HL等项目资助（funding support）。

# 参考文献（References）

（此处省略，按原文顺序可双语呈现）

# 附录（Appendix）

在附录中，我们提供了如下内容：

- 更多与现有单图像编辑方法的对比（More comparisons with existing single image editing methods，Appendix A）；
- SINE在单图像编辑及其衍生新型图像操作任务中的更多结果（More editing results and novel image manipulation tasks enabled by SINE，Appendix B）；
- 更多消融实验分析（More ablation analysis，Appendix C）；
- 关于方法局限性及未来工作展望的讨论（Discussion on limitations of the method and future work，Appendix D）。

## A. 更多对比（More Comparisons）

在主文已有对比基础上，补充与Prompt-to-Prompt等方法的实验结果，并与无需训练的方法（如SDEdit）进行差异分析。技术层面，SINE可在保真度（fidelity）和细节保持（detail preservation）方面显著优于其他方法。

## B. 更多编辑结果（More Editing Results）

展示了SINE在不同场景下的多样编辑结果，包括风格迁移（style transfer）、内容生成（content generation）等，均在大规模文本到图像模型（large-scale text-to-image model）基础上通过我们提出的微调技术获得。

## C. 更多消融分析（More Ablation Analysis）

通过调节引导步数$K$（guidance step）和权重$v$（guidance weight），系统分析两者对编辑效果的影响。结果表明，参数设定的合理性直接关系到编辑强度与保真度的平衡（balance between editing strength and fidelity）。

## D. 局限性（Limitations）

当模型遇到模糊或极具挑战性的编辑指令时，生成效果可能不佳。此外，部分情况下会出现颜色与背景细节（color and background details）的异常变化。未来工作可通过正则化（regularization）等手段进一步缓解过拟合（overfitting）现象，提升编辑质量。

（表格与图片说明等不再赘述，具体参见原文。）