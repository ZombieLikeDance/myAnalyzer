# High-Resolution Image Synthesis with Latent Diffusion Models

Robin Rombach1 \* Andreas Blattmann1 ∗ Dominik Lorenz1 Patrick Esser Bj¨orn Ommer1 1Ludwig Maximilian University of Munich & IWR, Heidelberg University, Germany Runway ML https://github.com/CompVis/latent-diffusion

# Abstract

By decomposing the image formation process into a sequential application of denoising autoencoders, diffusion models (DMs) achieve state-of-the-art synthesis results on image data and beyond. Additionally, their formulation allows for a guiding mechanism to control the image generation process without retraining. However, since these models typically operate directly in pixel space, optimization of powerful DMs often consumes hundreds of GPU days and inference is expensive due to sequential evaluations. To enable DM training on limited computational resources while retaining their quality and flexibility, we apply them in the latent space of powerful pretrained autoencoders. In contrast to previous work, training diffusion models on such a representation allows for the first time to reach a near-optimal point between complexity reduction and detail preservation, greatly boosting visual fidelity. By introducing cross-attention layers into the model architecture, we turn diffusion models into powerful and flexible generators for general conditioning inputs such as text or bounding boxes and high-resolution synthesis becomes possible in a convolutional manner. Our latent diffusion models (LDMs) achieve new state-of-the-art scores for image inpainting and class-conditional image synthesis and highly competitive performance on various tasks, including text-to-image synthesis, unconditional image generation and super-resolution, while significantly reducing computational requirements compared to pixel-based DMs.

# 1. Introduction

Image synthesis is one of the computer vision fields with the most spectacular recent development, but also among those with the greatest computational demands. Especially high-resolution synthesis of complex, natural scenes is presently dominated by scaling up likelihood-based models, potentially containing billions of parameters in autoregressive (AR) transformers [66,67]. In contrast, the promising results of GANs [3, 27, 40] have been revealed to be mostly confined to data with comparably limited variability as their adversarial learning procedure does not easily scale to modeling complex, multi-modal distributions. Recently, diffusion models [82], which are built from a hierarchy of denoising autoencoders, have shown to achieve impressive results in image synthesis [30,85] and beyond [7,45,48,57], and define the state-of-the-art in class-conditional image synthesis [15,31] and super-resolution [72]. Moreover, even unconditional DMs can readily be applied to tasks such as inpainting and colorization [85] or stroke-based synthesis [53], in contrast to other types of generative models [19, 46, 69]. Being likelihood-based models, they do not exhibit mode-collapse and training instabilities as GANs and, by heavily exploiting parameter sharing, they can model highly complex distributions of natural images without involving billions of parameters as in AR models [67].

![](images/8d38237f7273d531864a21f790e6b62558dfee5b21220520ec0bd832fa418e60.jpg)  
Figure 1. Boosting the upper bound on achievable quality with less agressive downsampling. Since diffusion models offer excellent inductive biases for spatial data, we do not need the heavy spatial downsampling of related generative models in latent space, but can still greatly reduce the dimensionality of the data via suitable autoencoding models, see Sec. 3. Images are from the DIV2K [1] validation set, evaluated at $5 1 2 ^ { 2 }$ px. We denote the spatial downsampling factor by $f$ . Reconstruction FIDs [29] and PSNR are calculated on ImageNet-val. [12]; see also Tab. 8.

Democratizing High-Resolution Image Synthesis DMs belong to the class of likelihood-based models, whose mode-covering behavior makes them prone to spend excessive amounts of capacity (and thus compute resources) on modeling imperceptible details of the data [16, 73]. Although the reweighted variational objective [30] aims to address this by undersampling the initial denoising steps, DMs are still computationally demanding, since training and evaluating such a model requires repeated function evaluations (and gradient computations) in the high-dimensional space of RGB images. As an example, training the most powerful DMs often takes hundreds of GPU days (e.g. 150 - $1 0 0 0 \mathrm { V } 1 0 0$ days in [15]) and repeated evaluations on a noisy version of the input space render also inference expensive, so that producing $5 0 \mathrm { k }$ samples takes approximately 5 days [15] on a single A100 GPU. This has two consequences for the research community and users in general: Firstly, training such a model requires massive computational resources only available to a small fraction of the field, and leaves a huge carbon footprint [65, 86]. Secondly, evaluating an already trained model is also expensive in time and memory, since the same model architecture must run sequentially for a large number of steps (e. $\phantom { + } 3 . \ 2 5 - 1 0 0 0$ steps in [15]).

To increase the accessibility of this powerful model class and at the same time reduce its significant resource consumption, a method is needed that reduces the computational complexity for both training and sampling. Reducing the computational demands of DMs without impairing their performance is, therefore, key to enhance their accessibility.

Departure to Latent Space Our approach starts with the analysis of already trained diffusion models in pixel space: Fig. 2 shows the rate-distortion trade-off of a trained model. As with any likelihood-based model, learning can be roughly divided into two stages: First is a perceptual compression stage which removes high-frequency details but still learns little semantic variation. In the second stage, the actual generative model learns the semantic and conceptual composition of the data (semantic compression). We thus aim to first find a perceptually equivalent, but computationally more suitable space, in which we will train diffusion models for high-resolution image synthesis.

Following common practice [11, 23, 66, 67, 96], we separate training into two distinct phases: First, we train an autoencoder which provides a lower-dimensional (and thereby efficient) representational space which is perceptually equivalent to the data space. Importantly, and in contrast to previous work [23,66], we do not need to rely on excessive spatial compression, as we train DMs in the learned latent space, which exhibits better scaling properties with respect to the spatial dimensionality. The reduced complexity also provides efficient image generation from the latent space with a single network pass. We dub the resulting model class Latent Diffusion Models (LDMs).

A notable advantage of this approach is that we need to train the universal autoencoding stage only once and can therefore reuse it for multiple DM trainings or to explore possibly completely different tasks [81]. This enables efficient exploration of a large number of diffusion models for various image-to-image and text-to-image tasks. For the latter, we design an architecture that connects transformers to the DM’s UNet backbone [71] and enables arbitrary types of token-based conditioning mechanisms, see Sec. 3.3.

In sum, our work makes the following contributions:

(i) In contrast to purely transformer-based approaches [23, 66], our method scales more graceful to higher dimensional data and can thus (a) work on a compression level which provides more faithful and detailed reconstructions than previous work (see Fig. 1) and (b) can be efficiently

Semantic Compression   
80 $$ Generative Model:   
60 LatentDiffusionModel (LDM)   
40 Perceptual Compression   
20 $$ Autoencoder+GAN 0 0 0.5 1.5 Rate (bits/dim) 00e C 日

We propose latent diffusion models (LDMs) as an effective generative model and a separate mild compression stage that only eliminates imperceptible details. Data and images from [30].

applied to high-resolution synthesis of megapixel images.

(ii) We achieve competitive performance on multiple tasks (unconditional image synthesis, inpainting, stochastic super-resolution) and datasets while significantly lowering computational costs. Compared to pixel-based diffusion approaches, we also significantly decrease inference costs.

(iii) We show that, in contrast to previous work [93] which learns both an encoder/decoder architecture and a score-based prior simultaneously, our approach does not require a delicate weighting of reconstruction and generative abilities. This ensures extremely faithful reconstructions and requires very little regularization of the latent space.

(iv) We find that for densely conditioned tasks such as super-resolution, inpainting and semantic synthesis, our model can be applied in a convolutional fashion and render large, consistent images of $\sim 1 0 2 4 ^ { 2 }$ px.

(v) Moreover, we design a general-purpose conditioning mechanism based on cross-attention, enabling multi-modal training. We use it to train class-conditional, text-to-image and layout-to-image models.

(vi) Finally, we release pretrained latent diffusion and autoencoding models at https : / / github . com/CompVis/latent-diffusion which might be reusable for a various tasks besides training of DMs [81].

# 2. Related Work

Generative Models for Image Synthesis The high dimensional nature of images presents distinct challenges to generative modeling. Generative Adversarial Networks (GAN) [27] allow for efficient sampling of high resolution images with good perceptual quality [3, 42], but are difficult to optimize [2, 28, 54] and struggle to capture the full data distribution [55]. In contrast, likelihood-based methods emphasize good density estimation which renders optimization more well-behaved. Variational autoencoders (VAE) [46] and flow-based models [18, 19] enable efficient synthesis of high resolution images [9, 44, 92], but sample quality is not on par with GANs. While autoregressive models (ARM) [6, 10, 94, 95] achieve strong performance in density estimation, computationally demanding architectures [97] and a sequential sampling process limit them to low resolution images. Because pixel based representations of images contain barely perceptible, high-frequency details [16,73], maximum-likelihood training spends a disproportionate amount of capacity on modeling them, resulting in long training times. To scale to higher resolutions, several two-stage approaches [23,67,101,103] use ARMs to model a compressed latent image space instead of raw pixels.

Recently, Diffusion Probabilistic Models (DM) [82], have achieved state-of-the-art results in density estimation [45] as well as in sample quality [15]. The generative power of these models stems from a natural fit to the inductive biases of image-like data when their underlying neural backbone is implemented as a UNet [15, 30, 71, 85]. The best synthesis quality is usually achieved when a reweighted objective [30] is used for training. In this case, the DM corresponds to a lossy compressor and allow to trade image quality for compression capabilities. Evaluating and optimizing these models in pixel space, however, has the downside of low inference speed and very high training costs. While the former can be partially adressed by advanced sampling strategies [47, 75, 84] and hierarchical approaches [31, 93], training on high-resolution image data always requires to calculate expensive gradients. We adress both drawbacks with our proposed LDMs, which work on a compressed latent space of lower dimensionality. This renders training computationally cheaper and speeds up inference with almost no reduction in synthesis quality (see Fig. 1).

Two-Stage Image Synthesis To mitigate the shortcomings of individual generative approaches, a lot of research [11, 23, 67, 70, 101, 103] has gone into combining the strengths of different methods into more efficient and performant models via a two stage approach. VQ-VAEs [67, 101] use autoregressive models to learn an expressive prior over a discretized latent space. [66] extend this approach to text-to-image generation by learning a joint distributation over discretized image and text representations. More generally, [70] uses conditionally invertible networks to provide a generic transfer between latent spaces of diverse domains. Different from VQ-VAEs, VQGANs [23, 103] employ a first stage with an adversarial and perceptual objective to scale autoregressive transformers to larger images. However, the high compression rates required for feasible ARM training, which introduces billions of trainable parameters [23, 66], limit the overall performance of such approaches and less compression comes at the price of high computational cost [23, 66]. Our work prevents such tradeoffs, as our proposed LDMs scale more gently to higher dimensional latent spaces due to their convolutional backbone. Thus, we are free to choose the level of compression which optimally mediates between learning a powerful first stage, without leaving too much perceptual compression up to the generative diffusion model while guaranteeing highfidelity reconstructions (see Fig. 1).

While approaches to jointly [93] or separately [80] learn an encoding/decoding model together with a score-based prior exist, the former still require a difficult weighting between reconstruction and generative capabilities [11] and are outperformed by our approach (Sec. 4), and the latter focus on highly structured images such as human faces.

# 3. Method

To lower the computational demands of training diffusion models towards high-resolution image synthesis, we observe that although diffusion models allow to ignore perceptually irrelevant details by undersampling the corresponding loss terms [30], they still require costly function evaluations in pixel space, which causes huge demands in computation time and energy resources.

We propose to circumvent this drawback by introducing an explicit separation of the compressive from the generative learning phase (see Fig. 2). To achieve this, we utilize an autoencoding model which learns a space that is perceptually equivalent to the image space, but offers significantly reduced computational complexity.

Such an approach offers several advantages: (i) By leaving the high-dimensional image space, we obtain DMs which are computationally much more efficient because sampling is performed on a low-dimensional space. (ii) We exploit the inductive bias of DMs inherited from their UNet architecture [71], which makes them particularly effective for data with spatial structure and therefore alleviates the need for aggressive, quality-reducing compression levels as required by previous approaches [23, 66]. (iii) Finally, we obtain general-purpose compression models whose latent space can be used to train multiple generative models and which can also be utilized for other downstream applications such as single-image CLIP-guided synthesis [25].

# 3.1. Perceptual Image Compression

Our perceptual compression model is based on previous work [23] and consists of an autoencoder trained by combination of a perceptual loss [106] and a patch-based [33] adversarial objective [20, 23, 103]. This ensures that the reconstructions are confined to the image manifold by enforcing local realism and avoids bluriness introduced by relying solely on pixel-space losses such as $L _ { 2 }$ or $L _ { 1 }$ objectives.

More precisely, given an image $\boldsymbol { x } \in \mathbb { R } ^ { H \times W \times 3 }$ in RGB space, the encoder $\mathcal { E }$ encodes $x$ into a latent representation $z \ = \ \mathcal { E } ( x )$ , and the decoder $\mathcal { D }$ reconstructs the image from the latent, giving $\tilde { x } = \mathcal { D } ( z ) = \mathcal { D } ( \mathcal { E } ( x ) )$ , where $\boldsymbol { z } ~ \in ~ \mathbb { R } ^ { h \times w \times c }$ . Importantly, the encoder downsamples the image by a factor $f = H / h = W / w$ , and we investigate different downsampling factors $f = 2 ^ { m }$ , with $m \in \mathbb { N }$ .

In order to avoid arbitrarily high-variance latent spaces, we experiment with two different kinds of regularizations. The first variant, $K L$ -reg., imposes a slight KL-penalty towards a standard normal on the learned latent, similar to a VAE [46, 69], whereas $_ { V Q }$ -reg. uses a vector quantization layer [96] within the decoder. This model can be interpreted as a VQGAN [23] but with the quantization layer absorbed by the decoder. Because our subsequent DM is designed to work with the two-dimensional structure of our learned latent space $z = \mathcal { E } ( x )$ , we can use relatively mild compression rates and achieve very good reconstructions. This is in contrast to previous works [23, 66], which relied on an arbitrary 1D ordering of the learned space $z$ to model its distribution autoregressively and thereby ignored much of the inherent structure of $z$ . Hence, our compression model preserves details of $x$ better (see Tab. 8). The full objective and training details can be found in the supplement.

# 3.2. Latent Diffusion Models

Diffusion Models [82] are probabilistic models designed to learn a data distribution $p ( x )$ by gradually denoising a normally distributed variable, which corresponds to learning the reverse process of a fixed Markov Chain of length $T$ . For image synthesis, the most successful models [15,30,72] rely on a reweighted variant of the variational lower bound on $p ( x )$ , which mirrors denoising score-matching [85]. These models can be interpreted as an equally weighted sequence of denoising autoencoders $\epsilon _ { \theta } ( x _ { t } , t )$ ; $t = 1 \dots T$ , which are trained to predict a denoised variant of their input $\boldsymbol { x } _ { t }$ , where $\mathbf { \Psi } _ { x _ { t } }$ is a noisy version of the input $x$ . The corresponding objective can be simplified to (Sec. B)

$$
L _ { D M } = \mathbb { E } _ { x , \epsilon \sim \mathcal { N } ( 0 , 1 ) , t } \Big [ \| \epsilon - \epsilon _ { \theta } \big ( x _ { t } , t \big ) \| _ { 2 } ^ { 2 } \Big ] ,
$$

with $t$ uniformly sampled from $\{ 1 , \ldots , T \}$ .

Generative Modeling of Latent Representations With our trained perceptual compression models consisting of $\mathcal { E }$ and $\mathcal { D }$ , we now have access to an efficient, low-dimensional latent space in which high-frequency, imperceptible details are abstracted away. Compared to the high-dimensional pixel space, this space is more suitable for likelihood-based generative models, as they can now (i) focus on the important, semantic bits of the data and (ii) train in a lower dimensional, computationally much more efficient space.

Unlike previous work that relied on autoregressive, attention-based transformer models in a highly compressed, discrete latent space [23, 66, 103], we can take advantage of image-specific inductive biases that our model offers. This includes the ability to build the underlying UNet primarily from 2D convolutional layers, and further focusing the objective on the perceptually most relevant bits using the reweighted bound, which now reads

![](images/db81c24299a8d60f2f8ad1fccd08a26236fb1a6bdf4196ec5103f0e23c8d3ced.jpg)  
Figure 3. We condition LDMs either via concatenation or by a more general cross-attention mechanism. See Sec. 3.3

$$
L _ { L D M } : = \mathbb { E } _ { \mathcal { E } ( x ) , \epsilon \sim \mathcal { N } ( 0 , 1 ) , t } \Big [ \| \epsilon - \epsilon _ { \theta } \bigl ( z _ { t } , t \bigr ) \| _ { 2 } ^ { 2 } \Big ] .
$$

The neural backbone $\epsilon _ { \theta } ( \circ , t )$ of our model is realized as a time-conditional UNet [71]. Since the forward process is fixed, $z _ { t }$ can be efficiently obtained from $\mathcal { E }$ during training, and samples from $p ( z )$ can be decoded to image space with a single pass through $\mathcal { D }$ .

# 3.3. Conditioning Mechanisms

Similar to other types of generative models [56, 83], diffusion models are in principle capable of modeling conditional distributions of the form $p ( z | y )$ . This can be implemented with a conditional denoising autoencoder $\epsilon _ { \theta } ( z _ { t } , t , y )$ and paves the way to controlling the synthesis process through inputs $y$ such as text [68], semantic maps [33, 61] or other image-to-image translation tasks [34].

In the context of image synthesis, however, combining the generative power of DMs with other types of conditionings beyond class-labels [15] or blurred variants of the input image [72] is so far an under-explored area of research.

We turn DMs into more flexible conditional image generators by augmenting their underlying UNet backbone with the cross-attention mechanism [97], which is effective for learning attention-based models of various input modalities [35,36]. To pre-process $y$ from various modalities (such as language prompts) we introduce a domain specific encoder $\tau _ { \theta }$ that projects $y$ to an intermediate representation $\tau _ { \theta } ( y ) \in \mathbb { R } ^ { M \times d _ { \tau } }$ , which is then mapped to the intermediate layers of the UNet via a cross-attention layer implementing Attention $\begin{array} { r } { ( Q , K , V ) = \mathrm { s o f t m a x } \left( \frac { Q K ^ { T } } { \sqrt { d } } \right) \cdot \bar { V } } \end{array}$ , with

$$
Q = W _ { Q } ^ { ( i ) } \cdot \varphi _ { i } ( z _ { t } ) , K = W _ { K } ^ { ( i ) } \cdot \tau _ { \theta } ( y ) , V = W _ { V } ^ { ( i ) } \cdot \tau _ { \theta } ( y ) .
$$

Here, $\varphi _ { i } ( z _ { t } ) \in \mathbb { R } ^ { N \times d _ { \epsilon } ^ { i } }$ denotes a (flattened) intermediate representation of the UNet implementing $\epsilon _ { \theta }$ and $W _ { V } ^ { ( i ) } \ \in$

![](images/704204684dd11951ff72bedd794f708757d86fef46fe9070bcbbcc60a51e8d50.jpg)  
Figure 4. Samples from LDMs trained on CelebAHQ [39], FFHQ [41], LSUN-Churches [102], LSUN-Bedrooms [102] and classconditional ImageNet [12], each with a resolution of $2 5 6 \times 2 5 6$ . Best viewed when zoomed in. For more samples cf . the supplement.

$\mathbb { R } ^ { d \times d _ { \epsilon } ^ { i } }$ , $W _ { Q } ^ { ( i ) } \in \mathbb { R } ^ { d \times d _ { \tau } }$ & $W _ { K } ^ { \left( i \right) } \in \mathbb { R } ^ { d \times d _ { \tau } }$ are learnable projection matrices [36, 97]. See Fig. 3 for a visual depiction.

Based on image-conditioning pairs, we then learn the conditional LDM via

$$
L _ { L D M } : = \mathbb { E } _ { \mathcal { E } ( x ) , y , \epsilon \sim \mathcal { N } ( 0 , 1 ) , t } \Big [ | | \epsilon - \epsilon _ { \theta } \big ( z _ { t } , t , \tau _ { \theta } ( y ) \big ) | | _ { 2 } ^ { 2 } \Big ] ,
$$

where both $\tau _ { \theta }$ and $\epsilon _ { \theta }$ are jointly optimized via Eq. 3. This conditioning mechanism is flexible as $\tau _ { \theta }$ can be parameterized with domain-specific experts, e.g. (unmasked) transformers [97] when $y$ are text prompts (see Sec. 4.3.1)

# 4. Experiments

LDMs provide means to flexible and computationally tractable diffusion based image synthesis of various image modalities, which we empirically show in the following. Firstly, however, we analyze the gains of our models compared to pixel-based diffusion models in both training and inference. Interestingly, we find that $L D M s$ trained in VQregularized latent spaces sometimes achieve better sample quality, even though the reconstruction capabilities of VQregularized first stage models slightly fall behind those of their continuous counterparts, $c f$ . Tab. 8. A visual comparison between the effects of first stage regularization schemes on $L D M$ training and their generalization abilities to resolutions $> 2 5 6 ^ { 2 }$ can be found in Appendix D.1. In E.2 we list details on architecture, implementation, training and evaluation for all results presented in this section.

# 4.1. On Perceptual Compression Tradeoffs

This section analyzes the behavior of our LDMs with different downsampling factors $f \in \{ 1 , 2 , 4 , 8 , 1 6 , 3 2 \}$ (abbreviated as $L D M - f$ , where $L D M - I$ corresponds to pixel-based DMs). To obtain a comparable test-field, we fix the computational resources to a single NVIDIA A100 for all experiments in this section and train all models for the same number of steps and with the same number of parameters.

Tab. 8 shows hyperparameters and reconstruction performance of the first stage models used for the LDMs compared in this section. Fig. 6 shows sample quality as a function of training progress for 2M steps of class-conditional models on the ImageNet [12] dataset. We see that, i) small downsampling factors for $L D M \ – \{ l , 2 \}$ result in slow training progress, whereas ii) overly large values of $f$ cause stagnating fidelity after comparably few training steps. Revisiting the analysis above (Fig. 1 and 2) we attribute this to i) leaving most of perceptual compression to the diffusion model and ii) too strong first stage compression resulting in information loss and thus limiting the achievable quality. LDM- $\{ 4 - l 6 \}$ strike a good balance between efficiency and perceptually faithful results, which manifests in a significant FID [29] gap of 38 between pixel-based diffusion (LDM-1) and LDM-8 after 2M training steps.

In Fig. 7, we compare models trained on CelebAHQ [39] and ImageNet in terms sampling speed for different numbers of denoising steps with the DDIM sampler [84] and plot it against FID-scores [29]. LDM- $\{ 4 \mathrm { - } \delta \}$ outperform models with unsuitable ratios of perceptual and conceptual compression. Especially compared to pixel-based LDM-1, they achieve much lower FID scores while simultaneously significantly increasing sample throughput. Complex datasets such as ImageNet require reduced compression rates to avoid reducing quality. In summary, LDM-4 and -8 offer the best conditions for achieving high-quality synthesis results.

# 4.2. Image Generation with Latent Diffusion

We train unconditional models of $2 5 6 ^ { 2 }$ images on CelebA-HQ [39], FFHQ [41], LSUN-Churches and -Bedrooms [102] and evaluate the i) sample quality and ii) their coverage of the data manifold using ii) FID [29] and ii) Precision-and-Recall [50]. Tab. 1 summarizes our results. On CelebA-HQ, we report a new state-of-the-art FID of 5.11, outperforming previous likelihood-based models as well as GANs. We also outperform LSGM [93] where a latent diffusion model is trained jointly together with the first stage. In contrast, we train diffusion models in a fixed space and avoid the difficulty of weighing reconstruction quality against learning the prior over the latent space, see Fig. 1-2.

![](images/2ea3378722c1c62e380b7c98c7d5f9ded4dad792fe21d4658cdca2d49e5b7aad.jpg)  
Figure 5. Samples for user-defined text prompts from our model for text-to-image synthesis, LDM-8 $( K L )$ , which was trained on the LAION [78] database. Samples generated with 200 DDIM steps and $\eta = 1 . 0$ . We use unconditional guidance [32] with $s = 1 0 . 0$ .

![](images/3cab82fe9ef3813eaecfae293bd03c7010ae48508facddf7c3de21d8b40e4144.jpg)  
Figure 6. Analyzing the training of class-conditional $L D M s$ with different downsampling factors $f$ over 2M train steps on the ImageNet dataset. Pixel-based LDM-1 requires substantially larger train times compared to models with larger downsampling factors $( L D M - \{ 4 - I 6 \} )$ ). Too much perceptual compression as in $L D M - 3 2$ limits the overall sample quality. All models are trained on a single NVIDIA A100 with the same computational budget. Results obtained with 100 DDIM steps [84] and $\kappa = 0$ .

Text-to-Image Synthesis on LAION. 1.45B Model.   

<html><body><table><tr><td colspan="4">CelebA-HQ 256 × 256</td><td colspan="4">FFHQ 256 × 256</td></tr><tr><td>Method</td><td>FID↓</td><td>Prec. ↑</td><td>Recall ↑</td><td>Method</td><td>FID↓</td><td>Prec. ↑</td><td>Recall ↑</td></tr><tr><td>DC-VAE [63]</td><td>15.8</td><td>·</td><td></td><td>ImageBART [21]</td><td>9.57</td><td></td><td>：</td></tr><tr><td>VQGAN+T. [23] (k=400)</td><td>10.2</td><td>·</td><td></td><td>U-Net GAN (+aug) [77]</td><td>10.9 (7.6)</td><td></td><td>·</td></tr><tr><td>PGGAN [39]</td><td>8.0</td><td>·</td><td></td><td>UDM [43]</td><td>5.54</td><td></td><td>·</td></tr><tr><td>LSGM [93]</td><td>7.22</td><td>：：</td><td></td><td>StyleGAN [41]</td><td>4.16</td><td>0.71</td><td>0.46</td></tr><tr><td>UDM [43]</td><td>7.16</td><td></td><td></td><td>ProjectedGAN [76]</td><td>3.08</td><td>0.65</td><td>0.46</td></tr><tr><td>LDM-4 (ours, 500-s†)</td><td>5.11</td><td>0.72</td><td>0.49</td><td>LDM-4 (ours, 200-s)</td><td>4.98</td><td>0.73</td><td>0.50</td></tr><tr><td colspan="4">LSUN-Churches 256 × 256</td><td colspan="4">LSUN-Bedrooms 256 × 256</td></tr><tr><td>Method</td><td>FID↓</td><td>Prec. ↑</td><td>Recall ↑</td><td>Method</td><td>FID↓</td><td>Prec. ↑</td><td>Recall ↑</td></tr><tr><td>DDPM [30]</td><td>7.89</td><td></td><td></td><td>ImageBART [21]</td><td>5.51</td><td></td><td></td></tr><tr><td>ImageBART [21]</td><td>7.32</td><td></td><td>·</td><td>DDPM [30]</td><td>4.9</td><td>-</td><td>：</td></tr><tr><td>PGGAN [39]</td><td>6.42</td><td></td><td></td><td>UDM [43]</td><td>4.57</td><td>-</td><td>-</td></tr><tr><td>StyleGAN [41]</td><td>4.21</td><td></td><td>·</td><td>StyleGAN [41]</td><td>2.35</td><td>0.59</td><td>0.48</td></tr><tr><td>StyleGAN2 [42]</td><td>3.86</td><td></td><td></td><td>ADM [15]</td><td>1.90</td><td>0.66</td><td>0.51</td></tr><tr><td>ProjectedGAN [76]</td><td>1.59</td><td>0.61</td><td>0.44</td><td>ProjectedGAN [76]</td><td>1.52</td><td>0.61</td><td>0.34</td></tr><tr><td>LDM-8* (ours, 200-s)</td><td>4.02</td><td>0.64</td><td>0.52</td><td>LDM-4 (ours,200-s)</td><td>2.95</td><td>0.66</td><td>0.48</td></tr></table></body></html>

![](images/7e401b357470ca9b09aee4e5e9a45dc4abbe8e145c8b242074657f1dee1756e5.jpg)  
Figure 7. Comparing LDMs with varying compression on the CelebA-HQ (left) and ImageNet (right) datasets. Different markers indicate $\{ 1 0 , 2 0 , 5 0 , 1 0 0 , 2 0 0 \}$ sampling steps using DDIM, from right to left along each line. The dashed line shows the FID scores for 200 steps, indicating the strong performance of $L D M -$ $\{ 4 \cdot 8 \}$ . FID scores assessed on 5000 samples. All models were trained for 500k (CelebA) / 2M (ImageNet) steps on an A100.

We outperform prior diffusion based approaches on all but the LSUN-Bedrooms dataset, where our score is close to ADM [15], despite utilizing half its parameters and requiring 4-times less train resources (see Appendix E.3.5).

Table 1. Evaluation metrics for unconditional image synthesis. CelebA-HQ results reproduced from [43, 63, 100], FFHQ from [42, 43]. $^ \dagger$ : $N$ -s refers to $N$ sampling steps with the DDIM [84] sampler. ∗: trained in $K L$ -regularized latent space. Additional results can be found in the supplementary.

Table 2. Evaluation of text-conditional image synthesis on the $2 5 6 \times 2 5 6$ -sized MS-COCO [51] dataset: with 250 DDIM [84] steps our model is on par with the most recent diffusion [59] and autoregressive [26] methods despite using significantly less parameters. $^ \dag / ^ { \ast }$ :Numbers from [109]/ [26]   

<html><body><table><tr><td colspan="5">Text-Conditional Image Synthesis</td></tr><tr><td>Method</td><td>FID↓</td><td>IS↑</td><td>Nparams</td><td></td></tr><tr><td>CogView† [17]</td><td>27.10</td><td>18.20</td><td>4B</td><td rowspan="2">self-ranking, rejection rate 0.017</td></tr><tr><td>LAFITE† [109]</td><td>26.94</td><td>26.02</td><td>75M</td></tr><tr><td>GLIDE* [59]</td><td>12.24</td><td>：：</td><td>6B</td><td>277 DDIM steps, c.f.g.[32] s = 3</td></tr><tr><td>Make-A-Scene* [26]</td><td>11.84</td><td></td><td>4B</td><td>c.f.g for AR models [98] s = 5</td></tr><tr><td>LDM-KL-8</td><td>23.31</td><td>20.03 ±0.33</td><td>1.45B</td><td>250 DDIM steps</td></tr><tr><td>LDM-KL-8-G*</td><td>12.63</td><td>30.29 ±0.42</td><td>1.45B</td><td>250 DDIM steps, c.f.g.[32] s = 1.5</td></tr></table></body></html>

Moreover, LDMs consistently improve upon GAN-based methods in Precision and Recall, thus confirming the advantages of their mode-covering likelihood-based training objective over adversarial approaches. In Fig. 4 we also show qualitative results on each dataset.

![](images/572cefef863256acbbdbd59ed8a631e14e81f56c80c92877f9eba2a6ebea2cf3.jpg)  
Figure 8. Layout-to-image synthesis with an LDM on COCO [4], see Sec. 4.3.1. Quantitative evaluation in the supplement D.3.

# 4.3. Conditional Latent Diffusion

# 4.3.1 Transformer Encoders for LDMs

By introducing cross-attention based conditioning into LDMs we open them up for various conditioning modalities previously unexplored for diffusion models. For textto-image image modeling, we train a 1.45B parameter $K L$ -regularized LDM conditioned on language prompts on LAION-400M [78]. We employ the BERT-tokenizer [14] and implement $\tau _ { \theta }$ as a transformer [97] to infer a latent code which is mapped into the UNet via (multi-head) crossattention (Sec. 3.3). This combination of domain specific experts for learning a language representation and visual synthesis results in a powerful model, which generalizes well to complex, user-defined text prompts, cf . Fig. 8 and 5. For quantitative analysis, we follow prior work and evaluate text-to-image generation on the MS-COCO [51] validation set, where our model improves upon powerful AR [17, 66] and GAN-based [109] methods, cf . Tab. 2. We note that applying classifier-free diffusion guidance [32] greatly boosts sample quality, such that the guided LDM-KL-8- $G$ is on par with the recent state-of-the-art AR [26] and diffusion models [59] for text-to-image synthesis, while substantially reducing parameter count. To further analyze the flexibility of the cross-attention based conditioning mechanism we also train models to synthesize images based on semantic layouts on OpenImages [49], and finetune on COCO [4], see Fig. 8. See Sec. D.3 for the quantitative evaluation and implementation details.

Lastly, following prior work [3, 15, 21, 23], we evaluate our best-performing class-conditional ImageNet models with $f \in \{ 4 , 8 \}$ from Sec. 4.1 in Tab. 3, Fig. 4 and Sec. D.4. Here we outperform the state of the art diffusion model ADM [15] while significantly reducing computational requirements and parameter count, cf . Tab 18.

# 4.3.2 Convolutional Sampling Beyond $2 5 6 ^ { 2 }$

By concatenating spatially aligned conditioning information to the input of $\epsilon _ { \theta }$ , $L D M s$ can serve as efficient generalpurpose image-to-image translation models. We use this to train models for semantic synthesis, super-resolution (Sec. 4.4) and inpainting (Sec. 4.5). For semantic synthesis, we use images of landscapes paired with semantic maps [23, 61] and concatenate downsampled versions of the semantic maps with the latent image representation of a $f = 4$ model (VQ-reg., see Tab. 8). We train on an input resolution of $2 5 6 ^ { 2 }$ (crops from $3 8 4 ^ { 2 }$ ) but find that our model generalizes to larger resolutions and can generate images up to the megapixel regime when evaluated in a convolutional manner (see Fig. 9). We exploit this behavior to also apply the super-resolution models in Sec. 4.4 and the inpainting models in Sec. 4.5 to generate large images between $5 1 2 ^ { 2 }$ and $1 0 2 4 ^ { 2 }$ . For this application, the signal-to-noise ratio (induced by the scale of the latent space) significantly affects the results. In Sec. D.1 we illustrate this when learning an LDM on (i) the latent space as provided by a $f = 4$ model (KL-reg., see Tab. 8), and (ii) a rescaled version, scaled by the component-wise standard deviation.

Table 3. Comparison of a class-conditional ImageNet $L D M$ with recent state-of-the-art methods for class-conditional image generation on ImageNet [12]. A more detailed comparison with additional baselines can be found in D.4, Tab. 10 and F. $c . f . g .$ . denotes classifier-free guidance with a scale $s$ as proposed in [32].   

<html><body><table><tr><td>Method</td><td>FID↓</td><td>IS↑</td><td>Precision↑</td><td>Recall↑</td><td>Nparams</td><td></td></tr><tr><td>BigGan-deep [3]</td><td>6.95</td><td>203.6±2.6</td><td>0.87</td><td>0.28</td><td>340M</td><td></td></tr><tr><td>ADM [15]</td><td>10.94</td><td>100.98</td><td>0.69</td><td>0.63</td><td>554M</td><td>250 DDIM steps</td></tr><tr><td>ADM-G [15]</td><td>4.59</td><td>186.7</td><td>0.82</td><td>0.52</td><td>608M</td><td>250 DDIM steps</td></tr><tr><td>LDM-4 (ours)</td><td>10.56</td><td>103.49 ± 1.24</td><td>0.71</td><td>0.62</td><td>400M</td><td>250 DDIM steps</td></tr><tr><td>LDM-4-G (ours)</td><td>3.60</td><td>247.67 ± 5.59</td><td>0.87</td><td>0.48</td><td>400M</td><td>250 steps, c.f.g [32], s =1.5</td></tr></table></body></html>

The latter, in combination with classifier-free guidance [32], also enables the direct synthesis of $> 2 5 6 ^ { 2 }$ images for the text-conditional LDM-KL-8-G as in Fig. 13.

![](images/a21a225f952a111dba8a4474f7fbd44dea605b66fdd4d976498a1ccf8c665784.jpg)  
Figure 9. A LDM trained on $2 5 6 ^ { 2 }$ resolution can generalize to larger resolution (here: $5 1 2 \times 1 0 2 4 )$ for spatially conditioned tasks such as semantic synthesis of landscape images. See Sec. 4.3.2.

# 4.4. Super-Resolution with Latent Diffusion

LDMs can be efficiently trained for super-resolution by diretly conditioning on low-resolution images via concatenation (cf . Sec. 3.3). In a first experiment, we follow SR3 [72] and fix the image degradation to a bicubic interpolation with $4 \times$ -downsampling and train on ImageNet following SR3’s data processing pipeline. We use the $f = 4$ autoencoding model pretrained on OpenImages (VQ-reg., cf . Tab. 8) and concatenate the low-resolution conditioning $y$ and the inputs to the UNet, i.e. $\tau _ { \theta }$ is the identity. Our qualitative and quantitative results (see Fig. 10 and Tab. 5) show competitive performance and LDM-SR outperforms SR3 in FID while SR3 has a better IS. A simple image regression model achieves the highest PSNR and SSIM scores; however these metrics do not align well with human perception [106] and favor blurriness over imperfectly aligned high frequency details [72]. Further, we conduct a user study comparing the pixel-baseline with LDM-SR. We follow SR3 [72] where human subjects were shown a low-res image in between two high-res images and asked for preference. The results in Tab. 4 affirm the good performance of LDM-SR. PSNR and SSIM can be pushed by using a post-hoc guiding mechanism [15] and we implement this image-based guider via a perceptual loss, see Sec. D.6.

![](images/b4407decee92e7d6c8c40ec8292538b3c7e6471167189489b5db07523168e55f.jpg)  
Figure 10. ImageNet $6 4  2 5 6$ super-resolution on ImageNet-Val. $L D M  – S R$ has advantages at rendering realistic textures but SR3 can synthesize more coherent fine structures. See appendix for additional samples and cropouts. SR3 results from [72].

<html><body><table><tr><td rowspan="2">User Study</td><td colspan="2">SR on ImageNet</td><td colspan="2">Inpainting on Places</td></tr><tr><td>Pixel-DM(f1)</td><td>LDM-4</td><td>LAMA [88]</td><td>LDM-4</td></tr><tr><td>Task 1: Preference vs GT个</td><td>16.0%</td><td>30.4%</td><td>13.6%</td><td>21.0%</td></tr><tr><td>Task 2:Preference Score ↑</td><td>29.4%</td><td>70.6%</td><td>31.9%</td><td>68.1%</td></tr></table></body></html>

Table 4. Task 1: Subjects were shown ground truth and generated image and asked for preference. Task 2: Subjects had to decide between two generated images. More details in E.3.6

Since the bicubic degradation process does not generalize well to images which do not follow this pre-processing, we also train a generic model, LDM-BSR, by using more diverse degradation. The results are shown in Sec. D.6.1.

Table 5. $\times 4$ upscaling results on ImageNet-Val. $( 2 5 6 ^ { 2 } )$ ; †: FID features computed on validation split, ‡: FID features computed on train split; ∗: Assessed on a NVIDIA A100   

<html><body><table><tr><td>Method</td><td>FID↓</td><td>IS ↑</td><td>PSNR ↑</td><td>SSIM↑</td><td>Nparams</td><td></td></tr><tr><td>Image Regression [72]</td><td>15.2</td><td>121.1</td><td>27.9</td><td>0.801</td><td>625M</td><td>N/A</td></tr><tr><td>SR3 [72]</td><td>5.2</td><td>180.1</td><td>26.4</td><td>0.762</td><td>625M</td><td>N/A</td></tr><tr><td>LDM-4 (ours, 100 steps)</td><td>2.8+/4.8</td><td>166.3</td><td>24.4±3.8</td><td>0.69±0.14</td><td>169M</td><td>4.62</td></tr><tr><td>emphLDM-4 (ours, big, 100 steps)</td><td>2.4†/4.3</td><td>174.9</td><td>24.7±4.1</td><td>0.71±0.15</td><td>552M</td><td>4.5</td></tr><tr><td>LDM-4 (ours, 50 steps, guiding)</td><td>4.4†/6.4‡</td><td>153.7</td><td>25.8 ±3.7</td><td>0.74±0.12</td><td>184M</td><td>0.38</td></tr></table></body></html>

<html><body><table><tr><td>Model (reg.-type)</td><td>train throughput samples/sec.</td><td>sampling throughput†</td><td>@512</td><td>train+val hours/epoch</td><td>FID@2k epoch 6</td></tr><tr><td></td><td></td><td>@256</td><td></td><td></td><td></td></tr><tr><td>LDM-1 (no first stage) LDM-4 (KL,w/ attn)</td><td>0.11 0.32</td><td>0.26 0.97</td><td>0.07 0.34</td><td>20.66 7.66</td><td>24.74 15.21</td></tr><tr><td>LDM-4 (VQ,w/ attn)</td><td>0.33</td><td>0.97</td><td>0.34</td><td>7.04</td><td>14.99</td></tr><tr><td>LDM-4 (VQ,w/o attn)</td><td>0.35</td><td>0.99</td><td>0.36</td><td>6.66</td><td>15.95</td></tr></table></body></html>

Table 6. Assessing inpainting efficiency. †: Deviations from Fig. 7 due to varying GPU settings/batch sizes cf . the supplement.

# 4.5. Inpainting with Latent Diffusion

Inpainting is the task of filling masked regions of an image with new content either because parts of the image are are corrupted or to replace existing but undesired content within the image. We evaluate how our general approach for conditional image generation compares to more specialized, state-of-the-art approaches for this task. Our evaluation follows the protocol of LaMa [88], a recent inpainting model that introduces a specialized architecture relying on Fast Fourier Convolutions [8]. The exact training & evaluation protocol on Places [108] is described in Sec. E.2.2.

We first analyze the effect of different design choices for the first stage. In particular, we compare the inpainting efficiency of LDM-1 (i.e. a pixel-based conditional DM) with $L D M { - } 4$ , for both $K L$ and $V Q$ regularizations, as well as $V Q$ - $L D M { - } 4$ without any attention in the first stage (see Tab. 8), where the latter reduces GPU memory for decoding at high resolutions. For comparability, we fix the number of parameters for all models. Tab. 6 reports the training and sampling throughput at resolution $2 5 6 ^ { 2 }$ and $5 1 2 ^ { 2 }$ , the total training time in hours per epoch and the FID score on the validation split after six epochs. Overall, we observe a speed-up of at least $2 . 7 \times$ between pixel- and latent-based diffusion models while improving FID scores by a factor of at least $1 . 6 \times$ .

The comparison with other inpainting approaches in Tab. 7 shows that our model with attention improves the overall image quality as measured by FID over that of [88]. LPIPS between the unmasked images and our samples is slightly higher than that of [88]. We attribute this to [88] only producing a single result which tends to recover more of an average image compared to the diverse results produced by our LDM cf . Fig. 21. Additionally in a user study (Tab. 4) human subjects favor our results over those of [88].

Based on these initial results, we also trained a larger diffusion model (big in Tab. 7) in the latent space of the VQregularized first stage without attention. Following [15], the UNet of this diffusion model uses attention layers on three levels of its feature hierarchy, the BigGAN [3] residual block for up- and downsampling and has 387M parameters instead of 215M. After training, we noticed a discrepancy in the quality of samples produced at resolutions $2 5 6 ^ { 2 }$ and $5 1 2 ^ { 2 }$ , which we hypothesize to be caused by the additional attention modules. However, fine-tuning the model for half an epoch at resolution $5 1 2 ^ { 2 }$ allows the model to adjust to the new feature statistics and sets a new state of the art FID on image inpainting (big, w/o attn, $w / f t$ in Tab. 7, Fig. 11.).

![](images/3cabfc911afab57933f9594e1f349b70b1b68d1989000ddeb559b8a71313e861.jpg)  
Figure 11. Qualitative results on object removal with our big, w/ $\mathbf { \mathcal { f } } t$ inpainting model. For more results, see Fig. 22.

# 5. Limitations & Societal Impact

Limitations While LDMs significantly reduce computational requirements compared to pixel-based approaches, their sequential sampling process is still slower than that of GANs. Moreover, the use of LDMs can be questionable when high precision is required: although the loss of image quality is very small in our $f = 4$ autoencoding models (see Fig. 1), their reconstruction capability can become a bottleneck for tasks that require fine-grained accuracy in pixel space. We assume that our superresolution models (Sec. 4.4) are already somewhat limited in this respect.

Societal Impact Generative models for media like imagery are a double-edged sword: On the one hand, they

<html><body><table><tr><td rowspan="2">Method</td><td colspan="2">40-50% masked</td><td colspan="2">All samples</td></tr><tr><td>FID↓</td><td>LPIPS↓</td><td>FID↓</td><td>LPIPS↓</td></tr><tr><td>LDM-4 (ours,big,w/ ft)</td><td>9.39</td><td>0.246± 0.042</td><td>1.50</td><td>0.137± 0.080</td></tr><tr><td>LDM-4 (ours,big,w/o ft)</td><td>12.89</td><td>0.257± 0.047</td><td>2.40</td><td>0.142± 0.085</td></tr><tr><td>LDM-4 (ours,w/ attn)</td><td>11.87</td><td>0.257± 0.042</td><td>2.15</td><td>0.144± 0.084</td></tr><tr><td>LDM-4 (ours,w/o attn)</td><td>12.60</td><td>0.259± 0.041</td><td>2.37</td><td>0.145± 0.084</td></tr><tr><td>LaMa [88]†</td><td>12.31</td><td>0.243± 0.038</td><td>2.23</td><td>0.134± 0.080</td></tr><tr><td>LaMa [88]</td><td>12.0</td><td>0.24</td><td>2.21</td><td>0.14</td></tr><tr><td>CoModGAN[107]</td><td>10.4</td><td>0.26</td><td>1.82</td><td>0.15</td></tr><tr><td>RegionWise [52]</td><td>21.3</td><td>0.27</td><td>4.75</td><td>0.15</td></tr><tr><td>DeepFill v2 [104]</td><td>22.1</td><td>0.28</td><td>5.20</td><td>0.16</td></tr><tr><td>EdgeConnect [58]</td><td>30.5</td><td>0.28</td><td>8.37</td><td>0.16</td></tr></table></body></html>

Table 7. Comparison of inpainting performance on 30k crops of size $5 1 2 \times 5 1 2$ from test images of Places [108]. The column $_ { 4 0 }$ - $50 \%$ reports metrics computed over hard examples where $40 \text{‰}$ of the image region have to be inpainted. †recomputed on our test set, since the original test set used in [88] was not available.

enable various creative applications, and in particular approaches like ours that reduce the cost of training and inference have the potential to facilitate access to this technology and democratize its exploration. On the other hand, it also means that it becomes easier to create and disseminate manipulated data or spread misinformation and spam. In particular, the deliberate manipulation of images (“deep fakes”) is a common problem in this context, and women in particular are disproportionately affected by it [13, 24].

Generative models can also reveal their training data [5, 90], which is of great concern when the data contain sensitive or personal information and were collected without explicit consent. However, the extent to which this also applies to DMs of images is not yet fully understood.

Finally, deep learning modules tend to reproduce or exacerbate biases that are already present in the data [22, 38, 91]. While diffusion models achieve better coverage of the data distribution than e.g. GAN-based approaches, the extent to which our two-stage approach that combines adversarial training and a likelihood-based objective misrepresents the data remains an important research question.

For a more general, detailed discussion of the ethical considerations of deep generative models, see e.g. [13].

# 6. Conclusion

We have presented latent diffusion models, a simple and efficient way to significantly improve both the training and sampling efficiency of denoising diffusion models without degrading their quality. Based on this and our crossattention conditioning mechanism, our experiments could demonstrate favorable results compared to state-of-the-art methods across a wide range of conditional image synthesis tasks without task-specific architectures.

# References

[1] Eirikur Agustsson and Radu Timofte. NTIRE 2017 challenge on single image super-resolution: Dataset and study. In 2017 IEEE Conference on Computer Vision and Pattern Recognition Workshops, CVPR Workshops 2017, Honolulu, HI, USA, July 21-26, 2017, pages 1122–1131. IEEE Computer Society, 2017. 1   
[2] Martin Arjovsky, Soumith Chintala, and Le´on Bottou. Wasserstein gan, 2017. 3   
[3] Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale GAN training for high fidelity natural image synthesis. In Int. Conf. Learn. Represent., 2019. 1, 2, 7, 8, 22, 28   
[4] Holger Caesar, Jasper R. R. Uijlings, and Vittorio Ferrari. Coco-stuff: Thing and stuff classes in context. In 2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18- 22, 2018, pages 1209–1218. Computer Vision Foundation / IEEE Computer Society, 2018. 7, 20, 22   
[5] Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, et al. Extracting training data from large language models. In 30th USENIX Security Symposium (USENIX Security 21), pages 2633–2650, 2021. 9   
[6] Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In ICML, volume 119 of Proceedings of Machine Learning Research, pages 1691–1703. PMLR, 2020. 3   
[7] Nanxin Chen, Yu Zhang, Heiga Zen, Ron J. Weiss, Mohammad Norouzi, and William Chan. Wavegrad: Estimating gradients for waveform generation. In ICLR. OpenReview.net, 2021. 1   
[8] Lu Chi, Borui Jiang, and Yadong Mu. Fast fourier convolution. In NeurIPS, 2020. 8   
[9] Rewon Child. Very deep vaes generalize autoregressive models and can outperform them on images. CoRR, abs/2011.10650, 2020. 3   
[10] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. CoRR, abs/1904.10509, 2019. 3   
[11] Bin Dai and David P. Wipf. Diagnosing and enhancing VAE models. In ICLR (Poster). OpenReview.net, 2019. 2, 3   
[12] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Fei-Fei Li. Imagenet: A large-scale hierarchical image database. In CVPR, pages 248–255. IEEE Computer Society, 2009. 1, 5, 7, 22   
[13] Emily Denton. Ethical considerations of generative ai. AI for Content Creation Workshop, CVPR, 2021. 9   
[14] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. CoRR, abs/1810.04805, 2018. 7   
[15] Prafulla Dhariwal and Alex Nichol. Diffusion models beat gans on image synthesis. CoRR, abs/2105.05233, 2021. 1, 2, 3, 4, 6, 7, 8, 18, 22, 25, 26, 28 [16] Sander Dieleman. Musings on typicality, 2020. 1, 3 [17] Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, and Jie Tang. Cogview: Mastering text-toimage generation via transformers. CoRR, abs/2105.13290,   
2021. 6, 7 [18] Laurent Dinh, David Krueger, and Yoshua Bengio. Nice: Non-linear independent components estimation, 2015. 3 [19] Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using real NVP. In 5th International Conference on Learning Representations, ICLR   
2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. OpenReview.net, 2017. 1, 3 [20] Alexey Dosovitskiy and Thomas Brox. Generating images with perceptual similarity metrics based on deep networks. In Daniel D. Lee, Masashi Sugiyama, Ulrike von Luxburg, Isabelle Guyon, and Roman Garnett, editors, Adv. Neural Inform. Process. Syst., pages 658–666, 2016. 3 [21] Patrick Esser, Robin Rombach, Andreas Blattmann, and Bjo¨rn Ommer. Imagebart: Bidirectional context with multinomial diffusion for autoregressive image synthesis. CoRR, abs/2108.08827, 2021. 6, 7, 22 [22] Patrick Esser, Robin Rombach, and Bjo¨rn Ommer. A note on data biases in generative models. arXiv preprint arXiv:2012.02516, 2020. 9 [23] Patrick Esser, Robin Rombach, and Bjo¨rn Ommer. Taming transformers for high-resolution image synthesis. CoRR, abs/2012.09841, 2020. 2, 3, 4, 6, 7, 21, 22, 29, 34, 36 [24] Mary Anne Franks and Ari Ezra Waldman. Sex, lies, and videotape: Deep fakes and free speech delusions. Md. L. Rev., 78:892, 2018. 9 [25] Kevin Frans, Lisa B. Soros, and Olaf Witkowski. Clipdraw: Exploring text-to-drawing synthesis through languageimage encoders. ArXiv, abs/2106.14843, 2021. 3 [26] Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, and Yaniv Taigman. Make-a-scene: Scenebased text-to-image generation with human priors. CoRR, abs/2203.13131, 2022. 6, 7, 16 [27] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron C. Courville, and Yoshua Bengio. Generative adversarial networks. CoRR, 2014. 1, 2 [28] Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron Courville. Improved training of wasserstein gans, 2017. 3 [29] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Adv. Neural Inform. Process. Syst., pages 6626–   
6637, 2017. 1, 5, 26 [30] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 1, 2, 3, 4,   
6, 17 [31] Jonathan Ho, Chitwan Saharia, William Chan, David J. Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. CoRR, os/2106.15282.2021 [32] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021. 6, 7, 16, 22,   
28, 37, 38 [33] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros. Image-to-image translation with conditional adversarial networks. In CVPR, pages 5967–5976. IEEE Computer Society, 2017. 3, 4 [34] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros. Image-to-image translation with conditional adversarial networks. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 5967–5976,   
2017. 4 [35] Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock, Evan Shelhamer, Olivier J. He´naff, Matthew M. Botvinick, Andrew Zisserman, Oriol Vinyals, and Joa˜o Carreira. Perceiver IO: A general architecture for structured inputs &outputs. CoRR, abs/2107.14795, 2021. 4 [36] Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Jo˜ao Carreira. Perceiver: General perception with iterative attention. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July   
2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pages 4651–4664. PMLR, 2021. 4, 5 [37] Manuel Jahn, Robin Rombach, and Bj¨orn Ommer. Highresolution complex scene synthesis with transformers. CoRR, abs/2105.06458, 2021. 20, 22, 27 [38] Niharika Jain, Alberto Olmo, Sailik Sengupta, Lydia Manikonda, and Subbarao Kambhampati. Imperfect imaganation: Implications of gans exacerbating biases on facial data augmentation and snapchat selfie lenses. arXiv preprint arXiv:2001.09528, 2020. 9 [39] Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of gans for improved quality, stability, and variation. CoRR, abs/1710.10196, 2017. 5, 6 [40] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In IEEE Conf. Comput. Vis. Pattern Recog., pages 4401–   
4410, 2019. 1 [41] T. Karras, S. Laine, and T. Aila. A style-based generator architecture for generative adversarial networks. In   
2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 5, 6 [42] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. CoRR, abs/1912.04958,   
2019. 2, 6, 28 [43] Dongjun Kim, Seungjae Shin, Kyungwoo Song, Wanmo Kang, and Il-Chul Moon. Score matching model for unbounded data score. CoRR, abs/2106.05527, 2021. 6 [44] Durk P Kingma and Prafulla Dhariwal. Glow: Generative flow with invertible 1x1 convolutions. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems, 2018. 3   
[45] Diederik P. Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. CoRR, abs/2107.00630, 2021. 1, 3, 16   
[46] Diederik P. Kingma and Max Welling. Auto-Encoding Variational Bayes. In 2nd International Conference on Learning Representations, ICLR, 2014. 1, 3, 4, 29   
[47] Zhifeng Kong and Wei Ping. On fast sampling of diffusion probabilistic models. CoRR, abs/2106.00132, 2021. 3   
[48] Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. Diffwave: A versatile diffusion model for audio synthesis. In ICLR. OpenReview.net, 2021. 1   
[49] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper R. R. Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Tom Duerig, and Vittorio Ferrari. The open images dataset V4: unified image classification, object detection, and visual relationship detection at scale. CoRR, abs/1811.00982, 2018. 7, 20, 22   
[50] Tuomas Kynka¨a¨nniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Improved precision and recall metric for assessing generative models. CoRR, abs/1904.06991, 2019. 5, 26   
[51] Tsung-Yi Lin, Michael Maire, Serge J. Belongie, Lubomir D. Bourdev, Ross B. Girshick, James Hays, Pietro Perona, Deva Ramanan, Piotr Dolla´r, and C. Lawrence Zitnick. Microsoft COCO: common objects in context. CoRR, abs/1405.0312, 2014. 6, 7, 27   
[52] Yuqing Ma, Xianglong Liu, Shihao Bai, Le-Yi Wang, Aishan Liu, Dacheng Tao, and Edwin Hancock. Region-wise generative adversarial imageinpainting for large missing areas. ArXiv, abs/1909.12507, 2019. 9   
[53] Chenlin Meng, Yang Song, Jiaming Song, Jiajun Wu, JunYan Zhu, and Stefano Ermon. Sdedit: Image synthesis and editing with stochastic differential equations. CoRR, abs/2108.01073, 2021. 1   
[54] Lars M. Mescheder. On the convergence properties of GAN training. CoRR, abs/1801.04406, 2018. 3   
[55] Luke Metz, Ben Poole, David Pfau, and Jascha SohlDickstein. Unrolled generative adversarial networks. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. OpenReview.net, 2017. 3   
[56] Mehdi Mirza and Simon Osindero. Conditional generative adversarial nets. CoRR, abs/1411.1784, 2014. 4   
[57] Gautam Mittal, Jesse H. Engel, Curtis Hawthorne, and Ian Simon. Symbolic music generation with diffusion models. CoRR, abs/2103.16091, 2021. 1   
[58] Kamyar Nazeri, Eric $\mathrm { N g }$ , Tony Joseph, Faisal Z. Qureshi, and Mehran Ebrahimi. Edgeconnect: Generative image inpainting with adversarial edge learning. ArXiv, abs/1901.00212, 2019. 9   
[59] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. GLIDE: towards photorealistic image generation and editing with text-guided diffusion models. CoRR, abs/2112.10741, 2021. 6, 7, 16   
[60] Anton Obukhov, Maximilian Seitzer, Po-Wei Wu, Semen Zhydenko, Jonathan Kyl, and Elvis Yu-Jing Lin. High-fidelity performance metrics for generative models in pytorch, 2020. Version: 0.3.0, DOI: 10.5281/zenodo.4957738. 26, 27   
[61] Taesung Park, Ming-Yu Liu, Ting-Chun Wang, and JunYan Zhu. Semantic image synthesis with spatially-adaptive normalization. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019. 4, 7   
[62] Taesung Park, Ming-Yu Liu, Ting-Chun Wang, and JunYan Zhu. Semantic image synthesis with spatially-adaptive normalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019. 22   
[63] Gaurav Parmar, Dacheng Li, Kwonjoon Lee, and Zhuowen Tu. Dual contradistinctive generative autoencoder. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2021, virtual, June 19-25, 2021, pages 823–832. Computer Vision Foundation / IEEE, 2021. 6   
[64] Gaurav Parmar, Richard Zhang, and Jun-Yan Zhu. On buggy resizing libraries and surprising subtleties in fid calculation. arXiv preprint arXiv:2104.11222, 2021. 26   
[65] David A. Patterson, Joseph Gonzalez, Quoc V. Le, Chen Liang, Lluis-Miquel Munguia, Daniel Rothchild, David R. So, Maud Texier, and Jeff Dean. Carbon emissions and large neural network training. CoRR, abs/2104.10350, 2021. 2   
[66] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. CoRR, abs/2102.12092, 2021. 1, 2, 3, 4, 7, 21, 27   
[67] Ali Razavi, A¨aron van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images with VQ-VAE-2. In NeurIPS, pages 14837–14847, 2019. 1, 2, 3, 22   
[68] Scott E. Reed, Zeynep Akata, Xinchen Yan, Lajanugen Logeswaran, Bernt Schiele, and Honglak Lee. Generative adversarial text to image synthesis. In ICML, 2016. 4   
[69] Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic backpropagation and approximate inference in deep generative models. In Proceedings of the 31st International Conference on International Conference on Machine Learning, ICML, 2014. 1, 4, 29   
[70] Robin Rombach, Patrick Esser, and Bjo¨rn Ommer. Network-to-network translation with conditional invertible neural networks. In NeurIPS, 2020. 3   
[71] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In MICCAI (3), volume 9351 of Lecture Notes in Computer Science, pages 234–241. Springer, 2015. 2, 3, 4   
[72] Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J. Fleet, and Mohammad Norouzi. Image super-resolution via iterative refinement. CoRR, abs/2104.07636, 2021. 1, 4, 8, 16, 22, 23, 27   
[73] Tim Salimans, Andrej Karpathy, Xi Chen, and Diederik P. Kingma. Pixelcnn $^ { + + }$ : Improving the pixelcnn with discretized logistic mixture likelihood and other modifications. CoRR, abs/1701.05517, 2017. 1, 3   
[74] Dave Salvator. NVIDIA Developer Blog. https: / / developer . nvidia . com / blog / getting - immediate-speedups-with-a100-tf32, 2020. 28   
[75] Robin San-Roman, Eliya Nachmani, and Lior Wolf. Noise estimation for generative diffusion models. CoRR, abs/2104.02600, 2021. 3   
[76] Axel Sauer, Kashyap Chitta, Jens Mu¨ller, and Andreas Geiger. Projected gans converge faster. CoRR, abs/2111.01007, 2021. 6   
[77] Edgar Scho¨nfeld, Bernt Schiele, and Anna Khoreva. A unet based discriminator for generative adversarial networks. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2020, Seattle, WA, USA, June 13-19, 2020, pages 8204–8213. Computer Vision Foundation / IEEE, 2020. 6   
[78] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion$4 0 0 \mathrm { m }$ : Open dataset of clip-filtered 400 million image-text pairs, 2021. 6, 7   
[79] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In Yoshua Bengio and Yann LeCun, editors, Int. Conf. Learn. Represent., 2015. 29, 43, 44, 45   
[80] Abhishek Sinha, Jiaming Song, Chenlin Meng, and Stefano Ermon. D2C: diffusion-denoising models for few-shot conditional generation. CoRR, abs/2106.06819, 2021. 3   
[81] Charlie Snell. Alien Dreams: An Emerging Art Scene. https : / / ml . berkeley . edu / blog / posts / clip-art/, 2021. [Online; accessed November-2021]. 2   
[82] Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. CoRR, abs/1503.03585, 2015. 1, 3, 4, 18   
[83] Kihyuk Sohn, Honglak Lee, and Xinchen Yan. Learning structured output representation using deep conditional generative models. In C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 28. Curran Associates, Inc., 2015. 4   
[84] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR. OpenReview.net, 2021. 3, 5, 6, 22   
[85] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Scorebased generative modeling through stochastic differential equations. CoRR, abs/2011.13456, 2020. 1, 3, 4, 18   
[86] Emma Strubell, Ananya Ganesh, and Andrew McCallum. Energy and policy considerations for modern deep learning research. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020, pages 13693–13696. AAAI Press, 2020. 2   
[87] Wei Sun and Tianfu Wu. Learning layout and style reconfigurable gans for controllable image synthesis. CoRR, abs/2003.11571, 2020. 22, 27   
[88] Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, and Victor S. Lempitsky. Resolution-robust large mask inpainting with fourier convolutions. ArXiv, abs/2109.07161, 2021. 8, 9, 26, 32   
[89] Tristan Sylvain, Pengchuan Zhang, Yoshua Bengio, R. Devon Hjelm, and Shikhar Sharma. Object-centric image generation from layouts. In Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021, Thirty-Third Conference on Innovative Applications of Artificial Intelligence, IAAI 2021, The Eleventh Symposium on Educational Advances in Artificial Intelligence, EAAI 2021, Virtual Event, February 2-9, 2021, pages 2647–2655. AAAI Press, 2021. 20, 22, 27   
[90] Patrick Tinsley, Adam Czajka, and Patrick Flynn. This face does not exist... but it might be yours! identity leakage in generative models. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1320–1328, 2021. 9   
[91] Antonio Torralba and Alexei A Efros. Unbiased look at dataset bias. In CVPR 2011, pages 1521–1528. IEEE, 2011. 9   
[92] Arash Vahdat and Jan Kautz. NVAE: A deep hierarchical variational autoencoder. In NeurIPS, 2020. 3   
[93] Arash Vahdat, Karsten Kreis, and Jan Kautz. Scorebased generative modeling in latent space. CoRR, abs/2106.05931, 2021. 2, 3, 5, 6   
[94] Aaron van den Oord, Nal Kalchbrenner, Lasse Espeholt, koray kavukcuoglu, Oriol Vinyals, and Alex Graves. Conditional image generation with pixelcnn decoders. In Advances in Neural Information Processing Systems, 2016. 3   
[95] Aa¨ron van den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. Pixel recurrent neural networks. CoRR, abs/1601.06759, 2016. 3   
[96] Aa¨ron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation learning. In NIPS, pages 6306–6315, 2017. 2, 4, 29   
[97] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NIPS, pages 5998–6008, 2017. 3, 4, 5, 7   
[98] Rivers Have Wings. Tweet on Classifier-free guidance for autoregressive models. https : / / twitter . com / RiversHaveWings / status / 1478093658716966912, 2022. 6   
[99] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Re´mi Louf, Morgan Funtowicz, and Jamie Brew. Huggingface’s transformers: State-of-the-art natural language processing. CoRR, abs/1910.03771, 2019. 26   
[100] Zhisheng Xiao, Karsten Kreis, Jan Kautz, and Arash Vahdat. VAEBM: A symbiosis between variational autoencoders and energy-based models. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. 6   
[101] Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation using VQ-VAE and transformers. CoRR, abs/2104.10157, 2021. 3   
[102] Fisher Yu, Yinda Zhang, Shuran Song, Ari Seff, and Jianxiong Xiao. LSUN: construction of a large-scale image dataset using deep learning with humans in the loop. CoRR, abs/1506.03365, 2015. 5   
[103] Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved vqgan, 2021. 3, 4   
[104] Jiahui Yu, Zhe L. Lin, Jimei Yang, Xiaohui Shen, Xin Lu, and Thomas S. Huang. Free-form image inpainting with gated convolution. 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pages 4470–4479, 2019. 9   
[105] K. Zhang, Jingyun Liang, Luc Van Gool, and Radu Timofte. Designing a practical degradation model for deep blind image super-resolution. ArXiv, abs/2103.14006, 2021. 23   
[106] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018. 3, 8, 19   
[107] Shengyu Zhao, Jianwei Cui, Yilun Sheng, Yue Dong, Xiao Liang, Eric I-Chao Chang, and Yan Xu. Large scale image completion via co-modulated generative adversarial networks. ArXiv, abs/2103.10428, 2021. 9   
[108] Bolei Zhou, A\` gata Lapedriza, Aditya Khosla, Aude Oliva, and Antonio Torralba. Places: A 10 million image database for scene recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40:1452–1464, 2018. 8, 9, 26   
[109] Yufan Zhou, Ruiyi Zhang, Changyou Chen, Chunyuan Li, Chris Tensmeyer, Tong Yu, Jiuxiang Gu, Jinhui Xu, and Tong Sun. LAFITE: towards language-free training for text-to-image generation. CoRR, abs/2111.13792, 2021. 6, 7, 16

# Appendix

![](images/a39ad6e19f76c5b225a135f0a1792d1edfd87cb2b3adc43ab987c672729630b9.jpg)  
Figure 12. Convolutional samples from the semantic landscapes model as in Sec. 4.3.2, finetuned on $5 1 2 ^ { 2 }$ images.

![](images/58e187575e0f0240a2033e65462967d17baf3d698e37b475ef68ea40070266c6.jpg)

![](images/344fe389e26d4e78f1c2610eb001da8d0d3734b4e4625a0b32da8146aa1a46fd.jpg)  
’A sunset over a mountain range, vector image.’   
Figure 13. Combining classifier free diffusion guidance with the convolutional sampling strategy from Sec. 4.3.2, our 1.45B paramete text-to-image model can be used for rendering images larger than the native $2 5 6 ^ { 2 }$ resolution the model was trained on.

# A. Changelog

Here we list changes between this version (https://arxiv.org/abs/2112.10752v2) of the paper and the previous version, i.e. https://arxiv.org/abs/2112.10752v1.

• We updated the results on text-to-image synthesis in Sec. 4.3 which were obtained by training a new, larger model (1.45B parameters). This also includes a new comparison to very recent competing methods on this task that were published on arXiv at the same time as ( [59, 109]) or after ( [26]) the publication of our work.   
• We updated results on class-conditional synthesis on ImageNet in Sec. 4.1, Tab. 3 (see also Sec. D.4) obtained by retraining the model with a larger batch size. The corresponding qualitative results in Fig. 26 and Fig. 27 were also updated. Both the updated text-to-image and the class-conditional model now use classifier-free guidance [32] as a measure to increase visual fidelity.   
• We conducted a user study (following the scheme suggested by Saharia et al [72]) which provides additional evaluation for our inpainting (Sec. 4.5) and superresolution models (Sec. 4.4).   
• Added Fig. 5 to the main paper, moved Fig. 18 to the appendix, added Fig. 13 to the appendix.

# B. Detailed Information on Denoising Diffusion Models

Diffusion models can be specified in terms of a signal-to-noise ratio $\begin{array} { r } { \mathrm { S N R } ( t ) = \frac { \alpha _ { t } ^ { 2 } } { \sigma _ { t } ^ { 2 } } } \end{array}$ consisting of sequences $( \alpha _ { t } ) _ { t = 1 } ^ { T }$ and $( \sigma _ { t } ) _ { t = 1 } ^ { T }$ which, starting from a data sample $x _ { 0 }$ , define a forward diffusion process $q$ as

$$
q ( x _ { t } | x _ { 0 } ) = \mathcal { N } ( x _ { t } | \alpha _ { t } x _ { 0 } , \sigma _ { t } ^ { 2 } \mathbb { I } )
$$

with the Markov structure for $s < t$ :

$$
\begin{array} { r l r } {  { q ( \boldsymbol { x } _ { t } | \boldsymbol { x } _ { s } ) = \mathcal { N } ( \boldsymbol { x } _ { t } | \alpha _ { t | s } \boldsymbol { x } _ { s } , \sigma _ { t | s } ^ { 2 } \mathbb { I } ) } } \\ & { } & { \alpha _ { t | s } = \frac { \alpha _ { t } } { \alpha _ { s } } } \\ & { } & { \sigma _ { t | s } ^ { 2 } = \sigma _ { t } ^ { 2 } - \alpha _ { t | s } ^ { 2 } \sigma _ { s } ^ { 2 } } \end{array}
$$

Denoising diffusion models are generative models $p ( x _ { 0 } )$ which revert this process with a similar Markov structure running backward in time, i.e. they are specified as

$$
p ( x _ { 0 } ) = \int _ { z } p ( x _ { T } ) \prod _ { t = 1 } ^ { T } p ( x _ { t - 1 } | x _ { t } )
$$

The evidence lower bound (ELBO) associated with this model then decomposes over the discrete time steps as

$$
- \log p ( x _ { 0 } ) \leq \mathbb { K L } ( q ( x _ { T } | x _ { 0 } ) | p ( x _ { T } ) ) + \sum _ { t = 1 } ^ { T } \mathbb { E } _ { q ( x _ { t } | x _ { 0 } ) } \mathbb { K L } ( q ( x _ { t - 1 } | x _ { t } , x _ { 0 } ) | p ( x _ { t - 1 } | x _ { t } ) )
$$

The prior $p ( x _ { T } )$ is typically choosen as a standard normal distribution and the first term of the ELBO then depends only on the final signal-to-noise ratio $\operatorname { S N R } ( T )$ . To minimize the remaining terms, a common choice to parameterize $p ( x _ { t - 1 } | x _ { t } )$ is to specify it in terms of the true posterior $q ( x _ { t - 1 } | x _ { t } , x _ { 0 } )$ but with the unknown $x _ { 0 }$ replaced by an estimate $x _ { \theta } ( x _ { t } , t )$ based on the current step $\boldsymbol { x } _ { t }$ . This gives [45]

$$
\begin{array} { r l } & { p ( x _ { t - 1 } | x _ { t } ) : = q ( x _ { t - 1 } | x _ { t } , x _ { \theta } ( x _ { t } , t ) ) } \\ & { \qquad = \mathcal { N } ( x _ { t - 1 } | \mu _ { \theta } ( x _ { t } , t ) , \sigma _ { t | t - 1 } ^ { 2 } \frac { \sigma _ { t - 1 } ^ { 2 } } { \sigma _ { t } ^ { 2 } } \mathbb { I } ) , } \end{array}
$$

where the mean can be expressed as

$$
\mu _ { \theta } ( x _ { t } , t ) = \frac { \alpha _ { t | t - 1 } \sigma _ { t - 1 } ^ { 2 } } { \sigma _ { t } ^ { 2 } } x _ { t } + \frac { \alpha _ { t - 1 } \sigma _ { t | t - 1 } ^ { 2 } } { \sigma _ { t } ^ { 2 } } x _ { \theta } ( x _ { t } , t ) .
$$

In this case, the sum of the ELBO simplify to

$$
\sum _ { i = 1 } ^ { T } \mathbb { E } _ { q ( x _ { t } | x _ { 0 } ) } \mathbb { K } \mathbb { L } ( q ( x _ { t - 1 } | x _ { t } , x _ { 0 } ) | p ( x _ { t - 1 } ) = \sum _ { t = 1 } ^ { T } \mathbb { E } _ { N ( \epsilon | 0 , 1 ) } \frac { 1 } { 2 } \big ( \mathrm { S N R } ( t - 1 ) - \mathrm { S N R } ( t ) \big ) \| x _ { 0 } - x _ { \theta } ( \alpha _ { t } x _ { 0 } + \epsilon ) \big )
$$

Following [30], we use the reparameterization

$$
\epsilon _ { \theta } ( x _ { t } , t ) = ( x _ { t } - \alpha _ { t } x _ { \theta } ( x _ { t } , t ) ) / \sigma _ { t }
$$

to express the reconstruction term as a denoising objective,

$$
\| x _ { 0 } - x _ { \theta } \big ( \alpha _ { t } x _ { 0 } + \sigma _ { t } \epsilon , t \big ) \| ^ { 2 } = \frac { \sigma _ { t } ^ { 2 } } { \alpha _ { t } ^ { 2 } } \| \epsilon - \epsilon _ { \theta } \big ( \alpha _ { t } x _ { 0 } + \sigma _ { t } \epsilon , t \big ) \| ^ { 2 }
$$

and the reweighting, which assigns each of the terms the same weight and results in Eq. (1).

![](images/8b686e11da4c844847752b213a934a5ff1980008a90fbe636f572a679f7ec93e.jpg)  
Figure 14. On landscapes, convolutional sampling with unconditional models can lead to homogeneous and incoherent global structure (see column 2). $L _ { 2 }$ -guiding with a low resolution image can help to reestablish coherent global structures.

An intriguing feature of diffusion models is that unconditional models can be conditioned at test-time [15, 82, 85]. In particular, [15] presented an algorithm to guide both unconditional and conditional models trained on the ImageNet dataset with a classifier $\log p _ { \Phi } ( y | x _ { t } )$ , trained on each $x _ { t }$ of the diffusion process. We directly build on this formulation and introduce post-hoc image-guiding:

For an epsilon-parameterized model with fixed variance, the guiding algorithm as introduced in [15] reads:

$$
\hat { \epsilon }  \epsilon _ { \theta } ( z _ { t } , t ) + \sqrt { 1 - \alpha _ { t } ^ { 2 } } \nabla _ { z _ { t } } \log p _ { \Phi } ( y | z _ { t } ) .
$$

This can be interpreted as an update correcting the “score” $\epsilon _ { \theta }$ with a conditional distribution $\log p _ { \Phi } ( \boldsymbol { y } | \boldsymbol { z } _ { t } )$ .

So far, this scenario has only been applied to single-class classification models. We re-interpret the guiding distribution $p _ { \Phi } ( y | T ( \mathcal { D } ( z _ { 0 } ( z _ { t } ) ) ) )$ as a general purpose image-to-image translation task given a target image $y$ , where $T$ can be any differentiable transformation adopted to the image-to-image translation task at hand, such as the identity, a downsampling operation or similar.

As an example, we can assume a Gaussian guider with fixed variance $\sigma ^ { 2 } = 1$ , such that

$$
\log p _ { \Phi } ( y | z _ { t } ) = - \frac { 1 } { 2 } \| y - T ( \mathcal { D } ( z _ { 0 } ( z _ { t } ) ) ) \| _ { 2 } ^ { 2 }
$$

becomes a $L _ { 2 }$ regression objective.

Fig. 14 demonstrates how this formulation can serve as an upsampling mechanism of an unconditional model trained on $2 5 6 ^ { 2 }$ images, where unconditional samples of size $2 5 6 ^ { 2 }$ guide the convolutional synthesis of $5 1 2 ^ { 2 }$ images and $T$ is a $2 \times$ bicubic downsampling. Following this motivation, we also experiment with a perceptual similarity guiding and replace the $L _ { 2 }$ objective with the LPIPS [106] metric, see Sec. 4.4.

# D. Additional Results

# D.1. Choosing the Signal-to-Noise Ratio for High-Resolution Synthesis

![](images/a0621e486c4358f1bd1db86e4326c0605a3321a34ddc3ec5d92ce6810164445a.jpg)  
Figure 15. Illustrating the effect of latent space rescaling on convolutional sampling, here for semantic image synthesis on landscapes. See Sec. 4.3.2 and Sec. D.1.

As discussed in Sec. 4.3.2, the signal-to-noise ratio induced by the variance of the latent space ( $\therefore e . \mathrm { V a r ( z ) } / \sigma _ { t } ^ { 2 } )$ significantly affects the results for convolutional sampling. For example, when training a LDM directly in the latent space of a KLregularized model (see Tab. 8), this ratio is very high, such that the model allocates a lot of semantic detail early on in the reverse denoising process. In contrast, when rescaling the latent space by the component-wise standard deviation of the latents as described in Sec. G, the SNR is descreased. We illustrate the effect on convolutional sampling for semantic image synthesis in Fig. 15. Note that the VQ-regularized space has a variance close to 1, such that it does not have to be rescaled.

# D.2. Full List of all First Stage Models

We provide a complete list of various autoenconding models trained on the OpenImages dataset in Tab. 8.

# D.3. Layout-to-Image Synthesis

Here we provide the quantitative evaluation and additional samples for our layout-to-image models from Sec. 4.3.1. We train a model on the COCO [4] and one on the OpenImages [49] dataset, which we subsequently additionally finetune on COCO. Tab 9 shows the result. Our COCO model reaches the performance of recent state-of-the art models in layout-toimage synthesis, when following their training and evaluation protocol [89]. When finetuning from the OpenImages model, we surpass these works. Our OpenImages model surpasses the results of Jahn et al [37] by a margin of nearly 11 in terms of FID. In Fig. 16 we show additional samples of the model finetuned on COCO.

# D.4. Class-Conditional Image Synthesis on ImageNet

Tab. 10 contains the results for our class-conditional LDM measured in FID and Inception score (IS). LDM-8 requires significantly fewer parameters and compute requirements (see Tab. 18) to achieve very competitive performance. Similar to previous work, we can further boost the performance by training a classifier on each noise scale and guiding with it,

<html><body><table><tr><td>f</td><td>|2</td><td>C</td><td>R-FID↓</td><td>R-IS ↑</td><td>PSNR↑</td><td>PSIM↓</td><td>SSIM ↑</td></tr><tr><td>16 VQGAN [23]</td><td>16384</td><td>256</td><td>4.98</td><td></td><td>19.9 ±3.4</td><td>1.83 ±0.42</td><td>0.51 ±0.18</td></tr><tr><td>16 VQGAN [23]</td><td>1024</td><td>256</td><td>7.94</td><td></td><td>19.4 ±3.3</td><td>1.98 ±0.43</td><td>0.50 ±0.18</td></tr><tr><td>8 DALL-E [66]</td><td>8192</td><td></td><td>32.01</td><td>1</td><td>22.8 ±2.1</td><td>1.95 ±0.51</td><td>0.73 ±0.13</td></tr><tr><td>32</td><td>16384</td><td>16</td><td>31.83</td><td>40.40 ±1.07</td><td>17.45 ±2.90</td><td>2.58 ±0.48</td><td>0.41 ±0.18</td></tr><tr><td>16</td><td>16384</td><td>8</td><td>5.15</td><td>144.55 ±3.74</td><td>20.83 ±3.61</td><td>1.73 ±0.43</td><td>0.54 ±0.18</td></tr><tr><td>8</td><td>16384</td><td>4</td><td>1.14</td><td>201.92 ±3.97</td><td>23.07 ±3.99</td><td>1.17 ±0.36</td><td>0.65 ±0.16</td></tr><tr><td>8</td><td>256</td><td>4</td><td>1.49</td><td>194.20 ±3.87</td><td>22.35 ±3.81</td><td>1.26 ±0.37</td><td>0.62 ±0.16</td></tr><tr><td>4</td><td>8192</td><td>3</td><td>0.58</td><td>224.78 ±5.35</td><td>27.43 ±4.26</td><td>0.53 ±0.21</td><td>0.82 ±0.10</td></tr><tr><td>4†</td><td>8192</td><td>3</td><td>1.06</td><td>221.94 ±4.58</td><td>25.21 ±4.17</td><td>0.72 ±0.26</td><td>0.76 ±0.12</td></tr><tr><td>4</td><td>256</td><td>3</td><td>0.47</td><td>223.81 ±4.58</td><td>26.43 ±4.22</td><td>0.62 ±0.24</td><td>0.80 ±0.11</td></tr><tr><td>2</td><td>2048</td><td>2</td><td>0.16</td><td>232.75 ±5.09</td><td>30.85 ±4.12</td><td>0.27 ±0.12</td><td>0.91 ±0.05</td></tr><tr><td>2</td><td>64</td><td>2</td><td>0.40</td><td>226.62 ±4.83</td><td>29.13 ±3.46</td><td>0.38 ±0.13</td><td>0.90 ±0.05</td></tr><tr><td>32</td><td>KL</td><td>64</td><td>2.04</td><td>189.53 ±3.68</td><td>22.27 ±3.93</td><td>1.41 ±0.40</td><td>0.61 ±0.17</td></tr><tr><td>32</td><td>KL</td><td>16</td><td>7.3</td><td>132.75 ±2.71</td><td>20.38 ±3.56</td><td>1.88 ±0.45</td><td>0.53 ±0.18</td></tr><tr><td>16</td><td>KL</td><td>16</td><td>0.87</td><td>210.31 ±3.97</td><td>24.08 ±4.22</td><td>1.07 ±0.36</td><td>0.68 ±0.15</td></tr><tr><td>16</td><td>KL</td><td>8</td><td>2.63</td><td>178.68 ±4.08</td><td>21.94 ±3.92</td><td>1.49 ±0.42</td><td>0.59 ±0.17</td></tr><tr><td>8</td><td>KL</td><td>4</td><td>0.90</td><td>209.90 ±4.92</td><td>24.19 ±4.19</td><td>1.02 ±0.35</td><td>0.69 ±0.15</td></tr><tr><td>4</td><td>KL</td><td>3</td><td>0.27</td><td>227.57 ±4.89</td><td>27.53 ±4.54</td><td>0.55 ±0.24</td><td>0.82 ±0.11</td></tr><tr><td>2</td><td>KL</td><td>2</td><td>0.086</td><td>232.66 ±5.16</td><td>32.47 ±4.19</td><td>0.20 ±0.09</td><td>0.93 ±0.04</td></tr></table></body></html>

Table 8. Complete autoencoder zoo trained on OpenImages, evaluated on ImageNet-Val. $\dagger$ denotes an attention-free autoencoder.

![](images/d924c8e874ac0df122178241aaadfb9d8ae7957d8f521a6876d97b8a7e0de074.jpg)  
Figure 16. More samples from our best model for layout-to-image synthesis, LDM-4, which was trained on the OpenImages dataset and finetuned on the COCO dataset. Samples generated with 100 DDIM steps and $\eta = 0$ . Layouts are from the COCO validation set.

see Sec. C. Unlike the pixel-based methods, this classifier is trained very cheaply in latent space. For additional qualitative results, see Fig. 26 and Fig. 27.

Table 9. Quantitative comparison of our layout-to-image models on the COCO [4] and OpenImages [49] datasets. †: Training from scratch on COCO; ∗: Finetuning from OpenImages.   

<html><body><table><tr><td rowspan="2">Method</td><td>COCO256 × 256</td><td>OpenImages 256 × 256</td><td>OpenImages 512 × 512</td></tr><tr><td>FID↓</td><td>FID↓</td><td>FID↓</td></tr><tr><td>LostGAN-V2 [87]</td><td>42.55</td><td>-</td><td>-</td></tr><tr><td>OC-GAN [89]</td><td>41.65</td><td>-</td><td></td></tr><tr><td>SPADE [62]</td><td>41.11</td><td></td><td>-</td></tr><tr><td>VQGAN+T[37]</td><td>56.58</td><td>45.33</td><td>48.11</td></tr><tr><td>LDM-8 (100 steps,ours)</td><td>42.06†</td><td></td><td>■</td></tr><tr><td>LDM-4 (200 steps,ours)</td><td>40.91*</td><td>32.02</td><td>35.80</td></tr></table></body></html>

<html><body><table><tr><td>Method</td><td>FID↓</td><td>IS↑</td><td>Precision↑</td><td>Recall个</td><td>Nparams</td><td></td></tr><tr><td>SR3 [72]</td><td>11.30</td><td></td><td></td><td></td><td>625M</td><td></td></tr><tr><td>ImageBART [21]</td><td>21.19</td><td></td><td></td><td></td><td>3.5B</td><td>--</td></tr><tr><td>ImageBART[21]</td><td>7.44</td><td></td><td></td><td></td><td>3.5B</td><td>0.05 acc.rate*</td></tr><tr><td>VQGAN+T [23]</td><td>17.04</td><td>70.6±1.8</td><td></td><td></td><td>1.3B</td><td></td></tr><tr><td>VQGAN+T [23]</td><td>5.88</td><td>304.8±3.6</td><td>-</td><td>-</td><td>1.3B</td><td>0.05 acc.rate*</td></tr><tr><td>BigGan-deep [3]</td><td>6.95</td><td>203.6±2.6</td><td>0.87</td><td>0.28</td><td>340M</td><td></td></tr><tr><td>ADM [15]</td><td>10.94</td><td>100.98</td><td>0.69</td><td>0.63</td><td>554M</td><td>250 DDIM steps</td></tr><tr><td>ADM-G [15]</td><td>4.59</td><td>186.7</td><td>0.82</td><td>0.52</td><td>608M</td><td>250 DDIM steps</td></tr><tr><td>ADM-G,ADM-U [15]</td><td>3.85</td><td>221.72</td><td>0.84</td><td>0.53</td><td>n/a</td><td>2 × 250 DDIM steps</td></tr><tr><td>CDM [31]</td><td>4.88</td><td>158.71±2.26</td><td>-</td><td>-</td><td>n/a</td><td>2 × 100 DDIM steps</td></tr><tr><td>LDM-8 (ours)</td><td>17.41</td><td>72.92 ±2.6</td><td>0.65</td><td>0.62</td><td>395M</td><td>200 DDIM steps,2.9M train steps,batch size 64</td></tr><tr><td>LDM-8-G (ours)</td><td>8.11</td><td>190.43 ±2.60</td><td>0.83</td><td>0.36</td><td>506M</td><td>200 DDIM steps,classifier scale 10,2.9M train steps, batch size 64</td></tr><tr><td>LDM-8 (ours)</td><td>15.51</td><td>79.03 ±1.03</td><td>0.65</td><td>0.63</td><td>395M</td><td>200 DDIMsteps,4.8M train steps,batch size 64</td></tr><tr><td>LDM-8-G (ours)</td><td>7.76</td><td>209.52±4.24</td><td>0.84</td><td>0.35</td><td>506M</td><td>200 DDIM steps,classifier scale 10,4.8Mtrain steps,batch size 64</td></tr><tr><td>LDM-4 (ours)</td><td>10.56</td><td>103.49±1.24</td><td>0.71</td><td>0.62</td><td>400M</td><td>250 DDIM steps,178K train steps,batch size 1200</td></tr><tr><td>LDM-4-G (ours)</td><td>3.95</td><td>178.22±2.43</td><td>0.81</td><td>0.55</td><td>400M</td><td>250 DDIM steps,unconditional guidance [32] scale 1.25,178K train steps,batch size 1200</td></tr><tr><td>LDM-4-G (ours)</td><td>3.60</td><td>247.67±5.59</td><td>0.87</td><td>0.48</td><td>400M</td><td>250 DDIM steps,unconditional guidance [32] scale 1.5,178K train steps,batch size 1200</td></tr></table></body></html>

Table 10. Comparison of a class-conditional ImageNet $L D M$ with recent state-of-the-art methods for class-conditional image generation on the ImageNet [12] dataset.∗: Classifier rejection sampling with the given rejection rate as proposed in [67].

# D.5. Sample Quality vs. V100 Days (Continued from Sec. 4.1)

![](images/3dafb59d482473fc0ce4fc47ac8e9adb87c20829e8fe61be88ab4483564056e7.jpg)  
Figure 17. For completeness we also report the training progress of class-conditional LDMs on the ImageNet dataset for a fixed number of $3 5 \mathrm { V } 1 0 0 \$ days. Results obtained with 100 DDIM steps [84] and $\kappa = 0$ . FIDs computed on 5000 samples for efficiency reasons.

For the assessment of sample quality over the training progress in Sec. 4.1, we reported FID and IS scores as a function of train steps. Another possibility is to report these metrics over the used resources in V100 days. Such an analysis is additionally provided in Fig. 17, showing qualitatively similar results.

<html><body><table><tr><td>Method</td><td>FID↓</td><td>IS↑</td><td>PSNR ↑</td><td>SSIM ↑</td></tr><tr><td>Image Regression [72]</td><td>15.2</td><td>121.1</td><td>27.9</td><td>0.801</td></tr><tr><td>SR3 [72]</td><td>5.2</td><td>180.1</td><td>26.4</td><td>0.762</td></tr><tr><td>LDM-4 (ours,100 steps)</td><td>2.8†/4.8t</td><td>166.3</td><td>24.4±3.8</td><td>0.69±0.14</td></tr><tr><td>LDM-4 (ours,50 steps,guiding)</td><td>4.4†/6.4‡</td><td>153.7</td><td>25.8±3.7</td><td>0.74±0.12</td></tr><tr><td>LDM-4 (ours,100 steps,guiding)</td><td>4.4†/6.4‡</td><td>154.1</td><td>25.7±3.7</td><td>0.73±0.12</td></tr><tr><td>LDM-4 (ours,100 steps,+15 ep.)</td><td>2.6† /4.6‡</td><td>169.76±5.03</td><td>24.4±3.8</td><td>0.69±0.14</td></tr><tr><td>Pixel-DM(100 steps,+15 ep.)</td><td>5.1†/7.1‡</td><td>163.06±4.67</td><td>24.1±3.3</td><td>0.59±0.12</td></tr></table></body></html>

# D.6. Super-Resolution

For better comparability between LDMs and diffusion models in pixel space, we extend our analysis from Tab. 5 by comparing a diffusion model trained for the same number of steps and with a comparable number 1 of parameters to our LDM. The results of this comparison are shown in the last two rows of Tab. 11 and demonstrate that LDM achieves better performance while allowing for significantly faster sampling. A qualitative comparison is given in Fig. 20 which shows random samples from both LDM and the diffusion model in pixel space.

D.6.1 LDM-BSR: General Purpose SR Model via Diverse Image Degradation   
Figure 18. LDM-BSR generalizes to arbitrary inputs and can be used as a general-purpose upsampler, upscaling samples from a classconditional LDM (image cf . Fig. 4) to $1 0 2 4 ^ { 2 }$ resolution. In contrast, using a fixed degradation process (see Sec. 4.4) hinders generalization.   
![](images/8831f15620de286781bd80b59f0294724dc4a806840b6408539b0cbcff7d238b.jpg)  
Table 11. $\times 4$ upscaling results on ImageNet-Val. $( 2 5 6 ^ { 2 } )$ ; †: FID features computed on validation split, ‡: FID features computed on train split. We also include a pixel-space baseline that receives the same amount of compute as LDM-4. The last two rows received 15 epochs of additional training compared to the former results.

To evaluate generalization of our LDM-SR, we apply it both on synthetic LDM samples from a class-conditional ImageNet model (Sec. 4.1) and images crawled from the internet. Interestingly, we observe that LDM-SR, trained only with a bicubicly downsampled conditioning as in [72], does not generalize well to images which do not follow this pre-processing. Hence, to obtain a superresolution model for a wide range of real world images, which can contain complex superpositions of camera noise, compression artifacts, blurr and interpolations, we replace the bicubic downsampling operation in LDM-SR with the degration pipeline from [105]. The BSR-degradation process is a degradation pipline which applies JPEG compressions noise, camera sensor noise, different image interpolations for downsampling, Gaussian blur kernels and Gaussian noise in a random order to an image. We found that using the bsr-degredation process with the original parameters as in [105] leads to a very strong degradation process. Since a more moderate degradation process seemed apppropiate for our application, we adapted the parameters of the bsr-degradation (our adapted degradation process can be found in our code base at https: //github.com/CompVis/latent-diffusion). Fig. 18 illustrates the effectiveness of this approach by directly comparing LDM-SR with LDM-BSR. The latter produces images much sharper than the models confined to a fixed preprocessing, making it suitable for real-world applications. Further results of LDM-BSR are shown on LSUN-cows in Fig. 19.

# E. Implementation Details and Hyperparameters

# E.1. Hyperparameters

We provide an overview of the hyperparameters of all trained LDM models in Tab. 12, Tab. 13, Tab. 14 and Tab. 15.   

<html><body><table><tr><td></td><td>CelebA-HQ 256× 256</td><td>FFHQ256× 256</td><td>LSUN-Churches 256 × 256</td><td>LSUN-Bedrooms 256 × 256</td></tr><tr><td>f</td><td>4</td><td>4</td><td>8</td><td>4</td></tr><tr><td>z-shape</td><td>64×64×3</td><td>64×64×3</td><td>-</td><td>64×64×3</td></tr><tr><td>|</td><td>8192</td><td>8192</td><td>1</td><td>8192</td></tr><tr><td>Diffusion steps</td><td>1000</td><td>1000</td><td>1000</td><td>1000</td></tr><tr><td>Noise Schedule</td><td>linear</td><td>linear</td><td>linear</td><td>linear</td></tr><tr><td>Nparams</td><td>274M</td><td>274M</td><td>294M</td><td>274M</td></tr><tr><td>Channels</td><td>224</td><td>224</td><td>192</td><td>224</td></tr><tr><td>Depth</td><td>2</td><td>2</td><td>2</td><td>2</td></tr><tr><td>Channel Multiplier</td><td>1,2,3,4</td><td>1,2,3,4</td><td>1,2,2,4,4</td><td>1,2,3,4</td></tr><tr><td>Attention resolutions</td><td>32,16,8</td><td>32,16,8</td><td>32,16,8,4</td><td>32,16,8</td></tr><tr><td>Head Channels</td><td>32</td><td>32</td><td>24</td><td>32</td></tr><tr><td>Batch Size</td><td>48</td><td>42</td><td>96</td><td>48</td></tr><tr><td>Iterations*</td><td>410k</td><td>635k</td><td>500k</td><td>1.9M</td></tr><tr><td>Learning Rate</td><td>9.6e-5</td><td>8.4e-5</td><td>5.e-5</td><td>9.6e-5</td></tr></table></body></html>

Table 12. Hyperparameters for the unconditional LDMs producing the numbers shown in Tab. 1. All models trained on a single NVIDIA A100.   

<html><body><table><tr><td></td><td>LDM-1</td><td>LDM-2</td><td>LDM-4</td><td>LDM-8</td><td>LDM-16</td><td>LDM-32</td></tr><tr><td>z-shape</td><td>256×256×3</td><td>128×128×2</td><td>64×64×3</td><td>32× 32 ×4</td><td>16×16×8</td><td>88×8×32</td></tr><tr><td></td><td></td><td>2048</td><td>8192</td><td>16384</td><td>16384</td><td>16384</td></tr><tr><td>Diffusion steps</td><td>1000</td><td>1000</td><td>1000</td><td>1000</td><td>1000</td><td>1000</td></tr><tr><td>Noise Schedule</td><td>linear</td><td>linear</td><td>linear</td><td>linear</td><td>linear</td><td>linear</td></tr><tr><td>Model Size</td><td>396M</td><td>391M</td><td>391M</td><td>395M</td><td>395M</td><td>395M</td></tr><tr><td>Channels</td><td>192</td><td>192</td><td>192</td><td>256</td><td>256</td><td>256</td></tr><tr><td>Depth</td><td>2</td><td>2</td><td>2</td><td>2</td><td>2</td><td>2</td></tr><tr><td>Channel Multiplier</td><td>1,1,2,2,4,4</td><td>1,2,2,4,4</td><td>1,2,3,5</td><td>1,2,4</td><td>1,2.4</td><td>1,2.4</td></tr><tr><td>NumberofHeads</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Batch Size</td><td>7</td><td>9</td><td>40</td><td>64</td><td>112</td><td>112</td></tr><tr><td>Iterations</td><td>2M</td><td>2M</td><td>2M</td><td>2M</td><td>2M</td><td>2M</td></tr><tr><td>Learning Rate</td><td>4.9e-5</td><td>6.3e-5</td><td>8e-5</td><td>6.4e-5</td><td>4.5e-5</td><td>4.5e-5</td></tr><tr><td>Conditioning</td><td>CA</td><td>CA</td><td>CA</td><td>CA</td><td>CA</td><td>CA</td></tr><tr><td>CA-resolutions</td><td>32,16,8</td><td>32,16,8</td><td>32,16,8</td><td>32,16,8</td><td>16,8,4</td><td>8,4,2</td></tr><tr><td>Embedding Dimension</td><td>512</td><td>512</td><td>512</td><td>512</td><td>512</td><td>512</td></tr><tr><td>Transformers Depth</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td></tr></table></body></html>

Table 13. Hyperparameters for the conditional $L D M s$ trained on the ImageNet dataset for the analysis in Sec. 4.1. All models trained on a single NVIDIA A100.

# E.2. Implementation Details

# E.2.1 Implementations of $\tau _ { \theta }$ for conditional LDMs

For the experiments on text-to-image and layout-to-image (Sec. 4.3.1) synthesis, we implement the conditioner $\tau _ { \theta }$ as an unmasked transformer which processes a tokenized version of the input $y$ and produces an output $\zeta : = \tau _ { \theta } ( y )$ , where $\zeta \in$ $\mathbb { R } ^ { M \times d _ { \tau } }$ . More specifically, the transformer is implemented from $N$ transformer blocks consisting of global self-attention layers, layer-normalization and position-wise MLPs as follows2:

Table 14. Hyperparameters for the unconditional LDMs trained on the CelebA dataset for the analysis in Fig. 7. All models trained on a single NVIDIA A100. ∗: All models are trained for 500k iterations. If converging earlier, we used the best checkpoint for assessing the provided FID scores.   

<html><body><table><tr><td></td><td>LDM-1</td><td>LDM-2</td><td>LDM-4</td><td>LDM-8</td><td>LDM-16</td><td>LDM-32</td></tr><tr><td>z-shape</td><td>256×256×3</td><td>128 × 128× 2</td><td>64×64×3</td><td>32 ×32×4</td><td>16 × 16 × 8</td><td>88×8×32</td></tr><tr><td>|</td><td></td><td>2048</td><td>8192</td><td>16384</td><td>16384</td><td>16384</td></tr><tr><td>Diffusion steps</td><td>1000</td><td>1000</td><td>1000</td><td>1000</td><td>1000</td><td>1000</td></tr><tr><td>Noise Schedule</td><td>linear</td><td>linear</td><td>linear</td><td>linear</td><td>linear</td><td>linear</td></tr><tr><td>Model Size</td><td>270M</td><td>265M</td><td>274M</td><td>258M</td><td>260M</td><td>258M</td></tr><tr><td>Channels</td><td>192</td><td>192</td><td>224</td><td>256</td><td>256</td><td>256</td></tr><tr><td>Depth</td><td>2</td><td>2</td><td>2</td><td>2</td><td>2</td><td>2</td></tr><tr><td>Channel Multiplier</td><td>1,1,2,2,4,4</td><td>1,2,2,4,4</td><td>1,2,3,4</td><td>1,2,4</td><td>1,2.4</td><td>1,2,4</td></tr><tr><td>Attention resolutions</td><td>32,16,8</td><td>32,16,8</td><td>32,16,8</td><td>32,16,8</td><td>16,8,4</td><td>8,4,2</td></tr><tr><td>Head Channels</td><td>32</td><td>32</td><td>32</td><td>32</td><td>32</td><td>32</td></tr><tr><td>Batch Size</td><td>9</td><td>11</td><td>48</td><td>96</td><td>128</td><td>128</td></tr><tr><td>Iterations*</td><td>500k</td><td>500k</td><td>500k</td><td>500k</td><td>500k</td><td>500k</td></tr><tr><td>Learning Rate</td><td>9e-5</td><td>1.1e-4</td><td>9.6e-5</td><td>9.6e-5</td><td>1.3e-4</td><td>1.3e-4</td></tr></table></body></html>

Table 15. Hyperparameters for the conditional $L D M s$ from Sec. 4. All models trained on a single NVIDIA A100 except for the inpainting model which was trained on eight V100.   

<html><body><table><tr><td>Task</td><td>Text-to-Image</td><td colspan="2">Layout-to-Image</td><td>Class-Label-to-Image</td><td>Super Resolution</td><td>Inpainting</td><td>Semantic-Map-to-Image</td></tr><tr><td>Dataset</td><td>LAION</td><td>OpenImages</td><td>COCO</td><td>ImageNet</td><td>ImageNet</td><td>Places</td><td>Landscapes</td></tr><tr><td>f</td><td>8</td><td>4</td><td>8</td><td>4</td><td>4</td><td>4</td><td>8</td></tr><tr><td>z-shape</td><td>32×32×4</td><td>64×64×3</td><td>32×32×4</td><td>64×64×3</td><td>64 ×64× 3</td><td>64×64×3</td><td>32 × 32 × 4</td></tr><tr><td></td><td></td><td>8192</td><td>16384</td><td>8192</td><td>8192</td><td>8192</td><td>16384</td></tr><tr><td>Diffusion steps</td><td>1000</td><td>1000</td><td>1000</td><td>1000</td><td>1000</td><td>1000</td><td>1000</td></tr><tr><td>Noise Schedule</td><td> linear</td><td>linear</td><td>linear</td><td>linear</td><td>linear</td><td>linear</td><td>linear</td></tr><tr><td>Model Size</td><td>1.45B</td><td>306M</td><td>345M</td><td>395M</td><td>169M</td><td>215M</td><td>215M</td></tr><tr><td>Channels</td><td>320</td><td>128</td><td>192</td><td>192</td><td>160</td><td>128</td><td>128</td></tr><tr><td>Depth</td><td>2</td><td>2</td><td>2</td><td>2</td><td>2</td><td>2</td><td>2</td></tr><tr><td>Channel Multiplier</td><td>1,2,4,4</td><td>1,2,3.4</td><td>1,2,4</td><td>1,2,3.5</td><td>1,2,2,4</td><td>1,4,8</td><td>1,4,8</td></tr><tr><td>Number of Heads</td><td>8</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Dropout</td><td></td><td></td><td>0.1</td><td>-</td><td>1</td><td></td><td>1</td></tr><tr><td>Batch Size</td><td>680</td><td>24</td><td>48</td><td>1200</td><td>64</td><td>128</td><td>48</td></tr><tr><td>Iterations</td><td>390K</td><td>4.4M</td><td>170K</td><td>178K</td><td>860K</td><td>360K</td><td>360K</td></tr><tr><td>Learning Rate</td><td>1.0e-4</td><td>4.8e-5</td><td>4.8e-5</td><td>1.0e-4</td><td>6.4e-5</td><td>1.0e-6</td><td>4.8e-5</td></tr><tr><td>Conditioning</td><td>CA</td><td>CA</td><td>CA</td><td>CA</td><td>concat</td><td>concat</td><td>concat</td></tr><tr><td>(C)A-resolutions</td><td>32,16,8</td><td>32,16,8</td><td>32,16, 8</td><td>32,16,8</td><td></td><td></td><td></td></tr><tr><td>Embedding Dimension</td><td>1280</td><td>512</td><td>512</td><td>512</td><td></td><td></td><td></td></tr><tr><td>Transformer Depth</td><td>1</td><td>3</td><td>2</td><td>1</td><td></td><td>-</td><td></td></tr></table></body></html>

$$
\begin{array} { r l } & { \zeta \gets \mathrm { T o k E m b } ( y ) + \mathrm { P o s E m b } ( \mathrm { y } ) } \\ & { \mathrm { f o r ~ } i = 1 , \dots , N : } \\ & { \quad \zeta _ { 1 } \gets \mathrm { L a y e r N o r m } ( \zeta ) } \\ & { \quad \zeta _ { 2 } \gets \mathrm { M u l t i H e a d S e l f A t t e n t i o n } ( \zeta _ { 1 } ) + \zeta } \\ & { \quad \zeta _ { 3 } \gets \mathrm { L a y e r N o r m } ( \zeta _ { 2 } ) } \\ & { \quad \zeta \gets \mathrm { M L P } ( \zeta _ { 3 } ) + \zeta _ { 2 } } \\ & { \quad \zeta \gets \mathrm { L a y e r N o r m } ( \zeta ) } \end{array}
$$

With $\zeta$ available, the conditioning is mapped into the UNet via the cross-attention mechanism as depicted in Fig. 3. We modify the “ablated UNet” [15] architecture and replace the self-attention layer with a shallow (unmasked) transformer consisting of $T$ blocks with alternating layers of (i) self-attention, (ii) a position-wise MLP and (iii) a cross-attention layer;

see Tab. 16. Note that without (ii) and (iii), this architecture is equivalent to the “ablated UNet”.

While it would be possible to increase the representational power of $\tau _ { \theta }$ by additionally conditioning on the time step $t$ , we do not pursue this choice as it reduces the speed of inference. We leave a more detailed analysis of this modification to future work.

For the text-to-image model, we rely on a publicly available3 tokenizer [99]. The layout-to-image model discretizes the spatial locations of the bounding boxes and encodes each box as a $( l , b , c )$ -tuple, where $l$ denotes the (discrete) top-left and $b$ the bottom-right position. Class information is contained in $c$ .

See Tab. 17 for the hyperparameters of $\tau _ { \theta }$ and Tab. 13 for those of the UNet for both of the above tasks.

Note that the class-conditional model as described in Sec. 4.1 is also implemented via cross-attention, where $\tau _ { \theta }$ is a single learnable embedding layer with a dimensionality of 512, mapping classes $y$ to $\zeta \in \mathbb { R } ^ { 1 \times 5 1 2 }$ .

<html><body><table><tr><td>input</td><td>Rhxwxc</td></tr><tr><td>LayerNorm</td><td>Rhxwxc</td></tr><tr><td>Conv1x1</td><td>Rhxwxd·nh</td></tr><tr><td>Reshape</td><td>Rh·wxd·nh</td></tr><tr><td>SelfAttention</td><td>Rh·wxd·nh</td></tr><tr><td>×T MLP</td><td>Rh·wxd·nh</td></tr><tr><td>CrossAttention</td><td>Rh.wxd.nh</td></tr><tr><td>Reshape</td><td>Rhxwxd·nh</td></tr><tr><td>Conv1x1</td><td>Rhxwxc</td></tr></table></body></html>

Table 16. Architecture of a transformer block as described in Sec. E.2.1, replacing the self-attention layer of the standard “ablated UNet” architecture [15]. Here, $n _ { h }$ denotes the number of attention heads and $d$ the dimensionality per head.

Table 17. Hyperparameters for the experiments with transformer encoders in Sec. 4.3.   

<html><body><table><tr><td></td><td>Text-to-Image</td><td>Layout-to-Image</td></tr><tr><td>seq-length</td><td>77</td><td>92</td></tr><tr><td>depth N</td><td>32</td><td>16</td></tr><tr><td>dim</td><td>1280</td><td>512</td></tr></table></body></html>

# E.2.2 Inpainting

For our experiments on image-inpainting in Sec. 4.5, we used the code of [88] to generate synthetic masks. We use a fixed set of 2k validation and 30k testing samples from Places [108]. During training, we use random crops of size $2 5 6 \times 2 5 6$ and evaluate on crops of size $5 1 2 \times 5 1 2$ . This follows the training and testing protocol in [88] and reproduces their reported metrics (see † in Tab. 7). We include additional qualitative results of LDM-4, w/ attn in Fig. 21 and of LDM-4, w/o attn, big, $w / f t$ in Fig. 22.

# E.3. Evaluation Details

This section provides additional details on evaluation for the experiments shown in Sec. 4.

# E.3.1 Quantitative Results in Unconditional and Class-Conditional Image Synthesis

We follow common practice and estimate the statistics for calculating the FID-, Precision- and Recall-scores [29,50] shown in Tab. 1 and 10 based on 50k samples from our models and the entire training set of each of the shown datasets. For calculating FID scores we use the torch-fidelity package [60]. However, since different data processing pipelines might lead to different results [64], we also evaluate our models with the script provided by Dhariwal and Nichol [15]. We find that results mainly coincide, except for the ImageNet and LSUN-Bedrooms datasets, where we notice slightly varying scores of 7.76 (torch-fidelity) vs. 7.77 (Nichol and Dhariwal) and 2.95 vs 3.0. For the future we emphasize the importance of a unified procedure for sample quality assessment. Precision and Recall are also computed by using the script provided by Nichol and Dhariwal.

# E.3.2 Text-to-Image Synthesis

Following the evaluation protocol of [66] we compute FID and Inception Score for the Text-to-Image models from Tab. 2 by comparing generated samples with 30000 samples from the validation set of the MS-COCO dataset [51]. FID and Inception Scores are computed with torch-fidelity.

# E.3.3 Layout-to-Image Synthesis

For assessing the sample quality of our Layout-to-Image models from Tab. 9 on the COCO dataset, we follow common practice [37, 87, 89] and compute FID scores the 2048 unaugmented examples of the COCO Segmentation Challenge split. To obtain better comparability, we use the exact same samples as in [37]. For the OpenImages dataset we similarly follow their protocol and use 2048 center-cropped test images from the validation set.

# E.3.4 Super Resolution

We evaluate the super-resolution models on ImageNet following the pipeline suggested in [72], i.e. images with a shorter size less than $2 5 6 { \mathrm { ~ p x } }$ are removed (both for training and evaluation). On ImageNet, the low-resolution images are produced using bicubic interpolation with anti-aliasing. FIDs are evaluated using torch-fidelity [60], and we produce samples on the validation split. For FID scores, we additionally compare to reference features computed on the train split, see Tab. 5 and Tab. 11.

# E.3.5 Efficiency Analysis

For efficiency reasons we compute the sample quality metrics plotted in Fig. 6, 17 and 7 based on $5 \mathrm { k }$ samples. Therefore, the results might vary from those shown in Tab. 1 and 10. All models have a comparable number of parameters as provided in Tab. 13 and 14. We maximize the learning rates of the individual models such that they still train stably. Therefore, the learning rates slightly vary between different runs cf . Tab. 13 and 14.

# E.3.6 User Study

For the results of the user study presented in Tab. 4 we followed the protocoll of [72] and and use the 2-alternative force-choice paradigm to assess human preference scores for two distinct tasks. In Task-1 subjects were shown a low resolution/masked image between the corresponding ground truth high resolution/unmasked version and a synthesized image, which was generated by using the middle image as conditioning. For SuperResolution subjects were asked: ’Which of the two images is a better high quality version of the low resolution image in the middle?’. For Inpainting we asked ’Which of the two images contains more realistic inpainted regions of the image in the middle?’. In Task-2, humans were similarly shown the lowres/masked version and asked for preference between two corresponding images generated by the two competing methods. As in [72] humans viewed the images for 3 seconds before responding.

# F. Computational Requirements

<html><body><table><tr><td>Method</td><td>Generator Compute</td><td>Classifier Compute</td><td>Overall Compute</td><td>Inference Throughput*</td><td>Nparams</td><td>FID↓</td><td>IS↑</td><td>Precision↑</td><td>Recal↑</td></tr><tr><td>LSUN Churches 256²</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>StyleGAN2 [42]t</td><td>64</td><td>_-</td><td>64</td><td></td><td>59M</td><td>3.86</td><td>__</td><td>-</td><td>-</td></tr><tr><td>LDM-8 (ours,100 steps, 410K)</td><td>18</td><td></td><td>18</td><td>6.80</td><td>256M</td><td>4.02</td><td></td><td>0.64</td><td>0.52</td></tr><tr><td>LSUN Bedrooms 256²</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>232</td><td>_-</td><td>232</td><td>0.03</td><td>52M</td><td>2.95</td><td></td><td>0.66</td><td>0.1</td></tr><tr><td>CelebA-HQ 2562</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LDM-4 (ours, 500 steps, 410K)</td><td>14.4</td><td></td><td>14.4</td><td>0.43</td><td>274M</td><td>5.11</td><td></td><td>0.72</td><td>0.49</td></tr><tr><td>FFHQ 2562</td><td></td><td></td><td></td><td></td><td></td><td></td><td>-</td><td></td><td></td></tr><tr><td>StyleGAN2 [42]</td><td>32.13</td><td></td><td>32.13†</td><td>-</td><td>59M</td><td>3.8</td><td></td><td></td><td></td></tr><tr><td>LDM-4 (ours,200 steps,635K)</td><td>26</td><td></td><td>26</td><td>1.07</td><td>274M</td><td>4.98</td><td></td><td>0.73</td><td>0.50</td></tr><tr><td>ImageNet 2562</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>2960</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VOGN-8 s</td><td></td><td></td><td>2960</td><td></td><td>5SM</td><td>1.</td><td>_-</td><td></td><td></td></tr><tr><td>BigGAN-deep [3]t</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>128-256</td><td></td><td>128-256 916</td><td>-</td><td>340M</td><td>6.95</td><td>203.6±26</td><td>0.87</td><td>0.28</td></tr><tr><td>ADM[15](250 steps)†</td><td>916 916</td><td>46</td><td>962</td><td>0.12</td><td>554M</td><td>10.94</td><td>100.98</td><td>0.69</td><td>0.63</td></tr><tr><td>ADM-G[15] (25 steps)†</td><td></td><td>46</td><td>962</td><td>0.7</td><td>608M</td><td>5.58</td><td></td><td>0.81</td><td>0.49</td></tr><tr><td>ADM-G [15](250 steps)t</td><td>916</td><td>30</td><td></td><td>0.07</td><td>608M</td><td>4.59</td><td>186.7</td><td>0.82</td><td>0.52</td></tr><tr><td>ADM-G,ADM-U [15](250 steps)†</td><td>329</td><td>12</td><td>349 91</td><td>n/a</td><td>n/a</td><td>3.85</td><td>221.72</td><td>0.84</td><td>0.53</td></tr><tr><td>LDM-8-G (ours,100,2.9M)</td><td>79</td><td></td><td></td><td>1.93</td><td>506M</td><td>8.11</td><td>190.4±2.6</td><td>0.83</td><td>0.36</td></tr><tr><td>LDM-8 (ours,200 ddim steps 2.9M,batch size 64)</td><td>79</td><td></td><td>79</td><td>1.9</td><td>395M</td><td>17.41</td><td>72.92</td><td>0.65</td><td>0.62</td></tr><tr><td>LDM-4 (ours,250 ddim steps 178K,batch size 1200)</td><td>271</td><td></td><td>271</td><td>0.7</td><td>400M</td><td>10.56</td><td>103.49±124</td><td>0.71</td><td>0.62</td></tr><tr><td>LDM-4-G (ours,250 ddimsteps 178K, batch size 1200,classifier-free guidance [32] scale 1.25)</td><td>271</td><td></td><td>271</td><td>0.4</td><td>400M</td><td>3.95</td><td>178.22 ±2.43</td><td>0.81</td><td>0.55</td></tr><tr><td></td><td>271</td><td></td><td>271</td><td>0.4</td><td>400M</td><td>3.60</td><td>247.67±5.5</td><td></td><td></td></tr><tr><td>LDM-4-G (ours,250 ddim steps 178K, batch size 1200,clasifier-free guidance [32j scale 1.5)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.87</td><td>0.48</td></tr></table></body></html>

Table 18. Comparing compute requirements during training and inference throughput with state-of-the-art generative models. Compute during training in V100-days, numbers of competing methods taken from [15] unless stated differently;∗: Throughput measured in samples/sec on a single NVIDIA A100;†: Numbers taken from [15] ; $\ddagger$ : Assumed to be trained on 25M train examples; ††: R-FID vs. ImageNet validation set

In Tab 18 we provide a more detailed analysis on our used compute ressources and compare our best performing models on the CelebA-HQ, FFHQ, LSUN and ImageNet datasets with the recent state of the art models by using their provided numbers, cf . [15]. As they report their used compute in V100 days and we train all our models on a single NVIDIA A100 GPU, we convert the A100 days to V100 days by assuming a $\times 2 . 2$ speedup of A100 vs $\mathrm { V 1 0 0 [ 7 4 ] ^ { 4 } }$ . To assess sample quality, we additionally report FID scores on the reported datasets. We closely reach the performance of state of the art methods as StyleGAN2 [42] and ADM [15] while significantly reducing the required compute resources.

# G. Details on Autoencoder Models

We train all our autoencoder models in an adversarial manner following [23], such that a patch-based discriminator $D _ { \psi }$ is optimized to differentiate original images from reconstructions $\mathcal { D } ( \mathcal { E } ( x ) )$ . To avoid arbitrarily scaled latent spaces, we regularize the latent $z$ to be zero centered and obtain small variance by introducing an regularizing loss term $L _ { r e g }$ . We investigate two different regularization methods: (i) a low-weighted Kullback-Leibler-term between $q \varepsilon ( z | x ) =$ $\mathcal { N } ( z ; \mathcal { E } _ { \mu } , \mathcal { E } _ { \sigma ^ { 2 } } )$ and a standard normal distribution $\mathcal { N } ( z ; 0 , 1 )$ as in a standard variational autoencoder [46, 69], and, (ii) regularizing the latent space with a vector quantization layer by learning a codebook of $| { \mathcal { Z } } |$ different exemplars [96]. To obtain high-fidelity reconstructions we only use a very small regularization for both scenarios, i.e. we either weight the $\mathbb { K L }$ term by a factor $\sim 1 0 ^ { - 6 }$ or choose a high codebook dimensionality $| { \mathcal { Z } } |$ .

The full objective to train the autoencoding model $( \mathcal { E } , \mathcal { D } )$ reads:

$$
{ L } _ { \mathrm { { A u t e o n c o d e r } } } = \operatorname* { m i n } _ { \mathcal { E } , D } \operatorname* { m a x } _ { \psi } \Big ( { L } _ { r e c } ( x , \mathcal { D } ( \mathcal { E } ( x ) ) ) - { L } _ { a d v } ( \mathcal { D } ( \mathcal { E } ( x ) ) ) + \log D _ { \psi } ( x ) + { L } _ { r e g } ( x ; \mathcal { E } , \mathcal { D } ) \Big )
$$

DM Training in Latent Space Note that for training diffusion models on the learned latent space, we again distinguish two cases when learning $p ( z )$ or $p ( z | y )$ (Sec. 4.3): (i) For a KL-regularized latent space, we sample $z = \mathcal { E } _ { \mu } ( x ) { + } \mathcal { E } _ { \sigma } ( x ) { \cdot } \varepsilon = : \mathcal { E } ( x )$ , where $\varepsilon \sim \mathcal { N } ( 0 , 1 )$ . When rescaling the latent, we estimate the component-wise variance

$$
\hat { \sigma } ^ { 2 } = \frac { 1 } { b c h w } \sum _ { b , c , h , w } ( z ^ { b , c , h , w } - \hat { \mu } ) ^ { 2 }
$$

from the first batch in the data, where $\begin{array} { r } { \hat { \mu } = \frac { 1 } { b c h w } \sum _ { b , c , h , w } z ^ { b , c , h , w } } \end{array}$ . The output of $\mathcal { E }$ is scaled such that the rescaled latent has unit standard deviation, i.e. $\begin{array} { r } { z  \frac { z } { \hat { \sigma } } = \frac { \mathcal { E } ( x ) } { \hat { \sigma } } } \end{array}$ . (ii) FoPr a VQ-regularized latent space, we extract $z$ before the quantization layer and absorb the quantization operation into the decoder, $i . e$ . it can be interpreted as the first layer of $\mathcal { D }$ .

# H. Additional Qualitative Results

Finally, we provide additional qualitative results for our landscapes model (Fig. 12, 23, 24 and 25), our class-conditional ImageNet model (Fig. 26 - 27) and our unconditional models for the CelebA-HQ, FFHQ and LSUN datasets (Fig. 28 - 31). Similar as for the inpainting model in Sec. 4.5 we also fine-tuned the semantic landscapes model from Sec. 4.3.2 directly on $5 1 2 ^ { 2 }$ images and depict qualitative results in Fig. 12 and Fig. 23. For our those models trained on comparably small datasets, we additionally show nearest neighbors in VGG [79] feature space for samples from our models in Fig. 32 - 34.

![](images/e9a7c732361986c0dd9ecb33bf16626fd5c424cc1a6533168699faf2678fc8ae.jpg)  
Figure 19. LDM-BSR generalizes to arbitrary inputs and can be used as a general-purpose upsampler, upscaling samples from the LSUNCows dataset to $1 0 2 4 ^ { 2 }$ resolution.

![](images/68888ebac8d1495e9c50ae042491b8fbd6f8e8f0bf1cda64438eed4cc25be814.jpg)  
Figure 20. Qualitative superresolution comparison of two random samples between LDM-SR and baseline-diffusionmodel in Pixelspac Evaluated on imagenet validation-set after same amount of training steps.

![](images/fe9edef475bd5900ab7dbc4ceed4301b1a4397a082c94cfb02a6acd2c05371e8.jpg)  
Figure 21. Qualitative results on image inpainting. In contrast to [88], our generative approach enables generation of multiple diver samples for a given input.

![](images/ab960d5b4b0de1fb8dc9e9e43d771cf5cfcdbdfe03b9fbc0a71ccc6b31a9df21.jpg)  
Figure 22. More qualitative results on object removal as in Fig. 11.

![](images/3f0be2ecb79033bbd2795525864e043ec8322f64f2d8db7d5806e82b66831ae1.jpg)  
Figure 23. Convolutional samples from the semantic landscapes model as in Sec. 4.3.2, finetuned on $5 1 2 ^ { 2 }$ images.

![](images/b3dd989bec9ab17bda4331a2c73d2b2489cbe1e601ed54e1e6ba489f16efd73d.jpg)  
Figure 24. A LDM trained on $2 5 6 ^ { 2 }$ resolution can generalize to larger resolution for spatially conditioned tasks such as semantic synthesis of landscape images. See Sec. 4.3.2.

![](images/c8dfbe3da2fe0aac3d2cd92464169121dd08c783d4d6904d1494bb61ed3c5009.jpg)  
Figure 25. When provided a semantic map as conditioning, our $L D M s$ generalize to substantially larger resolutions than those seen during training. Although this model was trained on inputs of size $2 5 6 ^ { 2 }$ it can be used to create high-resolution samples as the ones shown here, which are of resolution $1 0 2 4 \times 3 8 4$ .

![](images/7f1acaeb62c8e68b74961cdb459f96538793282704707a3095d8017a2c3cb040.jpg)  
Random class conditional samples on the ImageNet dataset   
Figure 26. Random samples from $L D M { - } 4$ trained on the ImageNet dataset. Sampled with classifier-free guidance [32] scale $s = 5 . 0$ and 200 DDIM steps with $\eta = 1 . 0$ .

![](images/1dd33782ae3e9a7339006e99a7e76ca9fe483c7f0a593a73592744ed63fc3ddd.jpg)  
Random class conditional samples on the ImageNet dataset   
Figure 27. Random samples from LDM-4 trained on the ImageNet dataset. Sampled with classifier-free guidance [32] scale $s = 3 . 0$ and 200 DDIM steps with $\eta = 1 . 0$ .

![](images/4fa43f9812bd6c1491201613235decfa47a6b80e0b5f35d84e0c1462fdeba28b.jpg)  
Random samples on the CelebA-HQ dataset   
Figure 28. Random samples of our best performing model LDM-4 on the CelebA-HQ dataset. Sampled with 500 DDIM steps and $\eta = 0$ $( \mathrm { F I D } = 5 . 1 5 ) \$ ).

![](images/dc50cdfac8d1026a65e2c27c0d3d599455626978a545cf52a70f27845d6d5623.jpg)  
Random samples on the FFHQ dataset   
Figure 29. Random samples of our best performing model LDM-4 on the FFHQ dataset. Sampled with 200 DDIM steps and $\eta = 1$ (FID $= 4 . 9 8 \dot { }$ ).

![](images/3d37d117cf13e267eb327573a5ec9b1a6272f647c16bdd1e02a3cddb253ca767.jpg)  
Random samples on the LSUN-Churches dataset   
Figure 30. Random samples of our best performing model LDM-8 on the LSUN-Churches dataset. Sampled with 200 DDIM steps and $\eta = 0$ $\mathrm { { F I D } } = 4 . 4 8 { , }$ ).

![](images/e221af7e53886c17c8a16e18afd8f4cccbd524d510331f92a774142a2e7218db.jpg)  
Random samples on the LSUN-Bedrooms dataset   
Figure 31. Random samples of our best performing model LDM-4 on the LSUN-Bedrooms dataset. Sampled with 200 DDIM steps and $\eta = 1$ $\mathrm { \ F I D } = 2 . 9 5 \$ ).

![](images/36b151806106bbed84bcccddc795ffb73a7e61a5d0ec0da31a882f37fff74322.jpg)  
Figure 32. Nearest neighbors of our best CelebA-HQ model, computed in the feature space of a VGG-16 [79]. The leftmost sample is from our model. The remaining samples in each row are its 10 nearest neighbors.

![](images/9dd403f98af69d7e1618c76e0db5000850abcacecb73a6be6a93d1061d447442.jpg)  
Nearest Neighbors on the FFHQ dataset   
Figure 33. Nearest neighbors of our best FFHQ model, computed in the feature space of a VGG-16 [79]. The leftmost sample is from our model. The remaining samples in each row are its 10 nearest neighbors.

![](images/0f3d8e331e25f23813a95c730593574d3c7ab53d560d259472ab044fcf4b0d33.jpg)  
Figure 34. Nearest neighbors of our best LSUN-Churches model, computed in the feature space of a VGG-16 [79]. The leftmost sample is from our model. The remaining samples in each row are its 10 nearest neighbors.