# SINE: SINgle Image Editing with Text-to-Image Diffusion Models

Zhixing Zhang1 Ligong Han1\* Arnab Ghosh2 Dimitris Metaxas1 Jian Ren2 1Rutgers University 2Snap Inc.

# Abstract

Recent works on diffusion models have demonstrated a strong capability for conditioning image generation, e.g., text-guided image synthesis. Such success inspires many efforts trying to use large-scale pre-trained diffusion models for tackling a challenging problem–real image editing. Works conducted in this area learn a unique textual token corresponding to several images containing the same object. However, under many circumstances, only one image is available, such as the painting of the Girl with a Pearl Earring. Using existing works on fine-tuning the pretrained diffusion models with a single image causes severe overfitting issues. The information leakage from the pretrained diffusion models makes editing can not keep the same content as the given image while creating new features depicted by the language guidance. This work aims to address the problem of single-image editing. We propose a novel model-based guidance built upon the classifier-free guidance so that the knowledge from the model trained on a single image can be distilled into the pre-trained diffusion model, enabling content creation even with one given image. Additionally, we propose a patch-based fine-tuning that can effectively help the model generate images of arbitrary resolution. We provide extensive experiments to validate the design choices of our approach and show promising editing capabilities, including changing style, content addition, and object manipulation1.

# 1. Introduction

Automatic real image editing is an exciting direction, enabling content generation and creation with minimal effort. Although many works have been conducted in this area, achieving high-fidelity semantic manipulation on an image is still a challenging problem for the generative models, considering the target image might be out of the training data distribution [4, 10, 18, 19, 25, 50, 51]. The recently introduced large-scale text-to-image models, e.g., DALL·E 2 [30], Imagen [35], Parti [47], and StableDiffusion [32], can perform high-quality and diverse image generation with natural language guidance. The success of these works has inspired many subsequent efforts to leverage the pre-trained large-scale models for real image editing [9, 13, 33]. They show that, with properly designed prompts and a limited number of fine-tuning steps, the text-to-image models can manipulate a given subject with text guidance.

![](images/42aab07136651b5f9eac21a9047800237cff081707db60e43a332316aed1f1e5.jpg)  
Figure 1. With only one real image, i.e., Source Image, our method is able to manipulate and generate the content in various ways, such as changing style, adding context, modifying the object, and enlarging the resolution, through guidance from the text prompt.

On the downside, the recent text-guided editing works that build upon the diffusion models suffer several limitations. First, the fine-tuning process might lead to the pretrained large-scale model overfit on the real image, which degrades the synthesized images’ quality when editing. To tackle these issues, methods like using multiple images with the same content and applying regularization terms on the same object have been introduced [4, 33]. However, querying multiple images with identical content or object might not be an available choice; for instance, there is only one painting for Girl with a Pearl Earring. Directly editing the single image brings information leakage from the pretrained large-scale models, generating images with different content (examples in Fig. 5); therefore, the application scenarios of these methods are greatly constrained. Second, these works lack a reasonable understanding of the object geometry for the edited image. Thus, generating images with different spatial size as the training data cause undesired artifacts, e.g., repeated objects, and incorrectly modified geometry (examples in Fig. 5). Such drawbacks restrict applying these methods for generating images with an arbitrary resolution, e.g., synthesizing high-resolution images from a single photo of a castle (as in Fig. 4), again limiting the usage of these methods.

In this work, we present SINE, a framework utilizing pre-trained text-to-image diffusion models for SINgle image Editing and content manipulation. We build our approach based upon existing text-guided image generation approaches [9, 33] and propose the following novel techniques to solve overfitting issues on content and geometry, and language drift [21, 24]:

• First, by appropriately modifying the classifier-free guidance [15], we introduce model-based classifier-free guidance that utilizes the diffusion model to provide the score guidance for content and structure. Taking advantage of the step-by-step sampling process used in diffusion models, we use the model fine-tuned on a single image to plant a content “seed” at the early stage of the denoising process and allow the pre-trained large-scale text-toimage model to edit creatively conditioned with the language guidance at a later stage.

• Second, to decouple the correlation between pixel position and content, we propose a patch-based fine-tuning strategy, enabling generation on arbitrary resolution.

With a text descriptor describing the content that is aimed to be manipulated and language guidance depicting the desired output, our approach can edit the single unique image to the targeted domain with details preserved in arbitrary resolution. The output image keeps the structure and background intact while having features well-aligned with the target language guidance. As shown in Fig. 1, trained on a painting Girl with a Pearl Earring with resolution as $5 1 2 \times 5 1 2$ , we can sample an image of a sculpture of the girl at the resolution of $6 4 0 \times 5 1 2$ with the identity features preserved. Moreover, our method can successfully handle various edits such as style transfer, content addition, and object manipulation (more examples in Fig. 3). We hope our method can further boost creative content creation by opening the door to editing arbitrary images.

# 2. Related Work

Text-guided image synthesis has drawn considerable attention in the generative model context [1, 2, 12, 27, 28, 31, 42, 45–48, 52]. The recent development of diffusion models [14,39–41] introduced new solutions to this problem and produced impressive results [26, 30, 32, 35]. With the significant improvement of these models, rather than training a large-scale text-to-image model from scratch, a leading line of works focuses on taking advantage of the existing pretrained model and manipulating images according to given natural language guidance [3, 9, 13, 17, 22, 33]. In these works, studies explore text-based interfaces for image editing [1, 3, 28], style transfer [20, 23], and generator domain adaption [10, 18].

The development of the diffusion model provides a giant and flexible design space for this task. Many works utilize pre-trained diffusion models as generative priors and are training-free. ILVR [6] guides the denoising process by replacing the low-frequency part of the sample with that of the target reference image. SDEdit [25] applies the diffusion process first on an image or a user-created semantic map and then conducts the denoising procedure conditioned with the desired output. Blended diffusion [3] performs language-guided inpainting with a given mask.

Another line of research showed great potential and semantic editing ability of fine-tuning. DiffusionCLIP [18] leverages the CLIP [29] model to provide gradients for image manipulation and delivers impressive results on style transfer. Textual-Inversion [9] and DreamBooth [33] finetune the text embedding or the full diffusion model using a few personalized images (typically $3 \sim 5$ ) to synthesize images of the same object in a novel context. These methods, however, either drastically change the layout of the original image when dealing with a single image or can not fully leverage the generalization ability of the pre-trained model for editing due to overfitting or language drift. Notably, Prompt-to-Prompt [13] controls the editing of synthesized images by manipulating the cross-attention maps; however, its editing ability is limited when applied to real images.

This work introduces a solution to achieve image fidelity and text alignment simultaneously. Our method can perform high-quality semantic editing globally and locally on one single image. On the other hand, previous works lack an understanding of the object geometry of the edited image. When editing the image at an arbitrary resolution, the artifacts in the results will be obvious. Prior works have investigated generating images at arbitrary resolution using positional encoding as inductive bias [34, 44, 44] so that the correlation between content and position can be eliminated. Anyres-GAN [5] adopt a patch training mechanism to leverage high-resolution data to help the generation of images in the low-resolution domain. We propose a patch-based finetuning method to achieve arbitrary resolution editing.

# 3. Methods

For one arbitrary in-the-wild image, our goal is to edit the image via language while preserving the maximal amount of details from the original image. To do so, we leverage the generalization ability of pre-trained large-scale text-to-image models [32]. An intuitive approach is to finetune the diffusion models with the single image and text description, similar to DreamBooth [33]. Ideally, it should provide a model that can reconstruct the input image using the given text descriptor and synthesize new images when given other language guidance. Unfortunately, we find the model can easily overfit the single trained image and its corresponding text description. Thus, although the fine-tuned model can still reconstruct the input image perfectly, it can no longer synthesize diverse images according to the given language guidance (as shown in Fig. 5). Moreover, it struggles to generate arbitrary resolution images due to the lack of positional information (as in Fig. 4).

To solve the above issues, we propose a test-time modelbased classifier-free guidance and a patch-based fine-tuning technique. An overview of our method is illustrated in Fig. 2. In the following sections, we review the backbone model used in our approach (Sec. 3.1). Then, we describe how to overcome the overfitting problem with model-based guidance (Sec. 3.2). Lastly, we present how to address the problem of limited resolution generation (Sec. 3.3).

# 3.1. Language-Guided Diffusion Models

We use the latent diffusion models (LDMs) [32] trained on a large-scale dataset as our base model and implement the proposed approaches by fine-tuning the pre-trained model. LDMs is a class of Denoising Diffusion Probabilistic Models (DDPMs) [14] that contains an auto-encoder trained on images, and a diffusion model learned on the latent space constructed by the auto-encoder. The encoder $\mathcal { E }$ encodes a given image $\dot { \boldsymbol { \mathcal { T } } } \in \mathbb { R } ^ { H \times W \times 3 }$ to a latent representation $\mathbf { z }$ , such that $\mathbf { z } = \mathcal { E } ( \mathcal { T } )$ . The decoder $\mathcal { D }$ reconstructs the estimated image $\tilde { \mathcal { T } }$ from the latent, such that $\tilde { \mathcal { T } } = \mathcal { D } ( { \mathbf z } )$ and $\tilde { \mathcal { T } } \approx \mathcal { I }$ . The diffusion model is trained to produce latent codes within the pre-trained latent space. The most intriguing property of LDMs is that the diffusion model can be conditioned on class labels, images, and text prompt. The conditional LDM is learned as follows:

$$
L _ { L D M } : = \mathbb { E } _ { \mathcal { E } ( \mathcal { T } ) , y , \epsilon \sim \mathcal { N } ( 0 , 1 ) , t } \left[ \Vert \epsilon - \epsilon _ { \theta } \left( \mathbf { z } _ { t } , t , \tau _ { \theta } ( y ) \right) \Vert _ { 2 } ^ { 2 } \right] ,
$$

where $t$ is the time step, $\mathbf { z } _ { t }$ is the latent noised to time $t , \epsilon$ is the unscaled noise sample, $\scriptstyle \epsilon _ { \theta }$ is the denoising model, $y$ is the conditioning input, and $\tau _ { \theta }$ maps $y$ to a conditioning vector. During training time, $\scriptstyle \epsilon _ { \theta }$ and $\scriptstyle { \tau _ { \theta } }$ are jointly optimized. A random noise tensor is sampled and denoised at inference time based on the conditioning input, e.g., text prompt, to produce a new latent. Inspired by DreamBooth [33], we construct the text prompt for fine-tuning a single image as “a photo/painting of a $[ * ]$ [class noun]”, where $^ { 6 6 } [ * ] ^ { 3 }$ is a unique identifier and “[class noun]” is a coarse class descriptor (e.g., “castle”, “lake”, “car”, etc.).

# 3.2. Model-Based Classifier-Free Guidance

With the above-presented LDMs, we introduce our approach, inspired by classifier-free guidance, to overcome overfitting when fine-tuning LDMs with one image.

Classifier-free guidance [15] is a technique widely adopted by prior text-to-image diffusion models [32, 35]. A single diffusion model is trained using conditional and unconditional objectives by randomly dropping the condition during training. When sampling, a linear combination of the conditional and unconditional score estimation is used:

![](images/2eb2cdb607c7ac84a7295d947998ea05ae071d647b25bd9a465e95989fb62782.jpg)  
Figure 2. Overview of our method. (a) Given a source image, we first randomly crop it into patches and get the corresponding latent code $\textbf { z }$ with the pre-trained encoder. At fine-tune time, the denoising model, $\scriptstyle \epsilon _ { \theta }$ , takes three inputs: noisy latent ${ \bf z } _ { T }$ , language condition c, and positional embedding for the area where the noisy latent is obtained. (b) During sampling, we give additional language guidance about the target domain to edit the image. Also, we sample a noisy latent code ${ \bf z } _ { T }$ with the dimension corresponding to the desired output resolution. Language conditioning for $\scriptstyle \epsilon _ { \theta }$ and $\mathbf { c }$ are given by pre-trained language encoder $\tau _ { \theta }$ with the target language guidance. While for the fine-tuned diffusion model, $\hat { \epsilon } _ { \theta }$ , in addition to the language conditioning ˆc, we also input the positional embedding for the whole image. We employ a linear combination between the score calculated by each model for the first $K$ steps and inference only on pre-trained $\scriptstyle \epsilon _ { \theta }$ after.

$$
\begin{array} { r } { \tilde { \epsilon } _ { \theta } \left( \mathbf { z } _ { t } , \mathbf { c } \right) = w \epsilon _ { \theta } \left( \mathbf { z } _ { t } , \mathbf { c } \right) + ( 1 - w ) \epsilon _ { \theta } \left( \mathbf { z } _ { t } \right) , } \end{array}
$$

where $\boldsymbol { \epsilon } _ { \boldsymbol { \theta } } \left( \mathbf { z } _ { t } , \mathbf { c } \right)$ and $\boldsymbol { \epsilon } _ { \boldsymbol { \theta } } \left( \mathbf { z } _ { t } \right)$ are the conditional and unconditional $\epsilon$ -predictions, $\mathbf { c }$ is the conditioning vector generated by $\tau _ { \theta }$ , and $w$ is the weight for the guidance. The predication is performed using the Tweedie’s formula [7], namely, $\left( \mathbf { z } _ { t } - \sqrt { 1 - \bar { \alpha } _ { t } } \tilde { \epsilon } _ { \theta } \right) / \sqrt { \bar { \alpha } _ { t } }$ , where $\bar { \alpha } _ { t }$ is a function of $t$ that affects the sampling quality.

Since we only have one image as the training data, e.g., painting of Mona Lisa, and one corresponding text descriptor of that image, the diffusion model suffers from overfitting, and severe language drifts after fine-tuning [33]. As a result, the fine-tuned model fails to synthesize images containing features from other language guidance. The overfitting issue might be due to only one repeated prompt used during fine-tuning, making other text prompts no longer accurate enough to control editing (see examples in Fig. 6).

Model-based classifier-free guidance. Existing “personalized” text-guided real image editing works only use one fine-tuned model for image generation and editing [9, 13, 33], ignoring the capacity of pre-trained large-scale textto-image models. Instead, to alleviate the overfitting of the fine-tuned model, we leverage the pre-trained text-toimage model for image generation with the provided language guidance and use the fine-tuned model to provide content features in a fashion of combining scores from the two models, similar to classifier-free guidance.

Specifically, let $\scriptstyle { \hat { \epsilon } } _ { \theta }$ denote the fine-tuned denoising model, and $\scriptstyle \epsilon _ { \theta }$ denote the pre-trained text-to-image model. During sampling, at specified steps, we use our fine-tuned model to guide the pre-trained one by using a linear combination of the scores from each model. Thus, the score estimation in Eqn. 2 becomes:

$$
\begin{array} { r } { \tilde { \epsilon } _ { \theta } \left( \mathbf { z } _ { t } , \mathbf { c } \right) = w \left( v \epsilon _ { \theta } \left( \mathbf { z } _ { t } , \mathbf { c } \right) + ( 1 - v ) \hat { \epsilon } _ { \theta } \left( \mathbf { z } _ { t } , \hat { \mathbf { c } } \right) \right) } \\ { + ( 1 - w ) \epsilon _ { \theta } \left( \mathbf { z } _ { t } \right) , \quad \quad } \end{array}
$$

where $v$ stands for the model guidance weight, cˆ is the language guidance token obtained from the fine-tuned diffusion model with the text prompt used during fine-tuning, and $\mathbf { c }$ is the target language conditioning obtained from the target prompt.

To prevent artifacts from the over-fitted model and maintain the fidelity of the generated image, we propose to sample using Eqn. 3 with $t > K$ and sample using Eqn. 2 for $t \leq K$ . From $K$ to 0, the denoising process only depends on the pre-trained model. Following this approach, we can fully leverage the generalization ability of the pre-trained model (examples in Fig. 6). Also note that this method could be generalized to include multiple prompts or even multiple modalities.

# 3.3. Patch-Based Fine-Tuning

With model-based classifier-free guidance, we are able to edit and manipulate a single image with given language guidance. Here, we further show how to improve the finetuning process for a single training image so that the finetuned model can better understand the content and geometry of the image. Thus, it can provide better content guidance for the large-scale text-to-image model during the sampling time and unleash the potential for generating arbitraryresolution images [8].

Limited-resolution generation. We first review the limitations of the current fine-tuning process. Given an input image $\boldsymbol { \mathcal { T } }$ with resolution as $H \times W$ , we can obtain a downsampled latent code $\textbf { z }$ from the pre-trained encoder. Since the text-to-image diffusion model is pre-trained at a fixed resolution, i.e., $p \times p$ , we need to resize the input image to a corresponding resolution $s p \times s p$ , where $s$ represents the scaling factor of the encoder, to match the resolution for reducing the fine-tuning cost. In essence, prior knowledge of the correlation between the position and content information is learned by the diffusion model. Thus, when sampling from a higher-resolution noise tensor, the generated latent code leads to artifacts like duplicates or position shifting (visual examples in Fig. 5). To tackle such drawbacks, we propose a simple yet effective fine-tuning method.

Patch-based fine-tuning. Inspired by Chai et al. [5], we treat our single training image as a function on coordinate for each pixel, bounded in $[ 0 , H ] \times [ 0 , W ]$ . The diffusion model still generates latent code at the fixed resolution $p \times p$ , but each latent code corresponds to a sub-area in the image. We denote the sub-area as $\mathbf { v } = [ h _ { 1 } , w _ { 1 } , h _ { 2 } , w _ { 2 } ]$ , where $( h _ { 1 } , w _ { 1 } ) \in [ 0 , H ] \times [ 0 , W ]$ and $( h _ { 2 } , w _ { 2 } ) \in ( h _ { 1 } , H ] \times$ $( w _ { 1 } , W ]$ indicate the top-left and bottom-right coordinates of the area, respectively. During fine-tuning, we sample patches from the image with different $\mathbf { v }$ and resize the patches to resolution $s p$ . We denote the resulted patch as $\mathcal { I } ( F ( \mathbf { v } ) ) \in \mathbb { R } ^ { s p \times s p \times 3 }$ , where $F$ is the normalization and Fourier embedding [16] of the specific area. The encoded latent code of the patch is $\begin{array} { r } { \mathbf { z _ { v } } = \mathcal { E } ( \mathcal { I } ( F ( \mathbf { v } ) ) ) } \end{array}$ . Our model uses the normalized Fourier embedding as an input to make the model learn the position-content correlation. Formally, our diffusion model is defined as $\hat { \mathbf { \epsilon } } _ { \theta } ( \mathbf { z } _ { t } , t , \tau _ { \theta } ( y ) , F ( \mathbf { v } ) )$ .

After fine-tuning, the model can generate latent code at different resolutions by giving the positional information directly to the model. The arbitrary resolution image editing is conducted by feeding two inputs to the model: the positional embedding of the whole image; and a randomly sampled noisy latent with the dimension corresponding to the resolution we want. When sampling in an arbitrary resolution, the model can still keep the structure of the original image intact (examples in Fig. 4). It is worth noting that with or without the correct position encoding, our framework still naturally permits retargeting, i.e., maintaining the aspect ratio of salient objects, like SinGAN [37], InGAN [38], and Drop-the-GAN [11].

# 4. Experiments

Implementation Details While our method can be generally applied to different frameworks, we implement it based on the recently released text-to-image LDM, Stable Diffusion [32]. The pre-trained model was pre-trained on $5 1 2 \times 5 1 2$ images from LAION dataset [36]. The spatial size of the latent code from the pre-trained model is $6 4 \times 6 4$ .

For patch-based fine-tuning, we randomly crop images to patches with height and width uniformly in the range of $[ 0 . 1 H , H ] \times [ 0 . 1 W , W ]$ and resize them to $5 1 2 \times 5 1 2$ . Experiments are conducted using $1 \times \mathrm { R T X 8 0 0 0 }$ GPU with a batch size of 1. The base learning rate is set to $1 \times 1 0 ^ { - 6 }$ . The number of time steps for the diffusion model, $T$ , is 1000. Experiments without and with patch-based fine-tuning are created after 800 and $1 0 , 0 0 0$ optimization steps, respectively. Unless otherwise noted, we adopt other hyperparameter choices from Stable Diffusion [32], and the results are generated with image resolution $5 1 2 \times 5 1 2$ and with latent dimension $6 4 \times 6 4$ . For sampling parameters, we choose $K = 4 0 0$ and $v = 0 . 7$ .

![](images/6c184746df3682ac0dc1194238ad7f91cc91f353eb8eff9a8ec639e83736c4b5.jpg)  
Figure 3. Editing on single source image from various domains. We employ our method on various images and edit them with two target prompts at $5 1 2 \times 5 1 2$ resolution. We show the wide range of edits our approach can be used, including but not limited to style transfer, content add-on, posture change, breed change, etc.

# 4.1. Qualitative Evaluation

To better understand various approaches, we collect images from a wide range of domains, i.e., free-to-use highresolution images from Flickr2 and Unsplash3. During finetuning, we apply a coarse class descriptor to the content we want to preserve, e.g., dog, cat, castle, etc. After optimization, we edit each image with diverse editing prompts. We randomly generate 4 edit results for each image and editing prompt and choose the best one (such a process is also applied to other comparison methods). Our work shows impressive editing ability when applied to various images with different language guidance.

![](images/cfaebbf52c489fea677722e711cee926d2c0ffe4c232b0422cd689867e1b59a7.jpg)  
Figure 4. Arbitrary resolution editing. Our method achieves higher-resolution image editing without artifacts like duplicates, even on ones that change the height-width ratio drastically.

As presented in Fig. 3, using the model-based classifierfree guidance (Sec. 3.2) enables us to apply various editing via text prompts on the single real images. Each image has two text prompts describing different features we want to edit, e.g., image style, background content, the texture of the content, etc. Our method can edit the related features while keeping the content intact. We further show our editing results on arbitrary resolution generation in Fig. 4. For each source image, we edit it with different prompts at various resolutions. As can be seen, our patch-based fine-tuning schedule (Sec. 3.3) successfully preserves the original portion and geometry features of the single source image, even on highly challenging resolution such as $5 1 2 \times 1 0 2 4$ .

# 4.2. Comparisons

We compare our method to concurrent leading techniques, Textual-Inversion [9] and DreamBooth [33], that can be used for single-image editing. Considering no official implementation has been released for DreamBooth, we adopt an unofficial but well-adopted and highly competitive implementation based on Stable Diffusion [32,43]. We compare these techniques strictly according to the detailed guidance provided with the implementations.

Fig. 5 shows the comparison results. As can be noticed, our method maintains the fidelity of the images while applying changes as desired. Furthermore, our approach has high authenticity and structural integrity even for higherresolution editing. For example, in the last row of Fig. 5, when the target prompt is “... standing on grass”, our method generates results by modifying the texture of the land on which the dog stands with other features intact. However, other methods result in a dramatic change in the structure of the whole image. Moreover, in the second row, when modifying the painting Mona Lisa, both DreamBooth [33] and Textual-Inversion [9] fail to edit the image. Our work also shows clear advantages over the approaches on training-free editings, such as ILVR [6], SDEdit [25], and Prompt-to-prompt [13], with the qualitative comparisons presented in the Appendix.

# 4.3. Ablation Analysis

Patch-based fine-tuning. In Fig. 5, we show the results of editing images in higher resolution when fine-tuned without or with the proposed patch-based fine-tuning technique (w/o pos vs. w/ pos). When sampling at a higher resolution, as in the right part of Fig. 5, the denoising model finetuned without the patch-based training mechanism performs poorly. In the first row, the castle towers get duplicated to meet the resolution, and in the third row, the bench gets stretched disproportionately. In essence, our patch-based fine-tuning technique enables the diffusion model to leverage the super-resolution ability of the decoder and edit images at arbitrary resolution during testing time.

Analysis of model-based classifier-free guidance. We generate editing results by directly sampling from the finetuned model using Eqn. 2. We fine-tune the model without the patch-based schedule for 800 steps to retain more generalization ability. In this case, the model can perfectly reconstruct the source image while preserving as much editing ability as possible. We denote the setting as w/o gudiance. Using the same fine-tuned model, we conduct experiments under model-based classifier-free guidance, which we denote as $w /$ guidance. As shown in Fig. 6, sampling without our model-based classifier-free guidance fails to react to the prompt, while our method can successfully edit images to match the target language guidance. We further analyze two hyper-parameters $K$ and $v$ ) in model-based classifier-free guidance.

Analysis on guidance step $K$ in Sec. 3.2. In Fig. 7, we show our results on editing with different settings of $K$ . We conduct this set of experiments by editing one single image with the same language guidance at $7 6 8 \times 7 6 8$ resolution. We set $v = 0 . 7$ . When $K = 0$ , the model-based classifier-free guidance is applied for each step of the denoising process. Since the generalization ability of the finetuned model is limited, in this case, the model fails to apply the desired property to the single source image. When $K = 1 , 0 0 0$ , the model-based classifier-free guidance is not applied to any step. Thus, the structure of the image is not preserved, and the generated result becomes a random sample of the pre-trained model.

![](images/d8fe12d0bc160b5505f04794d3c3ab8209816d4ff082f00be0e5c96b86de613a.jpg)  
Figure 5. Comparisons of various methods. We compare our method to DreamBooth [33] and Textual-Inversion [9]. On the left part of the figure, we edit at the resolution same as training time. On the right part, we edit the source image at a higher resolution. Our work successfully edits the image as required while preserving the details of the source images. We also compare our method without and with the patch-based fine-tuning mechanism (w/o pos vs. w/ pos). When editing at a fixed resolution, two settings perform equally, while at a higher resolution, the patch-based fine-tuning method successfully prevents artifacts.

We further show the quantitative results Fig. 8a. We repeat the abovementioned procedure over different $K$ and randomly sample 20 images for each $K$ . We calculate two metrics. To understand the editing result, the image fidelity that is measured by the LPIPS [49] distance between the original and the edited image. The text alignment calculated by the CLIP [29] score to understand the alignment between our generated images and target text. As can be seen, the image fidelity drops with the increase of $K$ , indicating more details provided by the pre-trained model instead of the fine-tuned one. The text alignment measurement improves since the more details generated by the pre-trained model, the better editing results align with the target domain. To preserve the edit result’s authenticity and fidelity to the source image, we set $K$ as 400.

Analysis on guidance weight $\boldsymbol { v }$ in Sec. 3.2. We further study the impact of the guidance weight $( v )$ in Fig. 9. We set

$K = 4 0 0$ and resolution as $7 6 8 \times 7 6 8$ for each edit and use the same random seed to generate the result. As can be observed, the value of $v$ controls the fidelity of the edit result. However, since the pre-trained model is trained at the resolution $5 1 2 \times 5 1 2$ , the generated image contains many artifacts. When $v = 1$ , the synthesized image entirely depends on the results from the pre-trained model. Additionally, we conduct quantitative experiments with LPIPS score measuring the image fidelity and CLIP score for the text alignment in Fig. 8b. When $v$ is close to 1, the fidelity decreases while the edited feature decreases. When $v$ is close to 0, the model relies mainly on the fine-tuned model for the output when $t > K$ . However, since the fine-tuned model contains poor generalization ability, there is a significant amount of artifacts in the generated results, which leads to a poor LPIPS score. We choose 0.7 for each edit in this work as a trade-off between fidelity and creativity.

# 4.4. More Editing Tasks

Face manipulation. Our method demonstrates promising editing ability for in-the-wild human faces. As shown in Fig. 10, our approach can edit locally and globally on human faces for various facial manipulation tasks, e.g., image stylization, adding accessories, and age changing.

Content removal. In Fig. 11a, we show the content removal using our method. We fine-tune the pre-trained largescale text-to-image model with the language descriptor as “a $[ * ]$ dog with a flower in mouth”. For sampling, we use text prompts such as “a dog” and “a $[ * ]$ dog” for the pretrained and fine-tuned models. The pre-trained model successfully removes the flower held in the mouth of the dog.

![](images/b714db8074b38d78fd3a68a4dadbfff0a0b5465cda49cb763953e08ba1f89e3d.jpg)  
Figure 6. Analysis of model-based classifier-free guidance. Directly sampling with target text using the fine-tuned model (w/o guidance) fails to generate images corresponding to the text prompt. In contrast, the model-based classifier-free guidance (w/ guidance) can synthesize high-fidelity images.

![](images/0e27b669eec50f11788ab9c07c734f07c156b5ef5a4e6afac82e09bbf6d4a151.jpg)  
Figure 7. Analysis on guidance step $K$ . Varying $K$ , we can decide the steps where the model-based guidance is applied, which controls the details from the source image and edits to be applied.

![](images/c22d44bba197d440ceff0954320426d506574a961f9df4f58f08e1885568f276.jpg)  
Figure 8. Trade-off between fidelity and alignment with target text. We calculate the CLIP score [29] (a higher CLIP score indicates a better alignment between the edit result and target text) and LPIPS score [49] (showing the fidelity between edit results and source image, the higher, the better).

![](images/ab65bfe4504a7aea6670b951a5d98494412720b6dad910f717060a1dfdacfdeb.jpg)  
Figure 9. Analysis of score interpolation. Varying $v$ with the same random seed gets editing results with various qualities.

Style generation. Our method can also be employed to learn the underlying style of an image. As shown in Fig. 11b, the model is fine-tuned with the text, “a painting in the $[ * ]$ style”. When sampling results, we feed the pre-trained model a prompt as “painting of a forest”. The model can successfully synthesize images with the specified content in the style of the given real image.

Style transfer. Our model-based classifier-free guidance can be leveraged to combine multiple models for providing the guidance. We show the result in Fig. 11c by doing a style transfer task with dual-model guidance. We finetune two models using prompts: “picture of a $[ * ]$ dog” and “painting in $[ * ]$ style”. During inference, we give the pretrained model the prompt “painting of a dog” and fine-tuned models with prompts the same as training. With guidance from two separate models, our method can generate images with the content from one and style from the other and achieve stylized generation.

![](images/1bb1c9abf1b0931950f9b60bab1dc7d4eac8ebf1b908595da9538bc633e0ddc8.jpg)  
Figure 10. In-the-wild human face manipulation. We conduct various editing on human face photos, locally or globally. The models are trained and edited at a resolution of $5 1 2 \times 5 1 2$ .

![](images/e277b247a53fdf886c1f6a517c918335094c1754d5f10b73004810ad23d13f2f.jpg)  
Figure 11. More applications. We show how our approach can be applied to various tasks in image editing, such as content removal (a), style generation (b), and style transfer (c).

# 5. Conclusion

This work introduces SINE, a method for single-image editing. With only one image and a brief description of the object in the image, our approach can enable a wide range of editing for arbitrary resolution, followed by the information depicted in the language guidance. To achieve such results, we leverage the pre-trained large-scale text-toimage diffusion model. Specifically, we first fine-tune the pre-trained model with our patch-based fine-tuning method until it overfits the single image. Then, during sampling time, we use the overfitted model to guide the pre-trained diffusion model for image synthesis, which maintains the fidelity of the results while taking advantage of the generalization ability of the pre-trained model. Compared with other methods, our approach has a better geometrical understanding of the image and thus can conduct complex editing to the images besides style transfer.

However, in some cases where confusing editing guidance is given for the diffusion model, e.g., a chair-shaped dog, our method could fail. In cases where drastic changes are to be applied, e.g., changing a dog to a tiger in the same posture, there are also noticeable artifacts. We show more

examples in the Appendix.

One future direction is improving the fidelity of the editing results, which could be achieved by alleviating the overfitting problem of the fine-tuned model.

Acknowledgments. This research has been partially funded by research grants to D. Metaxas through NSF IUCRC CARTA-1747778, 2235405, 2212301, 1951890, 2003874, 2310966, FA9550-23-1-0417, and NIH-5R01HL127661.

# References

[1] Rameen Abdal, Peihao Zhu, John Femiani, Niloy Mitra, and Peter Wonka. Clip2stylegan: Unsupervised extraction of stylegan edit directions. In ACM SIGGRAPH 2022 Conference Proceedings, pages 1–9, 2022. 2   
[2] Rameen Abdal, Peihao Zhu, Niloy J Mitra, and Peter Wonka. Styleflow: Attribute-conditioned exploration of stylegangenerated images using conditional continuous normalizing flows. ACM Transactions on Graphics (ToG), 40(3):1–21, 2021. 2   
[3] Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18208–18218, 2022. 2   
[4] Omer Bar-Tal, Dolev Ofri-Amar, Rafail Fridman, Yoni Kasten, and Tali Dekel. Text2live: Text-driven layered image and video editing. In ECCV, pages 707–723. Springer, 2022. 1   
[5] Lucy Chai, Michael Gharbi, Eli Shechtman, Phillip Isola, and Richard Zhang. Any-resolution training for highresolution image synthesis. ECCV, 2022. 2, 4   
[6] Jooyoung Choi, Sungwon Kim, Yonghyun Jeong, Youngjune Gwon, and Sungroh Yoon. Ilvr: Conditioning method for denoising diffusion probabilistic models. arXiv preprint arXiv:2108.02938, 2021. 2, 6, 13, 14   
[7] Hyungjin Chung, Jeongsol Kim, Michael T Mccann, Marc L Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. arXiv preprint arXiv:2209.14687, 2022. 3   
[8] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In CVPR, pages 12873–12883, 2021. 4   
[9] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 1, 2, 4, 6, 7   
[10] Rinon Gal, Or Patashnik, Haggai Maron, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. Stylegan-nada: Clipguided domain adaptation of image generators. TOG, 41(4):1–13, 2022. 1, 2   
[11] Niv Granot, Ben Feinstein, Assaf Shocher, Shai Bagon, and Michal Irani. Drop the gan: In defense of patches nearest neighbors as single image generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13460–13469, 2022. 4   
[12] Erik Ha¨rko¨nen, Aaron Hertzmann, Jaakko Lehtinen, and Sylvain Paris. Ganspace: Discovering interpretable gan controls. Advances in Neural Information Processing Systems, 33:9841–9850, 2020. 2   
[13] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 1, 2, 4, 6, 13   
[14] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NIPS, 33:6840–6851, 2020. 2,   
[15] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021. 2, 3   
[16] Tero Karras, Miika Aittala, Samuli Laine, Erik H¨ark¨onen, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Alias-free generative adversarial networks. Advances in Neural Information Processing Systems, 34:852–863, 2021. 4   
[17] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. arXiv preprint arXiv:2210.09276, 2022. 2   
[18] Gwanghyun Kim, Taesung Kwon, and Jong Chul Ye. Diffusionclip: Text-guided diffusion models for robust image manipulation. In CVPR, pages 2426–2435, 2022. 1, 2   
[19] Hyunsu Kim, Yunjey Choi, Junho Kim, Sungjoo Yoo, and Youngjung Uh. Exploiting spatial dimensions of latent in gan for real-time image editing. In CVPR, pages 852–861, 2021. 1   
[20] Mingi Kwon, Jaeseok Jeong, and Youngjung Uh. Diffusion models already have a semantic latent space. arXiv preprint arXiv:2210.10960, 2022. 2   
[21] Jason Lee, Kyunghyun Cho, and Douwe Kiela. Countering language drift via visual grounding. In 2019 Conference on Empirical Methods in Natural Language Processing and 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, pages 4385–4395. Association for Computational Linguistics, 2020. 2   
[22] Xihui Liu, Dong Huk Park, Samaneh Azadi, Gong Zhang, Arman Chopikyan, Yuxiao Hu, Humphrey Shi, Anna Rohrbach, and Trevor Darrell. More control for free! image synthesis with semantic diffusion guidance. arXiv preprint arXiv:2112.05744, 2021. 2   
[23] Zhi-Song Liu, Li-Wen Wang, Wan-Chi Siu, and Vicky Kalogeiton. Name your style: An arbitrary artist-aware image style transfer. arXiv preprint arXiv:2202.13562, 2022. 2   
[24] Yuchen Lu, Soumye Singhal, Florian Strub, Aaron Courville, and Olivier Pietquin. Countering language drift with seeded iterated learning. In ICML, pages 6437–6447. PMLR, 2020. 2   
[25] Chenlin Meng, Yang Song, Jiaming Song, Jiajun Wu, JunYan Zhu, and Stefano Ermon. Sdedit: Image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021. 1, 2, 6, 13, 14   
[26] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021. 2   
[27] Byong Mok Oh, Max Chen, Julie Dorsey, and Fr´edo Durand. Image-based modeling and photo editing. In Proceedings of the 28th annual conference on Computer graphics and interactive techniques, pages 433–442, 2001. 2   
[28] Or Patashnik, Zongze Wu, Eli Shechtman, Daniel Cohen-Or, and Dani Lischinski. Styleclip: Text-driven manipulation of stylegan imagery. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2085–2094, 2021. 2   
[29] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763. PMLR, 2021. 2, 7, 8   
[30] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022. 1, 2   
[31] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning, pages 8821–8831. PMLR, 2021. 2   
[32] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjo¨rn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, pages 10684– 10695, 2022. 1, 2, 3, 4, 5, 6, 13, 14   
[33] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. arXiv preprint arXiv:2208.12242, 2022. 1, 2, 3, 4, 6, 7, 13   
[34] Dohoon Ryu and Jong Chul Ye. Pyramidal denoising diffusion probabilistic models. arXiv preprint arXiv:2208.01864, 2022. 2   
[35] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, et al. Photorealistic text-to-image diffusion models with deep language understanding. NIPS, 2022. 1, 2, 3   
[36] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion- $4 0 0 \mathrm { m }$ : Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021. 4   
[37] Tamar Rott Shaham, Tali Dekel, and Tomer Michaeli. Singan: Learning a generative model from a single natural image. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4570–4580, 2019. 4   
[38] Assaf Shocher, Shai Bagon, Phillip Isola, and Michal Irani. Ingan: Capturing and retargeting the” dna” of a natural image. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4492–4501, 2019. 4   
[39] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning, pages 2256–2265. PMLR, 2015. 2   
[40] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021. 2   
[41] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in Neural Information Processing Systems, 32, 2019. 2   
[42] Ming Tao, Hao Tang, Songsong Wu, Nicu Sebe, Xiao-Yuan Jing, Fei Wu, and Bingkun Bao. Df-gan: Deep fusion generative adversarial networks for text-to-image synthesis. arXiv preprint arXiv:2008.05865, 2020. 2   
[43] Xavierxiao. Xavierxiao/dreambooth-stablediffusion: Implementation of dreambooth (https://arxiv.org/abs/2208.12242) with stable diffusion. 6   
[44] Rui Xu, Xintao Wang, Kai Chen, Bolei Zhou, and Chen Change Loy. Positional encoding as spatial inductive bias in gans. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13569– 13578, 2021. 2   
[45] Tao Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, and Xiaodong He. Attngan: Finegrained text to image generation with attentional generative adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1316– 1324, 2018. 2   
[46] Hui Ye, Xiulong Yang, Martin Takac, Rajshekhar Sunderraman, and Shihao Ji. Improving text-to-image synthesis using contrastive learning. arXiv preprint arXiv:2107.02423, 2021. 2   
[47] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2022. 1, 2   
[48] Han Zhang, Jing Yu Koh, Jason Baldridge, Honglak Lee, and Yinfei Yang. Cross-modal contrastive learning for text-toimage generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 833–842, 2021. 2   
[49] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 7, 8   
[50] Jiapeng Zhu, Yujun Shen, Deli Zhao, and Bolei Zhou. Indomain gan inversion for real image editing. In ECCV, pages 592–608. Springer, 2020. 1   
[51] Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros. Unpaired image-to-image translation using cycleconsistent adversarial networks. In ICCV, pages 2223–2232, 2017. 1

[52] Minfeng Zhu, Pingbo Pan, Wei Chen, and Yi Yang. Dm-gan: Dynamic memory generative adversarial networks for textto-image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5802–5810, 2019. 2

# Appendix

In the Appendix, we provide the following:

• More comparisons with exiting works on editing single real image (Appendix A).   
• More results for applying SINE on editing a single image and the novel image manipulation tasks that can be enabled by our approach (Appendix B).   
• More ablation analysis (Appendix C).   
• Discussion about the limitation of our method and possible future work (Appendix D).

# A. More Comparisons

Besides the comparison with existing works shown in the main paper, we provide more results by comparing our approach with Prompt-to-Promt [13]. In addition, we compare our methods with training-free single-image editing approaches, including SDEdit [25] and ILVR [6].

We first show the technical differences between our works and training-free methods in Tab. 1. SDEdit [25] applies the diffusion process on an image or a user-created semantic map to conduct the denoising procedure, conditioned with the desired output. ILVR [6] guides the denoising process by replacing the low-frequency part of the sample with that of the target reference image.

The visual comparisons are illustrated in Fig. 12. As can be seen, our approach significantly outperforms other methods for generating high-fidelity images with the maximal keeping of the details in the source image.

# B. More Editing Results

We provide more editing results in Fig. 13, Fig. 14, Fig. 15, Fig. 16, Fig. 17, and Fig. 18. All results are obtained by fine-tuning the large-scale text-to-image model [32] using our proposed patch-based method at the resolution of $5 1 2 \times 5 1 2$ and sampling with our introduced model-based classifier-free guidance at a higher resolution, e.g., $7 6 8 \times 1 0 2 4$ . Images on the top-left corner of these results are the real images utilized for fine-tuning. We specify the hyper-parameters used during sampling in the caption of each image.

# C. More Ablations

Analysis on guidance step $K$ and guidance weight $v$ . We conduct experiments by varying the guidance step $K$ and guidance weight $v$ in Fig. 19, Fig. 20, and Fig. 21. We use the same random seed and generate results with specific text prompts at a fixed resolution by varying the parameters. These experiments show the same behavior of our approach as mentioned in $\sec$ . By adjusting these two parameters, we can find an optimal combination specifically for the image and the target language guidance. In most cases, we adopt the parameters setting of $K = 4 0 0$ and $v = 0 . 7$ . However, we want our model to maintain more fidelity or apply a stronger edit in some instances. For example, the “optimal” setting we decide for experiments in Fig. 19 is $v = 0 . 5$ and $K = 4 0 0$ .

![](images/66b29b9c163119cccc94689887da5e1e38deae8176bc7e96b7374717d9eba2e7.jpg)  
Figure 12. Comparison results. We compare our method with ILVR [6], SDEdit [25], and Prompt-to-Prompt [13] on editing single real image. Note that when the hyper-parameter $N$ is set to 1, the process of ILVR is equivalent to stochastic SDEdit. We adopt the official implementation of ILVR for conducting experiments on SDEdit. For ILVR, we set downsample ratio of $N = 8$ . For SDEdit, we use the scholastic $\mathsf { q }$ sample. In both cases, we set $K = 4 0 0$ .

Analysis on regularization loss. Dreambooth [33] proposes to leverage Prior-Preservation Loss(PPL) to address the issues of overfitting and language drift. They propose to generate 200 samples with the pre-trained model using the prompt “a [class noun]”. Then, during fine-tuning, they use these samples to regulate the model with the PriorPreservation Loss to maintain the generalization ability of the model. However, in our experiments, as shown in Fig. 22, this loss does not improve the final results due to the uniqueness of certain pictures/paintings. On the contrary, more artifacts are introduced to the results, and the fidelity of the editing results decreases. Therefore, given the motivation of editing unique images, we forfeit the generalization ability provided by regularizing the model with the samples generated by the pre-trained model. We encourage our model to overfit a single image for the fidelity of the editing results.

# D. Limitations

We present some failure cases in Fig. 23. As mentioned in the main paper, when confusing guidance is given to the model or drastic change is to be applied, our method produces unsatisfying results. The language comprehension limitation of the pre-trained model and the over-fitting issue of our fine-tuned model can cause this. It would be an interesting future direction to explore how to over-fit on one single image without “forgetting” prior knowledge.

Also, as can be noticed in the second row of Fig. 10, the color of the sweater is changed in most cases. Also, the background letters are twisted after editing. Even though our method can perform editing with maximal protection of the details in the source image, editing strictly on a specific part of an image is also worth further exploration.

Table 1. The differences between our approach and other training-free methods for single image editing.   

<html><body><table><tr><td>Guidance</td><td>Finetune</td><td>Compatible w/LDM [32]</td><td>Position Control</td><td>Admits Multiple Inputs</td></tr><tr><td>SINE (Ours)</td><td>Required</td><td>√</td><td>√</td><td>√</td></tr><tr><td>ILVR [6]</td><td>Not Required</td><td>×</td><td>X</td><td>×</td></tr><tr><td>SDEdit [25]</td><td>Not Required</td><td>√</td><td>×</td><td>X</td></tr></table></body></html>

![](images/1531134750203521429cacd915612b89b4cc368f94ee11bb655e7dc0e8c03ac8.jpg)  
Figure 13. A children’s painting of a castle. The generation resolution is set to $H = 7 6 8$ and $W = 1 0 2 4$ . We use $K = 4 0 0$ and $v = 0 . 7$ in this sample.

![](images/3d6ec64bfe45f4a650f316da6bccfa8e3108ebb1c2b465a470cf8fd0e893c805.jpg)  
Figure 14. A painting of a castle in the style of Claude Monet. The output resolution is set to $H = 7 6 8$ and $W = 1 0 2 4$ . We use $K = 4 0 0$ and $v = 0 . 6 5$ in this example.

![](images/7b49f5530437aae072f7aa70faf3fd6034a904066370c70b00845cd13ccc8efe.jpg)  
Figure 15. A photo of a lake with many sailboats. The output resolution is set to $H = 7 6 8$ and $W = 1 0 2 4$ . We use $K = 4 0 0$ and $v = 0 . 7$ in this case.

![](images/4641014e60017816aff985f26e6f8795daeb55d975f5ab51042194da9a897e17.jpg)  
Figure 16. A desert. The output resolution is set to $H = 7 6 8$ and $W = 1 0 2 4$ . We use $K = 5 0 0$ and $v = 0 . 8$ in this case.

![](images/7632ee28c05793255658d08b305db82043893eb5304f08b32c5f1496bab5329f.jpg)  
Figure 17. A desert. The output resolution is set to $H = 7 6 8$ and $W = 1 0 2 4$ . We use $K = 5 0 0$ and $v = 0 . 8$ in this case.

![](images/ab8b588db6a8c8070ccd796ef44e5f65235f13ec5198e6a3278ea5bdf348c845.jpg)  
Figure 18. A watercolor painting of a girl. The output resolution is set to $H = 1 0 2 4$ and $W = 7 6 8$ . We use $K = 4 0 0$ and $v = 0 . 6$ in this case.

![](images/637f35fb752abf74a29fad653cfeccdf474a330f6a5a11d8304a459387a33a65.jpg)  
Figure 19. “A sculpture of a girl” with the resolution of $H = 6 4 0$ and $W = 5 1 2$ .

![](images/9e37c0b935ed6c78fec3e23def4c8e2600e8c89c244eafe173b5812ecb8bc464.jpg)  
Figure 20. “A coffee machine in the shape of a dog” with the resolution of $H = 5 1 2$ and $W = 5 1 2$ .

![](images/e090d9a3a60b81e6b1350154eef660dca85b966d25846e31a654c533175b6c77.jpg)  
Figure 21. “A castle covered by snow” with the resolution of $H = 5 1 2$ and $W = 7 6 8$ .

Source Image w/o PPL w/ PPL   
H号 雨河鹏朗室 “Painting of a castle in the style of Vincent Van Gogh” $( H = 5 1 2$ , $W = 5 1 2 ^ { \circ }$ ) “Oil painting of a lady in the style of PierreAuguste Renoir” $\cdot H = 5 1 2$ , $W = 5 1 2 \AA$ ) 福 “A dog standing on grass” $\cdot H = 5 1 2$ , $W = 5 1 2$ ) Figure 22. Analysis of Prior-Preservation Loss (PPL). Source Image “chair in the “tiger in the shape of a dog” shape of a dog”   
聚气商尚书 Source Image “cfaisrtelew-osrhka”ped “iceberg castle”

Figure 23. Failure cases. We showcase where our method fails to generate results with high fidelity and text alignment.