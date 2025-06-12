# StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery

Or Patashnik†\* Zongze Wu‡∗ Eli Shechtman§ Daniel Cohen-Or† Dani Lischinski‡ ‡Hebrew University of Jerusalem †Tel-Aviv University §Adobe Research

![](images/aa459a1d1375612cc10f8d272b52b6483a23674cf55179cb1d359584bb1d97b2.jpg)  
Figure 1. Examples of text-driven manipulations using StyleCLIP. Top row: input images; Bottom row: our manipulated results. The text prompt used to drive each manipulation appears under each column.

# Abstract

Inspired by the ability of StyleGAN to generate highly realistic images in a variety of domains, much recent work has focused on understanding how to use the latent spaces of StyleGAN to manipulate generated and real images. However, discovering semantically meaningful latent manipulations typically involves painstaking human examination of the many degrees of freedom, or an annotated collection of images for each desired manipulation. In this work, we explore leveraging the power of recently introduced Contrastive Language-Image Pre-training (CLIP) models in order to develop a text-based interface for StyleGAN image manipulation that does not require such manual effort. We first introduce an optimization scheme that utilizes a CLIPbased loss to modify an input latent vector in response to a user-provided text prompt. Next, we describe a latent mapper that infers a text-guided latent manipulation step for a given input image, allowing faster and more stable textbased manipulation. Finally, we present a method for mapping a text prompts to input-agnostic directions in StyleGAN’s style space, enabling interactive text-driven image manipulation. Extensive results and comparisons demonstrate the effectiveness of our approaches.

# 1. Introduction

Generative Adversarial Networks (GANs) [12] have revolutionized image synthesis, with recent style-based generative models [18, 19, 16] boasting some of the most realistic synthetic imagery to date. Furthermore, the learnt intermediate latent spaces of StyleGAN have been shown to possess disentanglement properties [6, 40, 13, 45, 50], which enable utilizing pretrained models to perform a wide variety of image manipulations on synthetic, as well as real, images.

Harnessing StyleGAN’s expressive power requires developing simple and intuitive interfaces for users to easily carry out their intent. Existing methods for semantic control discovery either involve manual examination (e.g., [13, 40, 50]), a large amount of annotated data, or pretrained classifiers [41, 1]. Furthermore, subsequent manipulations are typically carried out by moving along a direction in one of the latent spaces, using a parametric model, such as a 3DMM in StyleRig [45], or a trained normalized flow in StyleFlow [1]. Specific edits, such as virtual try-on [21] and aging [2] have also been explored.

Thus, existing controls enable image manipulations only along preset semantic directions, severely limiting the user’s creativity and imagination. Whenever an additional, unmapped, direction is desired, further manual effort and/or large quantities of annotated data are necessary.

In this work, we explore leveraging the power of recently introduced Contrastive Language-Image Pre-training (CLIP) models in order to enable intuitive text-based semantic image manipulation that is neither limited to preset manipulation directions, nor requires additional manual effort to discover new controls. The CLIP model is pretrained on 400 million image-text pairs harvested from the Web, and since natural language is able to express a much wider set of visual concepts, combining CLIP with the generative power of StyleGAN opens fascinating avenues for image manipulation. Figures 1 shows several examples of unique manipulations produced using our approach. Specifically, in this paper we investigate three techniques that combine CLIP with StyleGAN:

1. Text-guided latent optimization, where a CLIP model is used as a loss network [14]. This is the most versatile approach, but it requires a few minutes of optimization to apply a manipulation to an image.   
2. A latent residual mapper, trained for a specific text prompt. Given a starting point in latent space (the input image to be manipulated), the mapper yields a local step in latent space.   
3. A method for mapping a text prompt into an inputagnostic (global) direction in StyleGAN’s style space, providing control over the manipulation strength as well as the degree of disentanglement.

The results in this paper and the supplementary material demonstrate a wide range of semantic manipulations on images of human faces, animals, cars, and churches. These manipulations range from abstract to specific, and from extensive to fine-grained. Many of them have not been demonstrated by any of the previous StyleGAN manipulation works, and all of them were easily obtained using a combination of pretrained StyleGAN and CLIP models.

# 2. Related Work

# 2.1. Vision and Language

Joint representations Multiple works learn cross-modal Vision and language (VL) representations [8, 39, 44, 29, 24, 43, 23, 4, 26] for a variety of tasks, such as languagebased image retrieval, image captioning, and visual question answering. Following the success of BERT [9] in various language tasks, recent VL methods typically use Transformers [47] to learn the joint representations. A recent model, based on Contrastive Language-Image Pre-training (CLIP) [34], learns a multi-modal embedding space, which may be used to estimate the semantic similarity between a given text and an image. CLIP was trained on 400 million text-image pairs, collected from a variety of publicly available sources on the Internet. The representations learned by

CLIP have been shown to be extremely powerful, enabling state-of-the-art zero-shot image classification on a variety of datasets. We refer the reader to OpenAI’s Distill article [11] for an extensive exposition and discussion of the visual concepts learned by CLIP.

Text-guided image generation and manipulation The pioneering work of Reed et al. [37] approached text-guided image generation by training a conditional GAN [30], conditioned by text embeddings obtained from a pretrained encoder. Zhang et al. [54, 55] improved image quality by using multi-scale GANs. AttnGAN [52] incorporated an attention mechanism between the text and image features. Additional supervision was used in other works [37, 25, 20] to further improve the image quality.

A few studies focus on text-guided image manipulation. Some methods [10, 31, 27] use a GAN-based encoderdecoder architecture, to disentangle the semantics of both input images and text descriptions. ManiGAN [22] introduces a novel text-image combination module, which produces high-quality images. Differently from the aforementioned works, we propose a single framework that combines the high-quality images generated by StyleGAN, with the rich multi-domain semantics learned by CLIP.

Recently, DALL·E [35, 36], a 12-billion parameter version of GPT-3 [3], which at 16-bit precision requires over 24GB of GPU memory, has shown a diverse set of capabilities in generating and applying transformations to images guided by text. In contrast, our approach is deployable even on a single commodity GPU.

A concurrent work to ours, TediGAN [51], also uses StyleGAN for text-guided image generation and manipulation. By training an encoder to map text into the StyleGAN latent space, one can generate an image corresponding to a given text. To perform text-guided image manipulation, TediGAN encodes both the image and the text into the latent space, and then performs style-mixing to generate a corresponding image. In Section 7 we demonstrate that the manipulations achieved using our approach reflect better the semantics of the driving text.

In a recent online post, Perez [33] describes a text-toimage approach that combines StyleGAN and CLIP in a manner similar to our latent optimizer in Section 4. Rather than synthesizing an image from scratch, our optimization scheme, as well as the other two approaches described in this work, focus on image manipulation. While text-toimage generation is an intriguing and challenging problem, we believe that the image manipulation abilities we provide constitute a more useful tool for the typical workflow of creative artists.

# 2.2. Latent Space Image Manipulation

Many works explore how to utilize the latent space of a pretrained generator for image manipulation [6, 45, 50].

Specifically, the intermediate latent spaces in StyleGAN have been shown to enable many disentangled and meaningful image manipulations. Some methods learn to perform image manipulation in an end-to-end fashion, by training a network that encodes a given image into a latent representation of the manipulated image [32, 38, 2]. Other methods aim to find latent paths, such that traversing along them result in the desired manipulation. Such methods can be categorized into: (i) methods that use image annotations to find meaningful latent paths [40, 1], and (ii) methods that find meaningful directions without supervision, and require manual annotation for each direction [13, 42, 48, 49].

While most works perform image manipulations in the $\boldsymbol { \mathcal { W } }$ or $w +$ spaces, Wu et al. [50] proposed to use the StyleSpace $s$ , and showed that it is better disentangled than $\mathcal { W }$ and $\mathcal { W } +$ . Our latent optimizer and mapper work in the $\mathcal { W } +$ space, while the input-agnostic directions that we detect are in $s$ . In all three, the manipulations are derived directly from text input, and our only source of supervision is a pretrained CLIP model. As CLIP was trained on hundreds of millions of text-image pairs, our approach is generic and can be used in a multitude of domains without the need for domain- or manipulation-specific data annotation.

# 3. StyleCLIP Text-Driven Manipulation

In this work we explore three ways for text-driven image manipulation, all of which combine the generative power of StyleGAN with the rich joint vision-language representation learned by CLIP.

We begin in Section 4 with a simple latent optimization scheme, where a given latent code of an image in StyleGAN’s $\mathcal { W } +$ space is optimized by minimizing a loss computed in CLIP space. The optimization is performed for each (source image, text prompt) pair. Thus, despite it’s versatility, several minutes are required to perform a single manipulation, and the method can be difficult to control. A more stable approach is described in Section 5, where a mapping network is trained to infer a manipulation step in latent space, in a single forward pass. The training takes a few hours, but it must only be done once per text prompt. The direction of the manipulation step may vary depending on the starting position in $\mathcal { W } +$ , which corresponds to the input image, and thus we refer to this mapper as local.

Our experiments with the local mapper reveal that, for a wide variety of manipulations, the directions of the manipulation step are often similar to each other, despite different starting points. Also, since the manipulation step is performed in $\mathcal { W } +$ , it is difficult to achieve fine-grained visual effects in a disentangled manner. Thus, in Section 6 we explore a third text-driven manipulation scheme, which transforms a given text prompt into an input agnostic (i.e., global in latent space) mapping direction. The global direction is computed in StyleGAN’s style space $s$ [50], which

<html><body><table><tr><td></td><td>pre- proc.</td><td>train time</td><td>infer. time</td><td>input image dependent</td><td>latent space</td></tr><tr><td>optimizer</td><td>二</td><td>1</td><td>98 sec</td><td>yes</td><td>W+</td></tr><tr><td>mapper</td><td>1</td><td>10-12h</td><td>75 ms</td><td>yes</td><td>W+</td></tr><tr><td>global dir.</td><td>4h</td><td>1</td><td>72 ms</td><td>no</td><td>S</td></tr></table></body></html>

Table 1. Our three methods for combining StyleGAN and CLIP. The latent step inferred by the optimizer and the mapper depends on the input image, but the training is only done once per text prompt. The global direction method requires a one-time preprocessing, after which it may be applied to different (image, text prompt) pairs. Times are for a single NVIDIA GTX 1080Ti GPU.

is better suited for fine-grained and disentangled visual manipulation, compared to $\mathcal { W } +$ .

Table 1 summarizes the differences between the three methods outlined above, while visual results and comparisons are presented in the following sections.

# 4. Latent Optimization

A simple approach for leveraging CLIP to guide image manipulation is through direct latent code optimization. Specifically, given a source latent code $w _ { s } \in \mathcal { W } +$ , and a directive in natural language, or a text prompt $t$ , we solve the following optimization problem:

$$
\operatorname* { a r g m i n } _ { w \in \mathcal { W } + } D _ { \mathrm { C L I P } } ( G ( w ) , t ) + \lambda _ { \mathrm { L 2 } } \| w - w _ { s } \| _ { 2 } + \lambda _ { \mathrm { I D } } \mathcal { L } _ { \mathrm { I D } } ( w ) ,
$$

where $G$ is a pretrained StyleGAN1 generator and $D _ { \mathrm { C L I P } }$ is the cosine distance between the CLIP embeddings of its two arguments. Similarity to the input image is controlled by the $L _ { 2 }$ distance in latent space, and by the identity loss [38]:

$$
\mathcal { L } _ { \mathrm { I D } } \left( w \right) = 1 - \left. R ( G ( w _ { s } ) ) , R ( G ( w ) ) \right. ,
$$

where $R$ is a pretrained ArcFace [7] network for face recognition, and $\langle \cdot , \cdot \rangle$ computes the cosine similarity between it’s arguments. We solve this optimization problem through gradient descent, by back-propagating the gradient of the objective in (1) through the pretrained and fixed StyleGAN generator $G$ and the CLIP image encoder.

In Figure 3 we provide several edits that were obtained using this optimization approach after 200-300 iterations. The input images were inverted by e4e [46]. Note that visual characteristics may be controlled explicitly (beard, blonde) or implicitly, by indicating a real or a fictional person (Beyonce, Trump, Elsa). The values of $\lambda _ { \mathrm { L } 2 }$ and $\lambda _ { \mathrm { I D } }$ depend on the nature of the desired edit. For changes that shift towards another identity, $\lambda _ { \mathrm { I D } }$ is set to a lower value.

# 5. Latent Mapper

The latent optimization described above is versatile, as it performs a dedicated optimization for each (source image, text prompt) pair. On the downside, several minutes of optimization are required to edit a single image, and the method is somewhat sensitive to the values of its parameters. Below, we describe a more efficient process, where a mapping network is trained, for a specific text prompt $t$ , to infer a manipulation step $M _ { t } ( w )$ in the $\mathcal { W } +$ space, for any given latent image embedding $w \in \mathcal { W } +$ .

![](images/2e605091b5ccf97e2c677baa8aedac592097a2fc936896b575eb550e6d0eafef.jpg)  
Figure 2. The architecture of our text-guided mapper (using the text prompt “surprised”, in this example). The source image (left) is inverted into a latent code $\boldsymbol { w }$ . Three separate mapping functions are trained to generate residuals (in blue) that are added to $w$ to yield the target code, from which a pretrained StyleGAN (in green) generates an image (right), assessed by the CLIP and identity losses.

<html><body><table><tr><td></td><td>Mohawk</td><td>Afro</td><td>Bob-cut</td><td>Curly</td><td>Beyonce</td><td>Taylor Swift</td><td> Surprised</td><td>Purple hair</td></tr><tr><td>Mean</td><td>0.82</td><td>0.84</td><td>0.82</td><td>0.84</td><td>0.83</td><td>0.77</td><td>0.79</td><td>0.73</td></tr><tr><td>Std</td><td>0.096</td><td>0.085</td><td>0.095</td><td>0.088</td><td>0.081</td><td>0.107</td><td>0.893</td><td>0.145</td></tr></table></body></html>

Table 2. Average cosine similarity between manipulation directions obtained from mappers trained using differnt text prompts

![](images/306614bb81893fe6f9108f4b44ed2b2416f6c2843a1153fed04acd6499391fc7.jpg)  
Figure 3. Edits of real celebrity portraits obtained by latent optimization. The driving text prompt and the $( \lambda _ { \mathrm { L } 2 } , \lambda _ { \mathrm { I D } } )$ parameters for each edit are indicated under the corresponding result.

Architecture The architecture of our text-guided mapper is depicted in Figure 2. It has been shown that different StyleGAN layers are responsible for different levels of detail in the generated image [18]. Consequently, it is common to split the layers into three groups (coarse, medium, and fine), and feed each group with a different part of the (extended) latent vector. We design our mapper accordingly, with three fully-connected networks, one for each group/part. The architecture of each of these networks is the same as that of the StyleGAN mapping network, but with fewer layers (4 rather than 8, in our implementation). Denoting the latent code of the input image as $w = ( w _ { c } , w _ { m } , w _ { f } )$ , the mapper is defined by

$$
M _ { t } ( w ) = ( M _ { t } ^ { c } ( w _ { c } ) , M _ { t } ^ { m } ( w _ { m } ) , M _ { t } ^ { f } ( w _ { f } ) ) .
$$

Note that one can choose to train only a subset of the three mappers. There are cases where it is useful to preserve some attribute level and keep the style codes in the corresponding entries fixed.

Losses Our mapper is trained to manipulate the desired attributes of the image as indicated by the text prompt $t$ , while preserving the other visual attributes of the input image. The CLIP loss, $\mathcal { L } _ { \mathrm { C L I P } } ( w )$ guides the mapper to minimize the cosine distance in the CLIP latent space:

$$
\mathcal { L } _ { \mathrm { C L I P } } ( w ) = D _ { \mathrm { C L I P } } ( G ( w + M _ { t } ( w ) ) , t ) ,
$$

where $G$ denotes again the pretrained StyleGAN generator. To preserve the visual attributes of the original input image, we minimize the $L _ { 2 }$ norm of the manipulation step in the latent space. Finally, for edits that require identity preservation, we use the identity loss defined in eq. (2). Our total loss function is a weighted combination of these losses:

$$
\mathcal { L } ( w ) = \mathcal { L } _ { \mathrm { C L I P } } ( w ) + \lambda _ { L 2 } \left. M _ { t } ( w ) \right. _ { 2 } + \lambda _ { \mathrm { I D } } \mathcal { L } _ { \mathrm { I D } } ( w ) .
$$

As before, when the edit is expected to change the identity, we do not use the identity loss. The parameter values we use for the examples in this paper are $\lambda _ { \mathrm { L 2 } } = 0 . 8 , \lambda _ { \mathrm { I D } } = 0 . 1$ , except for the “Trump” manipulation in Figure 9, where the parameter values we use are $\lambda _ { \mathrm { L } 2 } = 2 , \lambda _ { \mathrm { I D } } = 0$ .

![](images/04007555a9f8b0561ae07e3ef65dd12fabf0056d0525c85272ceecc08946fda7.jpg)  
Figure 4. Hair style edits using our mapper. The driving text prompts are indicated below each column. All input images are inversions of real images.

![](images/6861f0728f98f97782a8dd8712138d970b354593906a350a640048888a6f6da4.jpg)  
Figure 5. Controlling more than one attribute with a single mapper. The driving text for each mapper is indicated below each column.

In Figure 4 we provide several examples for hair style edits, where a different mapper used in each column. In all of these examples, the mapper succeeds in preserving the identity and most of the other visual attributes that are not related to hair. Note, that the resulting hair appearance is adapted to the individual; this is particularly apparent in the “Curly hair” and “Bob-cut hairstyle” edits.

It should be noted that the text prompts are not limited to a single attribute at a time. Figure 5 shows four different combinations of hair attributes, straight/curly and short/long, each yielding the expected outcome. This degree of control has not been demonstrated by any previous method we’re aware of.

Since the latent mapper infers a custom-tailored manipulation step for each input image, it is interesting to examine the extent to which the direction of the step in latent space varies over different inputs. To test this, we first invert the test set of CelebA-HQ [28, 15] using e4e [46]. Next, we feed the inverted latent codes into several trained mappers and compute the cosine similarity between all pairs of the resulting manipulation directions. The mean and the standard deviation of the cosine similarity for each mapper is reported in Table 2. The table shows that even though the mapper infers manipulation steps that are adapted to the input image, in practice, the cosine similarity of these steps for a given text prompt is high, implying that their directions are not as different as one might expect.

# 6. Global Directions

While the latent mapper allows fast inference time, we find that it sometimes falls short when a fine-grained disentangled manipulation is desired. Furthermore, as we have seen, the directions of different manipulation steps for a given text prompt tend to be similar. Motivated by these observations, in this section we propose a method for mapping a text prompt into a single, global direction in StyleGAN’s style space $s$ , which has been shown to be more disentangled than other latent spaces [50].

Let $s \in \mathcal S$ denote a style code, and $G ( s )$ the corresponding generated image. Given a text prompt indicating a desired attribute, we seek a manipulation direction $\Delta s$ , such that $G ( s + \alpha \Delta s )$ yields an image where that attribute is introduced or amplified, without significantly affecting other attributes. The manipulation strength is controlled by $\alpha$ . Our high-level idea is to first use the CLIP text encoder to obtain a vector $\Delta t$ in CLIP’s joint language-image embedding and then map this vector into a manipulation direction $\Delta s$ in $s$ . A stable $\Delta t$ is obtained from natural language, using prompt engineering, as described below. The corresponding direction $\Delta s$ is then determined by assessing the relevance of each style channel to the target attribute.

More formally, denote by $\boldsymbol { \mathcal { T } }$ the manifold of image embeddings in CLIP’s joint embedding space, and by $\tau$ the manifold of its text embeddings. We distinguish between these two manifolds, because there is no one-to-one mapping between them: an image may contain a large number of visual attributes, which can hardly be comprehensively described by a single text sentence; conversely, a given sentence may describe many different images. During CLIP training, all embeddings are normalized to a unit norm, and therefore only the direction of embedding contains semantic information, while the norm may be ignored. Thus, in well trained areas of the CLIP space, we expect directions on the $\tau$ and $\boldsymbol { \mathcal { T } }$ manifolds that correspond to the same semantic changes to be roughly collinear (i.e., have large cosine similarity), and nearly identical after normalization.

Given a pair of images, $G ( s )$ and $G ( s + \alpha \Delta s )$ , we denote their $\boldsymbol { \mathcal { T } }$ embeddings by $i$ and $i + \Delta i$ , respectively. Thus, the difference between the two images in CLIP space is given by $\Delta i$ . Given a natural language instruction encoded as $\Delta t$ , and assuming collinearity between $\Delta t$ and $\Delta i$ , we can determine a manipulation direction $\Delta s$ by assessing the relevance of each channel in $s$ to the direction $\Delta i$ .

From natural language to $\Delta t$ In order to reduce text embedding noise, Radford et al. [34] utilize a technique called prompt engineering that feeds several sentences with the same meaning to the text encoder, and averages their embeddings. For example, for ImageNet zero-shot classification, a bank of 80 different sentence templates is used, such as “a bad photo of a $\{ \} ^ { \ast }$ , “a cropped photo of the $\{ \} ^ { \ast }$ , “a black and white photo of a $\{ \} ^ { \ast }$ , and “a painting of a $\{ \} ^ { \ast }$ . At inference time, the target class is automatically substituted into these templates to build a bank of sentences with similar semantics, whose embeddings are then averaged. This process improves zero-shot classification accuracy by an additional $3 . 5 \%$ over using a single text prompt.

Similarly, we also employ prompt engineering (using the same ImageNet prompt bank) in order to compute stable directions in $\tau$ . Specifically, our method should be provided with text description of a target attribute and a corresponding neutral class. For example, when manipulating images of cars, the target attribute might be specified as “a sports car”, in which case the corresponding neutral class might be “a car”. Prompt engineering is then applied to produce the average embeddings for the target and the neutral class, and the normalized difference between the two embeddings is used as the target direction $\Delta t$ .

Channelwise relevance Next, our goal is to construct a style space manipulation direction $\Delta s$ that would yield a change $\Delta i$ , collinear with the target direction $\Delta t$ . For this purpose, we need to assess the relevance of each channel $c$ of $s$ to a given direction $\Delta i$ in CLIP’s joint embedding space. We generate a collection of style codes $s \in { \mathcal { S } }$ , and perturb only the $c$ channel of each style code by adding a negative and a positive value. Denoting by $\Delta i _ { c }$ the CLIP space direction between the resulting pair of images, the relevance of channel $c$ to the target manipulation is estimated as the mean projection of $\Delta i _ { c }$ onto $\Delta i$ :

$$
R _ { c } ( \Delta i ) = \mathbb { E } _ { s \in { \cal S } } \{ \Delta i _ { c } \cdot \Delta i \}
$$

In practice, we use 100 image pairs to estimate the mean. The pairs of images that we generate are given by $G ( s \pm$ $\alpha \Delta s _ { c , \ - }$ ), where $\Delta s _ { c }$ is a zero vector, except its $c$ coordinate, which is set to the standard deviation of the channel. The magnitude of the perturbation is set to $\alpha = 5$ .

Having estimated the relevance $R _ { c }$ of each channel, we ignore channels whose $R _ { c }$ falls below a threshold $\beta$ . This parameter may be used to control the degree of disentanglement in the manipulation: using higher threshold values results in more disentangled manipulations, but at the same time the visual effect of the manipulation is reduced. Since various high-level attributes, such as age, involve a combination of several lower level attributes (for example, grey hair, wrinkles, and skin color), multiple channels are relevant, and in such cases lowering the threshold value may be preferable, as demonstrated in Figure 6. To our knowledge, the ability to control the degree of disentanglement in this manner is unique to our approach.

![](images/a6370267c6995ca04b2e8019dd2d893059d338982c7a1efb6f5a9f0ef5ea5e5c.jpg)  
Figure 6. Image manipulation driven by the prompt “grey hair” for different manipulation strengths and disentanglement thresholds. Moving along the $\Delta s$ direction, causes the hair color to become more grey, while steps in the $- \Delta s$ direction yields darker hair. The effect becomes stronger as the strength $\alpha$ increases. When the disentanglement threshold $\beta$ is high, only the hair color is affected, and as $\beta$ is lowered, additional correlated attributes, such as wrinkles and the shape of the face are affected as well.

In summary, given a target direction $\Delta i$ in CLIP space, we set

$$
\Delta s = \left\{ \begin{array} { l l } { \Delta i _ { c } \cdot \Delta i \quad } & { \mathrm { i f } \left| \Delta i _ { c } \cdot \Delta i \right| \geq \beta } \\ { 0 \quad } & { \mathrm { o t h e r w i s e } } \end{array} \right.
$$

Figures 7 and 8 show a variety of edits along text-driven manipulation directions determined as described above on images of faces, cars, and dogs. The manipulations in Figure 7 are performed using StyleGAN2 pretrained on FFHQ [18]. The inputs are real images, embedded in $w +$ space using the e4e encoder [46]. The figure demonstrates textdriven manipulations of 18 attributes, including complex concepts, such as facial expressions and hair styles. The manipulations in Figure 8 use StyleGAN2 pretrained on LSUN cars [53] (on real images) and on generated images from StyleGAN2-ada [17] pretrained on AFHQ dogs [5].

# 7. Comparisons and Evaluation

We now turn to compare the three methods presented and analyzed in the previous sections among themselves and to other methods. All the real images that we manipulate are inverted using the e4e encoder [46].

Text-driven image manipulation methods: We begin by comparing several text-driven facial image manipulation methods in Figure 9. We compare between our latent mapper method (Section 5), our global direction method (Section 6), and TediGAN [51]. For TediGAN, we use the authors’ official implementation, which has been recently updated to utilize CLIP for image manipulation, and thus is somewhat different from the method presented in their paper. We do not include results of the optimization method presented in Section 4, since its sensitivity to hyperparameters makes it time-consuming, and therefore not scalable.

![](images/69569195ff629d7ffccfd8bb8b88b329117152fc70419a0ae47bbd4d9af9e39a.jpg)  
Figure 7. A variety of edits along global text-driven manipulation directions, demonstrated on portraits of celebrities. Edits are performed using StyleGAN2 pretrained on FFHQ [18]. The inputs are real images, embedded in $w +$ space using the e4e encoder [46]. The target attribute used in the text prompt is indicated above each column.

![](images/ab58b3143fdc8095294aa61eb693c31b9364c898fed38bd1c1a0415d5aa22983.jpg)  
Figure 8. A variety of edits along global text-driven manipulation directions. Left: using StyleGAN2 pretrained on LSUN cars [53]. Righ using StyleGAN2-ada [17] pretrained on AFHQ dogs [5]. The target attribute used in the text prompt is indicated above each column.

We perform the comparison using three kinds of attributes ranging from complex, yet specific (e.g., “Trump”), less complex and less specific (e.g., “Mohawk”), to simpler and more common (e.g., “without wrinkles”). The complex “Trump” manipulation, involves several attributes such as blonde hair, squinting eyes, open mouth, somewhat swollen face and Trump’s identity. While a global latent direction is able to capture the main visual attributes, which are not specific to Trump, it fails to capture the specific identity. In contrast, the latent mapper is more successful. The “Mohawk hairstyle” is a less complex attribute, as it involves only hair, and it isn’t as specific. Thus, both our methods are able to generate satisfactory manipulations. The manipulation generated by the global direction is slightly less pronounced, since the direction in CLIP space is an average one. Finally, for the “without wrinkles” prompt, the global direction succeeds in removing the wrinkles, while keeping other attributes mostly unaffected, while the mapper fails. We attribute this to $\mathcal { W } +$ being less disentangled. We observed similar behavior on another set of attributes (“Obama”,“Angry”,“beard”). We conclude that for complex and specific attributes (especially those that involve identity), the mapper is able to produce better manipulations. For simpler and/or more common attributes, a global direction suffices, while offering more disentangled manipulations. We note that the results produced by TediGAN fail in all three manipulations shown in Figure 9.

![](images/4c40aeae11a18c7c31f636f5eefbdeee2dd1ff5c41a69bb3798cb6088d6f159b.jpg)  
Figure 9. We compare three methods that utilize StyleGAN and CLIP using three different kinds of attributes.

Other StyleGAN manipulation methods: In Figure 10, we show a comparison between our global direction method and several state-of-the-art StyleGAN image manipulation methods: GANSpace [13], InterFaceGAN [41], and StyleSpace [50]. The comparison only examines the attributes which all of the compared methods are able to manipulate (Gender, Grey hair, and Lipstick), and thus it does not include the many novel manipulations enabled by our approach. Since all of these are common attributes, we do not include our mapper in this comparison. Following Wu et al. [50], the manipulation step strength is chosen such that it induces the same amount of change in the logit value of the corresponding classifiers (pretrained on CelebA).

It may be seen that in GANSpace [13] manipulation is entangled with skin color and lighting, while in InterFaceGAN [41] the identity may change significantly (when manipulating Lipstick). Our manipulation is very similar to StyleSpace [50], which only changes the target attribute, while all other attributes remain the same.

![](images/5487502bc4093910148bbc9a339dfaf1cddcb121978ff573e11c395099398431.jpg)  
Figure 10. Comparison with state-of-the-art methods using the same amount of manipulation according to a pretrained attribute classifier.

In the supplementary material, we also show a comparison with StyleFLow [1], a state-of-the-art non-linear method. Our method produces results of similar quality, despite the fact that StyleFlow simultaneously uses several attribute classifiers and regressors (from the Microsoft face API), and is thus can manipulate a limited set of attributes. In contrast, our method requires no extra supervision.

Limitations. Our method relies on a pretrained StyleGAN generator and CLIP model for a joint language-vision embedding. Thus, it cannot be expected to manipulate images to a point where they lie outside the domain of the pretrained generator (or remain inside the domain, but in regions that are less well covered by the generator). Similarly, text prompts which map into areas of CLIP space that are not well populated by images, cannot be expected to yield a visual manipulation that faithfully reflects the semantics of the prompt. We have also observed that drastic manipulations in visually diverse datasets are difficult to achieve. For example, while tigers are easily transformed into lions (see Figure 1), we were less successful when transforming tigers to wolves, as demonstrated in the supplementary material.

# 8. Conclusions

We introduced three novel image manipulation methods, which combine the strong generative powers of StyleGAN with the extraordinary visual concept encoding abilities of CLIP. We have shown that these techniques enable a wide variety of unique image manipulations, some of which are impossible to achieve with existing methods that rely on annotated data. We have also demonstrated that CLIP provides fine-grained edit controls, such as specifying a desired hair style, while our method is able to control the manipulation strength and the degree of disentanglement. In summary, we believe that text-driven manipulation is a powerful image editing tool, whose abilities and importance will only continue to grow.

# References

[1] Rameen Abdal, Peihao Zhu, Niloy Mitra, and Peter Wonka. StyleFlow: attribute-conditioned exploration of StyleGANgenerated images using conditional continuous normalizing flows. arXiv preprint arXiv:2008.02401, 2020. 1, 3, 8, 12, 17   
[2] Yuval Alaluf, Or Patashnik, and Daniel Cohen-Or. Only a matter of style: Age transformation using a style-based regression model. arXiv preprint arXiv:2102.02754, 2021. 1, 3   
[3] T. Brown, B. Mann, Nick Ryder, Melanie Subbiah, J. Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, G. Kr¨uger, T. Henighan, R. Child, Aditya Ramesh, D. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, E. Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, J. Clark, Christopher Berner, Sam McCandlish, A. Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. arXiv, abs/2005.14165, 2020. 2   
[4] Yen-Chun Chen, Linjie Li, Licheng Yu, A. E. Kholy, Faisal Ahmed, Zhe Gan, Y. Cheng, and Jing jing Liu. Uniter: Universal image-text representation learning. In ECCV, 2020. 2   
[5] Yunjey Choi, Youngjung Uh, Jaejun Yoo, and Jung-Woo Ha. StarGAN v2: Diverse image synthesis for multiple domains. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8188–8197, 2020. 6, 7, 12, 15, 18   
[6] Edo Collins, Raja Bala, Bob Price, and Sabine Su¨sstrunk. Editing in style: Uncovering the local semantics of GANs. arXiv preprint arXiv:2004.14367, 2020. 1, 2   
[7] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In Proc. CVPR, pages 4690–4699, 2019. 3   
[8] Karan Desai and J. Johnson. VirTex: Learning visual representations from textual annotations. ArXiv, abs/2006.06666, 2020. 2   
[9] J. Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In NAACL-HLT, 2019.   
2 [10] H. Dong, Simiao Yu, Chao Wu, and Y. Guo. Semantic image synthesis via adversarial learning. Proc. ICCV, pages 5707–   
5715, 2017. 2 [11] Gabriel Goh, Nick Cammarata, Chelsea Voss, Shan Carter, Michael Petrov, Ludwig Schubert, Alec Radford, and Chris Olah. Multimodal neurons in artificial neural networks. Distill, https://distill.pub/2021/multimodal-neurons/, 2021. 2 [12] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in neural information processing systems, pages 2672–2680,   
2014. 1 [13] Erik Ha¨rko¨nen, Aaron Hertzmann, Jaakko Lehtinen, and Sylvain Paris. GANSpace: Discovering interpretable GAN controls. arXiv preprint arXiv:2004.02546, 2020. 1, 3, 8, 12 [14] Justin Johnson, Alexandre Alahi, and Li Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. In Proc. ECCV, 2016. 2 [15] Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of GANs for improved quality, stability, and variation. arXiv:1710.10196, 2017. 5 [16] Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Training generative adversarial networks with limited data. In Proc. NeurIPS, 2020. [17] Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Training generative adversarial networks with limited data. arXiv preprint arXiv:2006.06676, 2020. 6, 7, 12, 15, 18 [18] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proc. CVPR, pages 4401–4410, 2019. 1, 4, 6, 7 [19] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of StyleGAN. In Proc. CVPR, pages 8110–   
8119, 2020. 1, 3 [20] J. Y. Koh, Jason Baldridge, H. Lee, and Yinfei Yang. Textto-image generation grounded by fine-grained user attention. arXiv, abs/2011.03775, 2020. 2 [21] Kathleen M Lewis, Srivatsan Varadharajan, and Ira Kemelmacher-Shlizerman. VOGUE: Try-on by StyleGAN interpolation optimization. arXiv:2101.02285, 2021. 1 [22] Bowen Li, Xiaojuan Qi, Thomas Lukasiewicz, and Philip HS Torr. ManiGAN: Text-guided image manipulation. In Proc. CVPR, pages 7880–7889, 2020. 2 [23] Gen Li, N. Duan, Yuejian Fang, Daxin Jiang, and M. Zhou. Unicoder-VL: A universal encoder for vision and language by cross-modal pre-training. In Proc. AAAI, 2020. 2 [24] Liunian Harold Li, Mark Yatskar, Da Yin, C. Hsieh, and KaiWei Chang. Visualbert: A simple and performant baseline for vision and language. ArXiv, abs/1908.03557, 2019. 2 [25] Wenbo Li, Pengchuan Zhang, Lei Zhang, Qiuyuan Huang, X. He, Siwei Lyu, and Jianfeng Gao. Object-driven text-toimage synthesis via adversarial training. Proc. CVPR, pages   
12166–12174, 2019. 2   
[26] Xiujun Li, Xi Yin, C. Li, X. Hu, Pengchuan Zhang, Lei Zhang, Longguang Wang, H. Hu, Li Dong, Furu Wei, Yejin Choi, and Jianfeng Gao. Oscar: Object-semantics aligned pre-training for vision-language tasks. In ECCV, 2020. 2   
[27] Yahui Liu, Marco De Nadai, Deng Cai, Huayang Li, Xavier Alameda-Pineda, N. Sebe, and Bruno Lepri. Describe what to change: A text-guided unsupervised image-to-image translation approach. Proceedings of the 28th ACM International Conference on Multimedia, 2020. 2   
[28] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild, 2015. 5   
[29] Jiasen Lu, Dhruv Batra, D. Parikh, and Stefan Lee. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. In NeurIPS, 2019. 2   
[30] Mehdi Mirza and Simon Osindero. Conditional generative adversarial nets. arXiv:1411.1784, 2014. 2   
[31] Seonghyeon Nam, Yunji Kim, and S. Kim. Text-adaptive generative adversarial networks: Manipulating images with natural language. In NeurIPS, 2018. 2   
[32] Yotam Nitzan, Amit Bermano, Yangyan Li, and Daniel Cohen-Or. Face identity disentanglement via latent space mapping. ACM Trans. Graph., 39(6), Nov. 2020. 3   
[33] Victor Perez. Generating images from prompts using CLIP and StyleGAN. https://towardsdatascience.com/generatingimages-from-prompts-using-clip-and-stylegan1f9ed495ddda, 2021. 2   
[34] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. Image, 2:T2, 2021. 2, 6   
[35] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, and Scott Gray. DALL·E: Creating Images from Text. https://openai.com/blog/dall-e/, 2021. 2   
[36] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. arXiv:2102.12092, 2021. 2   
[37] S. Reed, Zeynep Akata, Xinchen Yan, L. Logeswaran, B. Schiele, and H. Lee. Generative adversarial text to image synthesis. In ICML, 2016. 2   
[38] Elad Richardson, Yuval Alaluf, Or Patashnik, Yotam Nitzan, Yaniv Azar, Stav Shapiro, and Daniel Cohen-Or. Encoding in style: a StyleGAN encoder for image-to-image translation. arXiv:2008.00951, 2020. 3   
[39] Mert Bulent Sariyildiz, Julien Perez, and Diane Larlus. Learning visual representations with caption annotations. arXiv preprint arXiv:2008.01392, 2020. 2   
[40] Yujun Shen, Jinjin Gu, Xiaoou Tang, and Bolei Zhou. Interpreting the latent space of GANs for semantic face editing. In Proc. CVPR, pages 9243–9252, 2020. 1, 3   
[41] Yujun Shen, Ceyuan Yang, Xiaoou Tang, and Bolei Zhou. InterFaceGAN: interpreting the disentangled face representation learned by GANs. arXiv preprint arXiv:2005.09635,   
[42] Yujun Shen and Bolei Zhou. Closed-form factorization of latent semantics in GANs. arXiv preprint arXiv:2007.06600, 2020. 3   
[43] Weijie Su, Xizhou Zhu, Yue Cao, Bin Li, Lewei Lu, Furu Wei, and Jifeng Dai. VL-BERT: Pre-training of generic visual-linguistic representations. In Proc. ICLR, 2020. 2   
[44] Hao Hao Tan and Mohit Bansal. LXMERT: Learning crossmodality encoder representations from transformers. In EMNLP/IJCNLP, 2019. 2   
[45] Ayush Tewari, Mohamed Elgharib, Gaurav Bharaj, Florian Bernard, Hans-Peter Seidel, Patrick P´erez, Michael Zollho¨fer, and Christian Theobalt. StyleRig: Rigging StyleGAN for 3d control over portrait images. arXiv preprint arXiv:2004.00121, 2020. 1, 2   
[46] Omer Tov, Yuval Alaluf, Yotam Nitzan, Or Patashnik, and Daniel Cohen-Or. Designing an encoder for stylegan image manipulation. arXiv preprint arXiv:2102.02766, 2021. 3, 5, 6, 7, 12   
[47] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, 2017. 2   
[48] Andrey Voynov and Artem Babenko. Unsupervised discovery of interpretable directions in the GAN latent space. arXiv preprint arXiv:2002.03754, 2020. 3   
[49] Binxu Wang and Carlos R Ponce. A geometric analysis of deep generative image models and its applications. In Proc. ICLR, 2021. 3   
[50] Zongze Wu, Dani Lischinski, and Eli Shechtman. StyleSpace analysis: Disentangled controls for StyleGAN image generation. arXiv:2011.12799, 2020. 1, 2, 3, 5, 8, 12   
[51] Weihao Xia, Yujiu Yang, Jing-Hao Xue, and Baoyuan Wu. TediGAN: Text-guided diverse face image generation and manipulation. arXiv preprint arXiv: 2012.03308, 2020. 2, 6   
[52] T. Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, and X. He. AttnGAN: Fine-grained text to image generation with attentional generative adversarial networks. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1316–1324, 2018. 2   
[53] Fisher Yu, Ari Seff, Yinda Zhang, Shuran Song, Thomas Funkhouser, and Jianxiong Xiao. Lsun: Construction of a large-scale image dataset using deep learning with humans in the loop. arXiv preprint arXiv:1506.03365, 2015. 6, 7, 15   
[54] Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, and Dimitris N Metaxas. StackGAN: Text to photo-realistic image synthesis with stacked generative adversarial networks. In Proc. ICCV, pages 5907– 5915, 2017. 2   
[55] Han Zhang, T. Xu, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, and Dimitris N. Metaxas. Stack${ \mathrm { G A N } } + +$ : Realistic image synthesis with stacked generative adversarial networks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 41:1947–1962, 2019. 2

# A. Latent Mapper – Ablation Study

In this section, we study the importance of various choices in the design of our latent mapper (Section 5).

# A.1. Architecture

The architecture of the mapper is rather simple and with relatively small number of parameters. Moreover, it has negligible effect on the inference time. Yet, it is natural to compare the presented architecture, which consists of three different mapping networks, to an architecture with a single mapping network. Intuitively, using a separate network for each group of style vector entries should better enable changes at several different levels of detail in the image. Indeed, we find that with driving text that requires such changes, e.g. “Donald Trump”, a single mapping network does not yield results that are as effective as those produced with three. An example is shown in Figure 11.

Although the full, three network mapper, gives better results for some driving texts, as mentioned in Section 5, we note that not all the three are needed when the manipulation should not affect some attributes. For example, for the hairstyle edits shown in Figure 5, the manipulation should not affect the color scheme of the image. Therefore, we perform these edits when training $M ^ { c }$ and $M ^ { m }$ only, that is, $M _ { t } ( w ) = ( M _ { t } ^ { c } ( w _ { c } ) , M _ { t } ^ { m } ( w _ { m } ) , 0 )$ . We show a comparison in Figure 12. As can be seen, by removing $M _ { f }$ from the architecture, we get slightly better results. Therefore, for the sake of simplicity and generalization of the method, we chose to describe the method with all three networks. In the main paper, the results shown were obtained with all three networks, while here we also show results with only

![](images/81c69c740a1aa0cc180f2aa32797f2613f017cdd4b9dc5244bafd76a35169fc0.jpg)  
Figure 11. Comparing our mapper architecture with a simpler architecture that uses a single mapping network. The simpler mapper fails to infer multiple changes correctly. The changes in the expression and in the hair-style are not strong enough to capture the identity of the target individual. On the other hand, there are unnecessary changes in the background color in the second row when using a single network.

![](images/22946cc06689c8333d14a69c8905dc7f47e6853b1ec7708ed73266dcedf16a31.jpg)  
Figure 12. Removing $M ^ { f }$ from our full architecture for edits which do not require color scheme manipulation yields slightly better results.

![](images/993c408ec4efbfcf4b45b317128c204693b17b8c059670fd0340650f900303ab.jpg)  
Figure 13. Replacing the CLIP loss with identity loss for the Beyonce edit. The identity loss is computed with respect to an image of Beyonce.

two (without $M _ { f }$ ).

# A.2. Losses

CLIP Loss To show the uniqueness of using a “celeb edit” with CLIP, we perform the following experiment. Instead of using the CLIP loss, we use the identity loss with respect to a single image of the desired celeb. Specifically, we perform this experiment by using an image of Beyonce. The results are shown in Figure 13. As can be seen, CLIP guides the mapper to perform a unique edit which cannot be achieved by simply using a facial recognition network.

ID Loss Here we show that the identity loss is significant for preserving the identity of the person in the input image. When using the default parameter setting of $\lambda _ { \mathrm { L } 2 } = 0 . 8$ with $\lambda _ { \mathrm { I D } } = 0$ (i.e., no identity loss), we observe that the mapper fails to preserve the identity, and introduces large changes. Therefore, we also experiment with $\lambda _ { \mathrm { L 2 } } ~ = ~ 1 . 6$ , however, this still does not preserve the original identity well enough. The results are shown in Figure 14.

![](images/1fa2bbef504adf3e6004cf8baab68058c2a62ad662781f9350e93d233316931f.jpg)  
Figure 14. Identity loss ablation study. Under each column we specify $( \lambda _ { \mathrm { L } 2 } , \lambda _ { \mathrm { I D } } )$ . In the second and the third columns we did not use the identity loss. As can be seen, the identity of individual in the input image is not preserved.

# B. Additional Results

In this section we provide additional results to those presented in the paper. Specifically, we begin with a variety of image manipulations obtained using our latent mapper. All manipulated images are taken from the CelebA-HQ and were inverted by e4e [46]. In Figure 15 we show a large gallery of hair style manipulations. In Figures 16 and 17 we show “celeb” edits, where the input image is manipulated to resemble a certain target celebrity. In Figure 18 we show a variety of expression edits.

Next, Figure 19 shows a variety of edits on non-face datasets, performed along text-driven global latent manipulation directions (Section 6).

Figure 20 shows image manipulations driven by the prompt “a photo of a male face” for different manipulation strengths and disentanglement thresholds. Moving along the global direction, causes the facial features to become more masculine, while steps in the opposite direction yields more feminine features. The effect becomes stronger as the strength $\alpha$ increases. When the disentanglement threshold $\beta$ is high, only the facial features are affected, and as $\beta$ is lowered, additional correlated attributes, such as hair length and facial hair are affected as well.

In Figure 21, we show another comparison between our global direction method and several state-of-the-art StyleGAN image manipulation methods: GANSpace [13], Inter

FaceGAN [41], and StyleSpace [50]. The comparison only examines the attributes which all of the compared methods are able to manipulate (Gender, Grey hair, and Lipstick), and thus it does not include the many novel manipulations enabled by our approach. Following Wu et al. [50], the manipulation step strength is chosen such that it induces the same amount of change in the logit value of the corresponding classifiers (pretrained on CelebA). It may be seen that in GANSpace [13] manipulation is entangled with skin color and lighting, while in InterFaceGAN [41] the identity may change significantly (when manipulating Lipstick). Our manipulation is very similar to StyleSpace [50], which only changes the target attribute, while all other attributes remain the same.

Figure 22 shows a comparison between StyleFlow [1] and our global directions method. It may be seen that our method is able to produce results of comparable visual quality, despite the fact that StyleFlow requires the simultaneous use of several attribute classifiers and regressors (from the Microsoft face API), and is thus able to manipulate a limited set of attributes. In contrast, our method required no extra supervision to produce these and all of the other manipulations demonstrated in this work.

Figure 23 shows an additional comparison between textdriven manipulation using our global directions method and our latent mapper. Our observations are similar to the ones we made regarding Figure 10 in the main paper.

Finally, Figure 24 demonstrates that drastic manipulations in visually diverse datasets are sometimes difficult to achieve using our global directions. Here we use StyleGAN-ada [17] pretrained on AFHQ wild [5], which contains wolves, lions, tigers and foxes. There is a smaller domain gap between tigers and lions, which mainly involves color and texture transformations. However, there is a larger domain gap between tigers and wolves, which, in addition to color and texture transformations, also involves more drastic shape deformations. This figure demonstrates that our global directions method is more successful in transforming tigers into lions, while failing in some cases to transform tigers to wolves.

# C. Video

We show examples of interactive text-driven image manipulation in our supplementary video. We use a simple heuristic method to determine the initial disentanglement threshold $( \beta )$ . The threshold is chosen such that $k$ channels will be active. For real face manipulation, we set the initial strength to $\alpha = 3$ and the disentanglement threshold so that $k = 2 0$ . For real car manipulation, we set the initial values to $\alpha = 3$ and $k = 1 0 0$ . For generated cat manipulation, we set the initial values to $\alpha = 7$ and $k = 1 0 0$ .

![](images/f67659a63f0fedc8006be6fa8e7b16df0124481249570102d205630a8a2c5ff2.jpg)  
Figure 15. Hair style manipulations obtained by the latent mapper. Except for the purple hair, all mappers were trained without $M ^ { f }$ .

![](images/b27f26b6de12c28650b9b63a46849d3ab5efaf6af0f6b490c11ba32085225c50.jpg)  
Figure 16. Celeb edits performed by the latent mapper.

![](images/b65e9ce093b42696a54920241980ed173a14a85122ba87f45297b7c8f16cdb38.jpg)  
Input Trump Mark Zuckerberg Johnny Depp Figure 17. Celeb edits performed by the latent mapper.

![](images/985b2075e584a33ddcb67eb44e4255e4f3291f227c6ce70e5f1f605210a04e4c.jpg)  
Figure 18. Expression edits performed by the latent mapper.

![](images/b8876387fb08b4bee241c2993d2d76d5faf8baab501bd146b54a6a5a9fdf2b2a.jpg)  
Figure 19. A variety of edits for non-face images along text-driven global latent manipulation directions. Left: using StyleGAN2-ada [17] pretrained on AFHQ cats [5]. Right: using StyleGAN2 pretrained on LSUN Church [53]. The target attribute used in the text prompt is indicated above each column.

![](images/4d675c2e596373d2823e3fb6a35394e79dd533ad2a243a2ef06dcdd78d62ecf3.jpg)  
Figure 20. We demonstrate gender manipulation (driven by the prompt “a photo of a male face”) for different manipulation strengths and disentanglement thresholds. Moving along the global direction, causes the facial features to become more masculine, while steps in the opposite direction yields more feminine features. The effect becomes stronger as the strength $\alpha$ increases. When the disentanglement threshold $\beta$ is high, only the facial features are affected, and as $\beta$ is lowered, additional correlated attributes, such as hair length and facial hair are affected as well.

![](images/7adb8e9fbc5e99e50d4be59b6a0b604bb735f33898ac0db09a48497a3300130f.jpg)  
Figure 21. Comparison with state-of-the-art methods using the same amount of manipulation according to a pretrained attribute classifie

![](images/01382534965ac1af5b1d7315355518340529bf6b19e5f4bc31dd4bf03781082b.jpg)  
Figure 22. Comparison between StyleFlow [1] and our global directions. Our method produces results of similar quality, despite the fact that StyleFlow simultaneously uses several attribute classifiers and regressors (from the Microsoft face API), and is thus able to manipulate a limited set of attributes. In contrast, our method requires no extra supervision.

![](images/5dfd54abccf8c1d7e1a367242d8e9a75ab08239f97513897d384c39813189a7a.jpg)  
Figure 23. We compare our global directions with our latent mapper using three different kinds of attributes.

![](images/d512f96dc135af101b4838dca16dd953e8b6e232f834e8aa9de7112bd19abc4f.jpg)  
Figure 24. Drastic manipulations in visually diverse datasets are sometimes difficult to achieve using our global directions. Here we use StyleGAN-ada [17] pretrained on AFHQ wild [5], which contains wolves, lions, tigers and foxes. There is a smaller domain gap between tigers and lions, which mainly involves color and texture transformations. However, there is a larger domain gap between tigers and wolves, which, in addition to color and texture transformations, also involves more drastic shape deformations. This figure demonstrates that our global directions method is more successful in transforming tigers into lions, while failing in some cases to transform tigers to wolves. The $\cdot _ { + } , \cdot _ { }$ and $\cdot _ { + + } ,$ indicate medium and strong manipulation strength, respectively.