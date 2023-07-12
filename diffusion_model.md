# Diffusion Model
Diffusion Model和其他生成模型最大的区别是它的latent code(z)和原图是同尺寸大小的，当然最近也有基于压缩的latent diffusion model。一句话概括diffusion model，即存在一系列高斯噪声($\mathbin{T}$轮)，将输入图片$\mathcal{x}_{0}$变为纯高斯噪声$\mathcal{x}_{T}$。而我们的模型则负责将$\mathcal{x}_{T}$复原回图片$\mathcal{x}_{0}$。这样一来其实diffusion model和GAN很像，都是给定噪声$\mathcal{x}_{T}$生成图片$\mathcal{x}_{0}$，但是要强调的是，这里的$\mathcal{x}_{T}$与图片$\mathcal{x}_{0}$是**同维度**的。

<div style="text-align:center">
<img src="./figs/generative_model.png">
</div>

---
## DDPM原理
---
### 问题概述
首先，为了简化问题，我们定义**batch size=1**，也就是每轮只有一张输入图片。将此输入图片用符号$\mathbf{x}_{0}$表示。扩散模型中，我们希望通过不断的加随机噪音，经过$T$次后，图像变成一个服从标准正态分布的随机量，即$\mathbf{x}_{T} \sim \mathcal{N}(0, \mathbf{I})$。这样我们在推理过程中，随机抽样一个服从$\mathcal{N}(0, \mathbf{I})$分布的张量，就可以通过逆向降噪过程获得一个全新的图片。

更一般化的，假设在添加$t$次噪音得到的图像为$\mathbf{x}_{t}$，如果通过$\mathbf{x}_{t}$的分布能够重构出$\mathbf{x}_{t-1}$的分布（注意不是$\mathbf{x}_{t}$本身），那么我们就能在之后的生成过程中抽样得到一个**具有随机性**的$\mathbf{x}_{t}$。如果能得到任意的$\mathbf{x}_{t-1}$，我们当然可以从$\mathbf{x}_{T}$重构出随机的$\mathbf{x}_{0}$. 
 
---
### Diffusion前向过程

所谓前向过程，即往图片上加噪声的过程。虽然这个步骤无法做到图片生成，但是这是理解diffusion model以及**构建训练样本GT**至关重要的一步。

给定真实图片$\mathbf{x}_{0} \sim q\left(\mathbf{x}\right)$，diffusion前向过程通过$\mathbin{T}$次累计对其添加高斯噪声，得到$\mathbf{x}_{1}, \mathbf{x}_{2}, \ldots, \mathbf{x}_{T}$，如下图的$q$过程。这里需要给定一系列的高斯分布方差的超参数$\{\beta_{t} \in \left(0, 1\right)\}_{t=1}^{T}$。前向过程由于每个时刻$t$只与$t-1$时刻有关，所以也可以看做马尔科夫过程：

$$
q\left(\mathbf{x}_{t} \mid \mathbf{x}_{t-1}\right) = \mathcal{N}\left(\mathbf{x}_{t}; \sqrt{1-\beta_{t}}\mathbf{x}_{t-1}, \beta_{t}\mathbf{I}\right) \qquad q\left(\mathbf{x}_{1:T} \mid \mathbf{x}_{0}\right) = \prod_{t=1}^{T}q\left(\mathbf{x}_{t} \mid \mathbf{x}_{t-1}\right)
\tag{1}
$$

这个过程中，随着$t$的增大，$\mathbf{x}_{t}$越来越接近纯噪声。当$T \rightarrow \infin$，$\mathbf{x}_{T}$是完全的高斯噪声(下面会证明，且与均值$\sqrt{1-\beta_{t}}$的选择有关)。且实际中$\beta_{t}$随着$t$增大是递增的，即$\beta_{1} \lt \beta_{2} \lt \ldots \lt \beta_{T}$。在GLIDE的code中，$\beta_{t}$是由0.0001到0.02线性插值(以$\mathcal{T}=1000$为基准，$\mathcal{T}$增加，$\beta_{T}$对应降低)。

<div style="text-ailgn:center">
<img src="./figs/ddpm.png">
</div>

---

前向过程结束介绍前，需要讲述一下diffusion在实现和推导过程中要用到的两个重要特征。

#### 特性1：重参数(reparameterization trick)
重参数技巧在很多工作(Gumbel Softmax, VAE)中有所引用。如果我们要从某个分布中随机采样(高斯分布)一个样本，这个过程是无法反传梯度的。而这个通过高斯噪声采样得到的$\mathbf{x}_{t}$的过程在diffusion中到处都是，因此我们需要通过重参数技巧来使得它可微。最通常的做法是把随机性通过一个独立的随机变量($\epsilon$)引导过去。举个例子，如果要从高斯分布$z \sim \mathcal{N}\left(z; \mu_{\theta}, \sigma_{\theta}^{2}\mathbf{I}\right)$采样一个$z$，我们可以写成：

$$
z = \mu_{\theta} + \sigma_{\theta} \odot \epsilon, \quad \epsilon \sim \mathcal{N}\left(0, \mathbf{I}\right)
$$

上式的$z$依旧是有随机性的，且满足均值为$\mu_{\theta}$方差为$\sigma_{\theta}^{2}$的高斯分布。这里的$\mu_{\theta}$，$\sigma_{\theta}^{2}$可以是由参数$\theta$的神经网络推断得到的。整个“采样”过程依旧梯度可导，随机性被转嫁到了$\epsilon$上。

#### 特性2：任意时刻的$\mathbf{x}_{t}$可以由$\mathbf{x}_{0}$和$\beta$表示
能够通过$\mathbf{x}_{0}$和$\beta$快速得到$\mathbf{x}_{t}$，对后续diffusion model的推断和推导有巨大作用。首先我们假设$\alpha_{t} = 1 - \beta_{t}$，并且$\bar{\alpha_{t}}=\prod_{i=1}^{T}\alpha_{i}$，由式$(1)$展开$\mathbf{x}_{t}$可以得到：

$$
\begin{aligned}
\mathbf{x}_{t} & = \sqrt{1-\beta_{t}}\mathbf{x}_{t-1} + \sqrt{\beta_{t}}\epsilon_{t-1} \quad \text{ where } \epsilon_{t-1}, \epsilon_{t-2}, \ldots \sim \mathcal{N}(0, \mathbf{I})\\
& =\sqrt{\alpha_{t}}\mathbf{x}_{t-1} + \sqrt{1-\alpha_{t}}\epsilon_{t-1} \\
& =\sqrt{\alpha_{t}}\left(\sqrt{\alpha_{t-1}}\mathbf{x}_{t-2} + \sqrt{1-\alpha_{t-1}}\epsilon_{t-2}\right) + \sqrt{1-\alpha_{t}}\epsilon_{t-1} \\
& =\sqrt{\alpha_{t}\alpha_{t-1}}\mathbf{x}_{t-2} + \left(\sqrt{\alpha_{t}\left(1-\alpha_{t-1}\right)}\epsilon_{t-2} + \sqrt{1-\alpha_{t}}\epsilon_{t-1}\right) \\
& =\sqrt{\alpha_{t} \alpha_{t-1}}\mathbf{x}_{t-2} + \sqrt{1-\alpha_{t}\alpha_{t-1}}\bar{\epsilon}_{t-2} \quad \text {where } \bar{\epsilon}_{t-2} \sim \mathcal{N}(0, \mathbf{I})  \text{ mergs two Gaussion$\left(*\right)$}\\
& =\ldots \\
& =\sqrt{\bar{\alpha}_{t}}\mathbf{x}_{0} + \sqrt{1-\bar{\alpha}_{t}}\epsilon. \\
q\left(\mathbf{x}_{t} \mid \mathbf{x}_{0}\right) & = \mathcal{N}\left(\mathbf{x}_{t}; \sqrt{\bar{\alpha_{t}}}\mathbf{x}_{0}, \left(1-\bar{\alpha_{t}}\right)\mathbf{I}\right).
\tag{2}
\end{aligned}
$$

由于独立高斯分布的可加性，即$\mathcal{N}\left(0, \sigma_{1}^{2}\right) + \mathcal{N}\left(0, \sigma_{2}^{2}\right) \sim \mathcal{N}\left(0, \left(\sigma_{1}^{2} + \sigma_{2}^{2} \right)\right)$，所以

$$
\begin{aligned}
    & \sqrt{\alpha_{t}\left(1-\alpha_{t-1}\right)}\epsilon_{t-2} \sim \mathcal{0, \alpha_{t}\left(1-\alpha_{t-1}\right)\mathbf{I}}, \\
    & \sqrt{1-\alpha_{t}}\epsilon_{t-1} \sim \mathcal{N}\left(0, \left(1-\alpha_{t}\right)\mathbf{I}\right), \\
    & \sqrt{\alpha_{t}\left(1-\alpha_{t-1}\right)}\epsilon_{t-2} + \sqrt{1-\alpha_{t}}\epsilon_{t-1} \sim \mathcal{N}\left(0, \left[\alpha_{t}\left(1-\alpha_{t-1}\right) + \left(1-\alpha_{t}\right)\right]\mathbf{I}\right) \\
    & = \mathcal{N}\left(0, \left(1-\alpha_{t}\alpha_{t-1}\right)\mathbf{I}\right). \tag{3}
\end{aligned}
$$

因此可以混合两个高斯分布得到标准差为$\sqrt{1-\alpha_{t}\alpha_{t-1}}$的混合高斯分布，式$(2)$中的$\bar{\epsilon_{t-2}}$仍然是标准高斯分布。而任意时刻的$\mathbf{x}_{t}$满足$q\left(\mathbf{x}_{t} \mid \mathbf{x}_{0}\right) = \mathcal{N}\left(\mathbf{x}_{t}; \sqrt{\bar{\alpha_{t}}}\mathbf{x}_{0}, \left(1-\bar{\alpha_{t}}\right)\mathbf{I}\right)$.

通过$Eq(2)$、$(3)$，可以发现当$\mathbin{T} \rightarrow \infin, \mathbf{x}_{T} \sim \mathcal{N}\left(0, \mathbf{I}\right)$，所以$\sqrt{1-\beta_{t}}$的均值系数能够稳定保证$\mathbf{x}_{T}$最后收敛到方差为1的保准高斯分布，且在$Eq(3)$的推导中也更为简洁优雅。

---

### Diffusion逆向(推断)过程

如果说前向过程(forward)是加噪的过程，那么逆向过程(reverse)就是diffusion的去噪推断过程。如果我们能够逐步得到逆转后的分布$q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}\right)$，就可以从标准的高斯分布$\mathbf{x}_{T} \sim \mathbf{N}\left(0, \mathbf{I}\right)$还原出原图$\mathbf{x}_{0}$。在文献[1]中证明了如果$q\left(\mathbf{x}_{t} \mid \mathbf{x}_{t-1}\right)$满足高斯分布且$\beta_{t}$足够小，$q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}\right)$仍然是一个高斯分布。然而我们无法简单推断$q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}\right)$，因此我们使用深度学习模型(参数为$\theta$，目前主流是U-Net+Attention的结构)去预测这样一个逆向分布$p_{\theta}$(类似VAE)：

$$
\begin{aligned}
    p_{\theta}\left(\mathbin{X}_{0:T}\right) & = p\left(\mathbf{x}_{T}\right)\prod_{t=1}^{T}p_{\theta}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}\right);  \\
    p_{\theta}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}\right) & = \mathcal{N}\left(\mathbf{x}_{t-1}; \boldsymbol{\mu}_{\theta}\left(\mathbf{x}_{t}, t\right), \boldsymbol{\Sigma}_{\theta}\left(\mathbf{x}_{t}, t\right)\right).
\end{aligned}
$$

虽然我们无法得到逆转后的分布$q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}\right)$，但是如果知道$\mathbf{x}_{0}$，我们是可以通过贝叶斯公式得到$q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}, \mathbf{x}_{0}\right)$为：

$$
q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}, \mathbf{x}_{0}\right) = \mathcal{N}\left(\mathbf{x}_{t-1}; \tilde{\mu}\left(\mathbf{x}_{t}, \mathbf{x}_{0}\right), \tilde{\beta_{t}}\mathbf{I}\right) \tag{8}
$$

过程如下：

$$
\begin{aligned}
    q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}, \mathbf{x}_{0}\right) & = \frac{q\left(\mathbf{x}_{t-1}, \mathbf{x}_{t}, \mathbf{x}_{0}\right)}{q\left(\mathbf{x}_{t}, \mathbf{x}_{0}\right)} \\
    & = \frac{q\left(\mathbf{x}_{0}\right) q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{0} \right) q\left(\mathbf{x}_{t} \mid \mathbf{x}_{t-1}, \mathbf{x}_{0}\right)}{q\left(\mathbf{x}_{t}, \mathbf{x}_{0}\right)} \\
    & = \frac{q\left(\mathbf{x}_{0}\right) q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{0} \right) q\left(\mathbf{x}_{t} \mid \mathbf{x}_{t-1}, \mathbf{x}_{0}\right)}{q\left(\mathbf{x}_{0}\right) q\left(\mathbf{x}_{t} \mid \mathbf{x}_{0}\right)} \\
    & = \frac{q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{0} \right) q\left(\mathbf{x}_{t} \mid \mathbf{x}_{t-1}, \mathbf{x}_{0}\right)}{q\left(\mathbf{x}_{t} \mid \mathbf{x}_{0}\right)} \\
    & = \frac{q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{0} \right) q\left(\mathbf{x}_{t} \mid \mathbf{x}_{t-1}\right)}{q\left(\mathbf{x}_{t} \mid \mathbf{x}_{0}\right)} \quad \text{由于$\mathbf{x}_{t-1} \rightarrow \mathbf{x}_{t}$与$\mathbf{x}_{0}$无关，因此可进行等价替换} \\
    & = q\left(\mathbf{x}_{t} \mid \mathbf{x}_{t-1}\right)\frac{q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{0} \right)}{q\left(\mathbf{x}_{t} \mid \mathbf{x}_{0}\right)} \\
    & \propto \exp \left(-\frac{1}{2}\left(\frac{\left(\mathbf{x}_{t}-\sqrt{\alpha_{t}} \mathbf{x}_{t-1}\right)^{2}}{\beta_{t}} + \frac{\left(\mathbf{x}_{t-1}-\sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_{0}\right)^{2}}{1-\bar{\alpha}_{t-1}} - \frac{\left(\mathbf{x}_{t}-\sqrt{\bar{\alpha}_{t}} \mathbf{x}_{0}\right)^{2}}{1-\bar{\alpha}_{t}}\right)\right) \\
    & =\exp \left(-\frac{1}{2}\left(\frac{\mathbf{x}_{t}^{2}-2 \sqrt{\alpha_{t}} \mathbf{x}_{t} \mathbf{x}_{t-1}+\alpha_{t} \mathbf{x}_{t-1}^{2}}{\beta_{t}}+\frac{\mathbf{x}_{t-1}^{2}-2 \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_{0} \mathbf{x}_{t-1}+\bar{\alpha}_{t-1} \mathbf{x}_{0}^{2}}{1-\bar{\alpha}_{t-1}}-\frac{\left(\mathbf{x}_{t}-\sqrt{\bar{\alpha}_{t}} \mathbf{x}_{0}\right)^{2}}{1-\bar{\alpha}_{t}}\right)\right) \\
    & = \exp \left(-\frac{1}{2}\left(\underbrace{\left(\frac{\alpha_{t}}{\beta_{t}}+\frac{1}{1-\bar{\alpha}_{t-1}}\right) \mathbf{x}_{t-1}^{2}}_{\mathbf{x}_{t-1} \text { 方差 }}-\underbrace{\left(\frac{2 \sqrt{\alpha_{t}}}{\beta_{t}} \mathbf{x}_{t}+\frac{2 \sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}} \mathbf{x}_{0}\right) \mathbf{x}_{t-1}}_{x_{t-1} \text { 均值 }}+\underbrace{C\left(\mathbf{x}_{t}, \mathbf{x}_{0}\right)}_{\text {与 } x_{t-1} \text { 方关 }}\right)\right) .
\end{aligned}
$$

由特性2：任意时刻的$\mathbf{x}_{t}$可以由$\mathbf{x}_{0}$和$\beta$表示，$\mathbf{x}_{t} = \sqrt{\bar{\alpha_{t}}}\mathbf{x}_{0} + \sqrt{1-\bar{\alpha_{t}}}\epsilon_{t}, \quad \epsilon \sim \mathcal{N}\left(0, \mathbf{I}\right)$可得：

$$
\left\{
    \begin{aligned}
    & \mathbf{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_{0} \sim \mathcal{N}\left(0, \left(1-\bar{\alpha}_{t-1}\right)\mathbf{I}\right) \\
    & \mathbf{x}_{t} - \sqrt{\bar{\alpha}_{t}}\mathbf{x}_{0} \sim \mathcal{N}\left(0, \left(1-\bar{\alpha_{t}}\right)\mathbf{I}\right)
    \end{aligned}
\right.
$$

上述过程巧妙地将**逆向**过程全部变回了**前向**过程，即$\left(\mathbf{x}_{t-1}, \mathbf{x}_{0}\right) \rightarrow \mathbf{x}_{t}; \quad \mathbf{x}_{0} \rightarrow \mathbf{x}_{t}; \quad \mathbf{x}_{0} \rightarrow \mathbf{x}_{t-1}$. 且$\mathcal{N}\left(\mu, \sigma^{2}\right)= \exp \left(-\frac{(x-\mu)^2}{2\sigma^2}\right) = \exp \left(-\frac{1}{2}\left(\frac{1}{\sigma^2}x^2 - \frac{2\mu}{\sigma^2}x + \frac{\mu^2}{\sigma^2}\right)\right)$，因此上式最终可整理成高斯分布概率密度函数形式。稍加整理我们可以得到$q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}, \mathbf{x}_{0}\right) = \mathcal{N}\left(\mathbf{x}_{t-1}; \tilde{\mu}\left(\mathbf{x}_{t}, \mathbf{x}_{0}\right), \tilde{\beta_{t}}\mathbf{I}\right)$中的方差和均值为：

$$
\begin{aligned}
    & \frac{1}{\sigma^{2}}=\frac{1}{\tilde{\beta}_{t}}=\left(\frac{\alpha_{t}}{\beta_{t}}+\frac{1}{1-\bar{\alpha}_{t-1}}\right) ; \quad \tilde{\beta}_{t}=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}} \cdot \beta_{t} \\
    & \frac{2 \mu}{\sigma^{2}}=\frac{2 \tilde{\mu}_{t}\left(\mathbf{x}_{t}, \mathbf{x}_{0}\right)}{\tilde{\beta}_{t}}=\left(\frac{2 \sqrt{\alpha_{t}}}{\beta_{t}} \mathbf{x}_{t}+\frac{2 \sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}} \mathbf{x}_{0}\right) ; \\
    & \tilde{\mu}_{t}\left(x_{t}, x_{0}\right)=\frac{\sqrt{{\alpha_{t}}}\left(1-\bar{\alpha}_{t-1}\right)}{1-\bar{\alpha}_{t}} \mathbf{x}_{t}+\frac{\sqrt{\bar{\alpha}_{t-1}} \beta_{t}}{1-\bar{\alpha}_{t}} \mathbf{x}_{0}. \tag{4}
\end{aligned}   
$$

根据特性2，可知$\mathbf{x}_{0} = \frac{1}{\sqrt{\bar{\alpha}_{t}}}\left(\mathbf{x}_{t} - \sqrt{1-\bar{\alpha}_{t}}\right)\epsilon_{t}$，因此代入式$(4)$可得：

$$
\begin{aligned}
    \tilde{\mu}_{t} & = \frac{\sqrt{\alpha_{t}}\left(1-\bar{\alpha}_{t-1}\right)}{1-\bar{\alpha}_{t}}\mathbf{x}_{t} + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_{t}}{1-\bar{\alpha}_{t}}\frac{1}{\sqrt{\bar{\alpha}_{t}}}\left(\mathbf{x}_{t} - \sqrt{1-\bar{\alpha}_{t}}\epsilon_{t}\right)  \\ 
    & = \frac{1}{\sqrt{\alpha_{t}}}\left(\mathbf{x}_{t} - \frac{\beta_{t}}{\sqrt{1-\bar{\alpha}_{t}}}\epsilon_{t}\right)
\end{aligned}
$$

其中高斯分布$\epsilon_{t}$为深度模型所预测的噪声(用于去噪)，可以看做$\epsilon_{\theta}\left(\mathbf{x}_{t}, t\right)$，即得到：

$$
\mu_{\theta}\left(\mathbf{x}_{t}, t\right) = \frac{1}{\sqrt{\alpha_{t}}}\left(\mathbf{x}_{t} - \frac{\beta_{t}}{\sqrt{1-\bar{\alpha}_{t}}}\epsilon_{\theta}\left(\mathbf{x}_{t}, t\right)\right) \tag{5}
$$

---
这样一来，DDPM的每一步的推断可以总结为：

(1) 每个时间步通过$\mathbf{x}_{t}$和$t$来预测高斯噪声$\epsilon_{\theta}\left(\mathbf{x}_{t}, t\right)$，随后由式$(5)$得到均值$\mu_{\theta}\left(\mathbf{x}_{t}, t\right)$.  
(2) 得到方差$\boldsymbol{\Sigma}_{\theta}\left(\mathbf{x}_{t}, t\right)$，DDPM中使用untrained $\boldsymbol{\Sigma}_{\theta}\left(\mathbf{x}_{t}, t\right) = \tilde{\beta}_{t}$，且认为$\tilde{\beta}_{t}=\beta_{t}$和$\tilde{\beta}_{t}=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}} \cdot \beta_{t}$结果近似，在GLIDE中则是根据网络预测trainable方差$\boldsymbol{\Sigma}_{\theta}\left(\mathbf{x}_{t}, t\right)$.  
(3) 根据$p_{\theta}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}\right) = \mathcal{N}\left(\mathbf{x}_{t-1}; \boldsymbol{\mu}_{\theta}\left(\mathbf{x}_{t}, t\right), \boldsymbol{\Sigma}_{\theta}\left(\mathbf{x}_{t}, t\right)\right)$得到$q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}\right)$的估计，利用重参数技巧得到$\mathbf{x}_{t-1}$.

---
### Diffusion训练
搞清楚diffusion的逆向过程之后，我们算是搞清楚diffusion的推断过程了。但是如何训练diffusion model以得到靠谱的$\boldsymbol{\mu}_{\theta}\left(\mathbf{x}_{t}, t\right)$和$\boldsymbol{\Sigma}_{\theta}\left(\mathbf{x}_{t}, t\right)$呢？通过对真实数据分布下，最大化模型预测分布的对数似然，即优化在$\mathbf{x}_{0} \sim q\left(\mathbf{x}_{0}\right)$下的$p_{\theta}\left(\mathbf{x}_{0}\right)$交叉熵:

$$
\mathcal{L} = \mathbb{E}_{q\left(\mathbf{x}_{0}\right)}\left[-\log p_{\theta}\left(\mathbf{x}_{0}\right)\right]. \tag{6}
$$

注意，上述最大化模型预测分布的对数似然意味着：
$$
\begin{aligned}
    & \boldsymbol{max} \quad \mathbb{E}_{q\left(\mathbf{x}_{0}\right)}\left[\log p_{\theta}\left(\mathbf{x}_{0}\right)\right] \\
    \Leftrightarrow \quad & \boldsymbol{min} \quad -\mathbb{E}_{q\left(\mathbf{x}_{0}\right)}\left[\log p_{\theta}\left(\mathbf{x}_{0}\right)\right] \\
    \Leftrightarrow \quad & \boldsymbol{min} \quad \mathbb{E}_{q\left(\mathbf{x}_{0}\right)}\left[-\log p_{\theta}\left(\mathbf{x}_{0}\right)\right]

\end{aligned} 
$$
 
即最小化在$\mathbf{x}_{0} \sim q\left(\mathbf{x}_{0}\right)$下的$p_{\theta}\left(\mathbf{x}_{0}\right)$交叉熵。

从图2可以看出这个过程很像VAE，即可以使用变分下限(Variational Lower Bound, VLB)来优化负对数似然。由于KL散度非负，可得到：

$$
\begin{aligned}
    -\log p_{\theta}\left(\mathbf{x}_{0}\right) & \le -\log p_{\theta}\left(\mathbf{x}_{0}\right) +  D_{\mathrm{KL}}\left(q\left(\mathbf{x}_{1:T} \mid \mathbf{x}_{0}\right) \| p_{\theta}\left(\mathbf{x}_{1:T} \mid \mathbf{x}_{0}\right)\right) \\
    & = - \log p_{\theta}\left(\mathbf{x}_{0}\right) + \mathbb{E}_{q \left(\mathbf{x}_{1:T} \mid \mathbf{x}_{0}\right)} \left[ \log \frac{q \left(\mathbf{x}_{1:T} \mid \mathbf{x}_{0}\right)}{p_{\theta}\left(\mathbf{x}_{0:T}\right) / p_{\theta}\left(\mathbf{x}_{0}\right)}\right]; \quad \text{where } p_{\theta}\left(\mathbf{x}_{1:T} \mid \mathbf{x}_{0}\right) = \frac{p_{\theta}\left(\mathbf{x}_{0:T}\right)}{p_{\theta}\left(\mathbf{x}_{0}\right)} \\
    & = - \log p_{\theta}\left(\mathbf{x}_{0}\right) + \mathbb{E}_{q \left(\mathbf{x}_{1:T} \mid \mathbf{x}_{0}\right)} \left[ \log \frac{q \left(\mathbf{x}_{1:T} \mid \mathbf{x}_{0}\right)}{p_{\theta}\left(\mathbf{x}_{0:T}\right) } + \underbrace{\log p_{\theta}\left(\mathbf{x}_{0}\right)}_{\text{与$q$无关}}\right] \\
    & = \mathbb{E}_{q \left(\mathbf{x}_{1:T} \mid \mathbf{x}_{0}\right)} \left[ \log \frac{q \left(\mathbf{x}_{1:T} \mid \mathbf{x}_{0}\right)}{p_{\theta}\left(\mathbf{x}_{0:T}\right) } \right]. \tag{7}
\end{aligned}
$$

对式$(7)$左右取期望$\mathbb{E}_{q\left(\mathbf{x}_{0}\right)}$，利用重积分中的Fubini定理：

$$
\mathcal{L}_{VLB} = \underbrace{\mathbb{E}_{q\left(\mathbf{x}_{0}\right)}\left(\mathbb{E}_{q\left(\mathbf{x}_{1:T} \mid \mathbf{x}_{0}\right)}\log \frac{q \left(\mathbf{x}_{1:T} \mid \mathbf{x}_{0}\right)}{p_{\theta}\left(\mathbf{x}_{0:T}\right) }\right) = \mathbb{E}_{q \left(\mathbf{x}_{0:T} \right)} \left[ \log \frac{q \left(\mathbf{x}_{1:T} \mid \mathbf{x}_{0}\right)}{p_{\theta}\left(\mathbf{x}_{0:T}\right) } \right]}_{Fubini\text{定理}} \ge  \mathbb{E}_{q\left(\mathbf{x}_{0}\right)}\left[-\log p_{\theta}\left(\mathbf{x}_{0}\right)\right]
$$

因此我们可以通过最小化$\mathcal{L}_{VLB}$来优化目标函数，能够最小化$\mathcal{L}_{VLB}$即可最小化我们的目标函数$(6)$。

---

#### Fubini定理
Fubini定理给出了使用逐次积分的方法计算双重积分的条件。在这些条件下，不仅能够使用逐次积分计算双重积分，而且交换逐次积分的顺序时，积分结果不变。

若  
$$
\int_{A \times B} \left |  f\left(x, y \right) \right | d\left(x, y\right) < \infin, 
$$
其中$A$和$B$都是$\sigma-$有限测度空间，$A \times B$是$A$和$B$的**积可测空间**，$f: A \times B \mapsto \mathbb{C}$是可测函数，那么  
$$
\int_{A}\left(\int_{B} f\left(x, y\right)dy\right)dx = \int_{B}\left(\int_{A} f\left(x, y\right)dx\right)dy = \int_{A \times B} \left |  f\left(x, y \right) \right | d\left(x, y\right),
$$
前二者是在两个测度空间上的逐次积分，但是积分次序不同；第三个是在乘积空间上关于乘积测度的积分。  
特别地，如果$f\left(x, y\right) = h(x)g(y)$，则
$$
\int_{A}h(x)dx\int_{B}g(y)dy = \int_{A \times B}f(x, y)d(x, y).
$$
如果条件中的绝对积分值不是有限，那么上述两个逐次积分的值可能不同。  

Fubini定理一个应用是计算高斯积分：
$$
\int_{-\infin}^{\infin} e^{-\alpha x^2}dx = \sqrt{\frac{\pi}{\alpha}}
$$

#### Jensen不等式
如果$X$是随机变量，$g$是凸函数，则
$$
g(\mathbb{E}\left[ X \right]) \le \mathbb{E}\left[g(X)\right]
$$
等式当且仅当$X$是一个常数或者$g$是线性时成立，这个性质称为Jensen不等式。

---

另一方面，通过Jensen不等式也可以得到一样的目标：

$$
\begin{aligned}
    \mathcal{L} & = \mathbb{E}_{q(\mathbf{x}_{0})}\left[- \log p_{\theta(\mathbf{x}_{0})}\right] \\
    & = - \mathbb{E}_{q(\mathbf{x}_{0})} \log \left(p_{\theta}(\mathbf{x}_{0}) \cdot \underbrace{\int p_{\theta}(\mathbf{x}_{1:T})d\mathbf{x}_{1:T}}_{\text{积分为1}}\right) \\
    & = - \mathbb{E}_{q(\mathbf{x}_{0})} \log \left(\int p_{\theta}(\mathbf{x}_{0:T})d\mathbf{x}_{1:T}\right) \\
    & = - \mathbb{E}_{q(\mathbf{x}_{0})} \log \left(\int q(\mathbf{x}_{1:T} \mid \mathbf{x}_{0}) \frac{p_{\theta}(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \mid \mathbf{\mathbf{x}_{0}})})d\mathbf{x}_{1:T}\right) \\
    & = - \mathbb{E}_{q(\mathbf{x}_{0})} \log \left(\mathbb{E}_{q(\mathbf{x}_{1:T} \mid \mathbf{x}_{0})} \frac{p_{\theta}(\mathbf{x}_{0:T})}{q\left(\mathbf{x}_{1:T} \mid \mathbf{\mathbf{x}_{0}}\right)}\right) \\
    & \le - \mathbb{E}_{q(\mathbf{x}_{0})}\mathbb{E}_{q(\mathbf{x}_{1:T} \mid \mathbf{x}_{0})}\left[\log \frac{p_{\theta}(\mathbf{x}_{0:T})}{q\left(\mathbf{x}_{1:T} \mid \mathbf{\mathbf{x}_{0}}\right)} \right] \qquad \text{Jensen不等式} \\
    & = - \mathbb{E}_{q(\mathbf{x}_{0:T})} \log \frac{p_{\theta}(\mathbf{x}_{0:T})}{q\left(\mathbf{x}_{1:T} \mid \mathbf{\mathbf{x}_{0}}\right)} \\
    & = \mathbb{E}_{q(\mathbf{x}_{0:T})} \log \frac{q\left(\mathbf{x}_{1:T} \mid \mathbf{\mathbf{x}_{0}}\right)}{p_{\theta}(\mathbf{x}_{0:T})} = \mathcal{L}_{VLB}.
\end{aligned}
$$

---
进一步对$\mathcal{L}_{VLB}$推导，可以得到熵与多个$KL$散度的累加：
$$
\begin{aligned}
    L_{\mathrm{VLB}} & =\mathbb{E}_{q\left(\mathbf{x}_{0:T}\right)}\left[\log \frac{q\left(\mathbf{x}_{1: T} \mid \mathbf{x}_0\right)}{p_\theta\left(\mathbf{x}_{0: T}\right)}\right] \\
    & =\mathbb{E}_q\left[\log \frac{\prod_{t=1}^T q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right)}{p_\theta\left(\mathbf{x}_T\right) \prod_{t=1}^T p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)}\right] \\
    & =\mathbb{E}_q\left[-\log p_\theta\left(\mathbf{x}_T\right)+\sum_{t=1}^T \log \frac{q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right)}{p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)}\right] \\
    & =\mathbb{E}_q\left[-\log p_\theta\left(\mathbf{x}_T\right)+\sum_{t=2}^T \log \frac{q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right)}{p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)}+\log \frac{q\left(\mathbf{x}_1 \mid \mathbf{x}_0\right)}{p_\theta\left(\mathbf{x}_0 \mid \mathbf{x}_1\right)}\right] \\
    & =\mathbb{E}_q\left[-\log p_\theta\left(\mathbf{x}_T\right)+\sum_{t=2}^T \log \left(\frac{q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right)}{p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)} \cdot \frac{q\left(\mathbf{x}_t \mid \mathbf{x}_0\right)}{q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_0\right)}\right)+\log \frac{q\left(\mathbf{x}_1 \mid \mathbf{x}_0\right)}{p_\theta\left(\mathbf{x}_0 \mid \mathbf{x}_1\right)}\right] \\
    & =\mathbb{E}_q\left[-\log p_\theta\left(\mathbf{x}_T\right)+\sum_{t=2}^T \log \frac{q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right)}{p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)}+\sum_{t=2}^T \log \frac{q\left(\mathbf{x}_t \mid \mathbf{x}_0\right)}{q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_0\right)}+\log \frac{q\left(\mathbf{x}_1 \mid \mathbf{x}_0\right)}{p_\theta\left(\mathbf{x}_0 \mid \mathbf{x}_1\right)}\right] \\
    & =\mathbb{E}_q\left[-\log p_\theta\left(\mathbf{x}_T\right)+\sum_{t=2}^T \log \frac{q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right)}{p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)}+\log \frac{q\left(\mathbf{x}_T \mid \mathbf{x}_0\right)}{q\left(\mathbf{x}_1 \mid \mathbf{x}_0\right)}+\log \frac{q\left(\mathbf{x}_1 \mid \mathbf{x}_0\right)}{p_\theta\left(\mathbf{x}_0 \mid \mathbf{x}_1\right)}\right] \\
    & =\mathbb{E}_q\left[\log \frac{q\left(\mathbf{x}_T \mid \mathbf{x}_0\right)}{p_\theta\left(\mathbf{x}_T\right)}+\sum_{t=2}^T \log \frac{q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right)}{p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)}-\log p_\theta\left(\mathbf{x}_0 \mid \mathbf{x}_1\right)\right] \\
    & =\mathbb{E}_q[\underbrace{D_{KL}\left(q\left(\mathbf{x}_T \mid \mathbf{x}_0\right) \| p_\theta\left(\mathbf{x}_T\right)\right)}_{L_T} + \sum_{t=2}^T \underbrace{D_{KL}\left(q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right) \| p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)\right)}_{L_{t-1}} \underbrace{- \log p_{\theta}\left(\mathbf{x}_{0} \mid \mathbf{x}_{1}\right)}_{L_{0}}]
\end{aligned}
$$

注：上面推导的最后一步将$q(\mathbf{x}_{0:T})$表示为$q(\mathbf{x}_{1:T} \mid \mathbf{x}_{0})q(\mathbf{x}_{0})$，再将对应的$q$放入括号里即可得到。

也可以写成：

$$
\begin{aligned}
    & \mathcal{L}_{VLB} = L_{T} + L_{T-1} + \ldots + L_{0} \\
    & L_{T} = D_{KL}\left(q\left(\mathbf{x}_T \mid \mathbf{x}_0\right) \| p_\theta\left(\mathbf{x}_T\right)\right) \\
    & L_{t} = D_{KL}\left(q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right) \| p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)\right); \qquad 1 \le t \le \mathbin{T} - 1 \\
    & L_{0} = - \log p_{\theta}\left(\mathbf{x}_{0} \mid \mathbf{x}_{1}\right).
\end{aligned}
$$

由于前向$q$没有科学系参数，而$\mathbf{x}_{T}$则是纯高斯噪声，$L_{T}$可以当做常量忽略。而$L_{t}$则可以看做拉进两个高斯分布$q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\mu}(\mathbf{x}_{t}, \mathbf{x}_{0}), \tilde{\beta}_{t}\mathbf{I})$和$p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_{\theta}(\mathbf{x}_{t}, t), \boldsymbol{\Sigma}_{\theta})$，根据高斯分布的KL散度公式求解下式：

$$
L_{t}=\mathbb{E}_{q}\left[\frac{1}{2\left\|\Sigma_{\theta}\left(\mathbf{x}_{t}, t\right)\right\|_{2}^{2}}\left\|\tilde{\mu}_{t}\left(\mathbf{x}_{t}, \mathbf{x}_{0}\right)-\mu_{\theta}\left(\mathbf{x}_{t}, t\right)\right\|^{2}\right]+C,
$$

其中$C$是与模型参数$\theta$无关的常量。把$\tilde{\mu}(\mathbf{x}_{t}, \mathbf{x}_{0})$和$\boldsymbol{\mu}_{\theta}(\mathbf{x}_{t}, t)$以及$\mathbf{x}_{t}$代入上式可得：

$$
\begin{aligned}
    \tilde{\mu}_{t} & = \frac{1}{\sqrt{\alpha_{t}}}\left(\mathbf{x}_{t} - \frac{\beta_{t}}{\sqrt{1-\bar{\alpha}_{t}}}\epsilon_{t}\right) \quad \mu_{\theta}\left(\mathbf{x}_{t}, t\right) = \frac{1}{\sqrt{\alpha_{t}}}\left(\mathbf{x}_{t} - \frac{\beta_{t}}{\sqrt{1-\bar{\alpha}_{t}}}\epsilon_{\theta}\left(\mathbf{x}_{t}, t\right)\right)\\
    L_{t} & =\mathbb{E}_{\mathbf{x}_{0}, \epsilon}\left[\frac{1}{2\left\|\Sigma_{\theta}\left(\mathbf{x}_{t}, t\right)\right\|_{2}^{2}}\left\|\tilde{\mu}_{t}\left(\mathbf{x}_{t}, \mathbf{x}_{0}\right)-\mu_{\theta}\left(\mathbf{x}_{t}, t\right)\right\|^{2}\right] \\
    & =\mathbb{E}_{\mathbf{x}_{0}, \epsilon}\left[\frac{1}{2\left\|\Sigma_{\theta}\left(\mathbf{x}_{t}, t\right)\right\|_{2}^{2}} \left \| \frac{1}{\sqrt{\alpha_{t}}}\left(\mathbf{x}_{t}-\frac{\beta_{t}}{\sqrt{1-\bar{a}_{t}}} \epsilon_{t}\right)-\frac{1}{\sqrt{\alpha_{t}}}\left(\mathbf{x}_{t}-\frac{\beta_{t}}{\sqrt{1-\bar{a}_{t}}} \epsilon_{\theta}\left(\mathbf{x}_{t}, t\right) \right)\right \|^2\right] \\
    & =\mathbb{E}_{x_{0}, \epsilon}\left[\frac{\beta_{t}^{2}}{2 \alpha_{t}\left(1-\bar{\alpha}_{t}\left\|\Sigma_{\theta}\right\|_{2}^{2}\right)}\left\|\epsilon_{t}-\epsilon_{\theta}\left(\mathbf{x}_{t}, t\right)\right\|^{2}\right] \\
    & =\mathbb{E}_{\mathbf{x}_{0}, \epsilon}\left[\frac{\beta_{t}^{2}}{2 \alpha_{t}\left(1-\bar{\alpha}_{t}\left\|\Sigma_{\theta}\right\|_{2}^{2}\right)}\left\|\epsilon_{t}-\epsilon_{\theta}\left(\sqrt{\bar{\alpha}_{t}} \mathbf{x}_{0}+\sqrt{1-\bar{\alpha}_{t}} \epsilon_{t}, t\right)\right\|^{2}\right] .
\end{aligned}
$$

由上式可知，diffusion训练的核心就是学习高斯噪声$\epsilon_{t}, \epsilon_{\theta}$之间的MSE。

$L_{0}=- \log p_{\theta}(\mathbf{x}_{0} \mid \mathbf{x}_{1})$相当于最后一步的熵，DDPM文中指出，从$\mathbf{x}_{1}$到$\mathbf{x}_{0}$应该是一个离散化过程，因为图像RGB值都是离散化的。DDPM针对$p_{\theta}(\mathbf{x}_{0} \mid \mathbf{x}_{1})$构建了一个离散化的分段积分累乘，类似于基于分数目标的自回归(Autoregressive)学习。

DDPM将loss进一步简化为：

$$
L_{t}^{simple}=\mathbb{E}_{\mathbf{x}_{0}, \epsilon}\left[\left\|\epsilon_{t}-\epsilon_{\theta}\left(\sqrt{\bar{\alpha}_{t}} \mathbf{x}_{0}+\sqrt{1-\bar{\alpha}_{t}} \epsilon_{t}, t\right)\right\|^{2}\right].
$$

正如之前提过的，DDPM并没有将模型预测的方差$\Sigma_{\theta}(\mathbf{x}_{t}, t)$考虑到训练和推断中，而是通过untrained $\beta_{t}$或者$(4)$中的$\tilde{\beta}_{t}$代替。因为$\Sigma_{\theta}$可能导致训练的不稳定。

因此，训练过程可以看做:  
(1) **获取输入$\mathbf{x}_{0}$，从$1, \ldots, T$随机采样一个$t$.**  
(2) **从标准高斯分布采样一个噪声$\epsilon \sim \mathcal{N}(0, \mathbf{I})$.**  
(3) **最小化$\left\|\epsilon_{t}-\epsilon_{\theta}\left(\sqrt{\bar{\alpha}_{t}} \mathbf{x}_{0}+\sqrt{1-\bar{\alpha}_{t}} \epsilon_{t}, t\right)\right\|$.**

算法流程图如下：

<div style="text-ailgn:center">
<img src="./figs/ddpm_train_test.png">
</div>

---
#### 一元连续高斯分布的KL散度
$$
\begin{aligned}
    \text { Let } p(x) & =\mathcal{N}\left(\mu_{1}, \sigma_{1}\right), \quad q(x)=\mathcal{N}\left(\mu_{2}, \sigma_{2}\right) \\
    \mathbf{KL}(p, q) & = -\int p(x) \log q(x) d x+\int p(x) \log p(x) d x \\
    & = -\int p(x) \log \frac{1}{\left(2 \pi \sigma_{2}^{2}\right)^{(1 / 2)}} e^{-\frac{\left(x-\mu_{2}\right)^{2}}{2 \sigma_{2}^{2}}} d x-\frac{1}{2}\left(1+\log 2 \pi \sigma_{1}^{2}\right) \\
    & = \frac{1}{2} \log \left(2 \pi \sigma_{2}^{2}\right)-\int p(x) \log e^{-\frac{\left(x-\mu_{2}\right)^{2}}{2 \sigma_{2}^{2}}} d x-\frac{1}{2}\left(1+\log 2 \pi \sigma_{1}^{2}\right) \\
    & = \frac{1}{2} \log \left(2 \pi \sigma_{2}^{2}\right)+\frac{\int p(x) x^{2} d x-\int p(x) 2 x \mu_{2} d x+\int p(x) \mu_{2}^{2} d x}{2 \sigma_{2}^{2}}-\frac{1}{2}\left(1+\log 2 \pi \sigma_{1}^{2}\right) \\
    & = \frac{1}{2} \log \left(2 \pi \sigma_{2}^{2}\right)+\frac{\left\langle x^{2}\right\rangle-2\langle x\rangle \mu_{2}+\mu_{2}^{2}}{2 \sigma_{2}^{2}}-\frac{1}{2}\left(1+\log 2 \pi \sigma_{1}^{2}\right) \\
    & = \frac{1}{2} \log \left(2 \pi \sigma_{2}^{2}\right)+\frac{\sigma_{1}^{2}+\left(\mu_{1}-\mu_{2}\right)^{2}}{2 \sigma_{2}^{2}}-\frac{1}{2}\left(1+\log 2 \pi \sigma_{1}^{2}\right) \\
    & = \log \frac{\sigma_{2}}{\sigma_{1}}+\frac{\sigma_{1}^{2}+\left(\mu_{1}-\mu_{2}\right)^{2}}{2 \sigma_{2}^{2}}-\frac{1}{2}
\end{aligned}
$$

其中$\langle \rangle$代表$p$分布下的期望，且$\mathbb{var}(x)=\langle x^2 \rangle - \langle x \rangle^2, \quad \langle x^2 \rangle = \sigma_{1}^{2} + \mu_{1}^{2}$.

#### 多元连续高斯分布的KL散度
$$
\text{ Let } p(x) = \mathcal{N}(\boldsymbol{\mu}_{1}, \boldsymbol{\Sigma}_{1}), \quad q(x) = \mathcal{N}(\boldsymbol{\mu}_{2}, \boldsymbol{\Sigma}_{2}).
$$

多元高斯分布：
$$
\mathcal{N}(x \mid \mu, \Sigma) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp \left\{-\frac{1}{2}(x-\mu)^{T}\Sigma^{-1}(x-\mu)\right\}
$$

矩阵的迹的性质：
- $tr(\alpha A + \beta B) = \alpha tr(A) + \beta tr(B)$
- $tr(A) = tr(A^{T})$
- $tr(AB) = tr(BA)$
- tr(ABC) = tr(CAB) = tr(BCA)
- 对于列向量$\lambda$，$\lambda^{T}A\lambda$为一个标量，因此：$\lambda = tr(\lambda^{T}A\lambda) = tr(A \lambda \lambda^{T})$

多变量分布中期望$\boldsymbol{E}$与协方差$\boldsymbol{\Sigma}$的性质：
- $\boldsymbol{E}\left[xx^{T}\right] = \boldsymbol{\Sigma} + \mu \mu^{T}$
$$
\begin{aligned}
    \boldsymbol{\Sigma} & = E\left[(x-\mu)(x-\mu)^{T}\right] \\
    & = E\left[xx^{T} - x\mu^{T} - \mu x^{T} + \mu \mu^{T}\right] \\
    & = E\left[xx^{T}\right] - \mu \mu^{T} - \mu \mu^{T} + \mu \mu^{T} \\
    & = E\left[xx^{T}\right] - \mu \mu^{T}
\end{aligned}
$$
- $E(x^{T}Ax)=tr(A\Sigma)+\mu^{T}A\mu$

$$
\begin{aligned}
    E(x^{T}Ax) & = E\left[tr(x^{T}Ax)\right] \\
    & = E\left[tr(Axx^{T})\right] \\
    & = tr\left[E(Axx^{T})\right] \\
    & = tr\left[AE(xx^{T})\right] \\
    & = tr\left[A(\Sigma + \mu \mu^{T})\right] \\
    & = tr(A\Sigma) + tr(A\mu \mu^{T}) \\
    & = tr(A\Sigma) + tr(\mu^{T} A \mu) \\
    & = tr(A\Sigma) + \mu^{T}A\mu
\end{aligned}
$$

证：
$$
\begin{aligned}
    D_{K L}(p \| q) & = E_{p}[\log p-\log q] \\
    & =\frac{1}{2} E_{p}\left[-\log \left|\Sigma_{1}\right|-\left(x-u_{1}\right)^{T} \Sigma_{1}^{-1}\left(x-u_{1}\right)+\log \left|\Sigma_{2}\right|+\left(x-u_{2}\right)^{T} \Sigma_{2}^{-1}\left(x-u_{2}\right)\right] \\
    & =\frac{1}{2} \log \frac{\left|\Sigma_{2}\right|}{\left|\Sigma_{1}\right|}+\frac{1}{2} E_{p}\left[-\left(x-u_{1}\right)^{T} \Sigma_{1}^{-1}\left(x-u_{1}\right)+\left(x-u_{2}\right)^{T} \Sigma_{2}^{-1}\left(x-u_{2}\right)\right] \\
    & =\frac{1}{2} \log \frac{\left|\Sigma_{2}\right|}{\left|\Sigma_{1}\right|}+\frac{1}{2} E_{p}\left\{-\operatorname{tr}\left[\Sigma_{1}^{-1}\left(x-u_{1}\right)\left(x-u_{1}\right)^{T}\right]+\operatorname{tr}\left[\Sigma_{2}^{-1}\left(x-u_{2}\right)\left(x-u_{2}\right)^{T}\right]\right\} \\
    & =\frac{1}{2} \log \frac{\left|\Sigma_{2}\right|}{\left|\Sigma_{1}\right|}+\frac{1}{2} E_{p}\left\{-\operatorname{tr}\left[\Sigma_{1}^{-1}\left(x-u_{1}\right)\left(x-u_{1}\right)^{T}\right]\right\}+\frac{1}{2} E_{p}\left\{\operatorname{tr}\left[\Sigma_{2}^{-1}\left(x-u_{2}\right)\left(x-u_{2}\right)^{T}\right]\right\} \\
    & =\frac{1}{2} \log \frac{\left|\Sigma_{2}\right|}{\left|\Sigma_{1}\right|}-\frac{1}{2} \operatorname{tr}\left\{E_{p}\left[\Sigma_{1}^{-1}\left(x-u_{1}\right)\left(x-u_{1}\right)^{T}\right]\right\}+\frac{1}{2} \operatorname{tr}\left\{E_{p}\left[\Sigma_{2}^{-1}\left(x-u_{2}\right)\left(x-u_{2}\right)^{T}\right]\right\} \\
    & =\frac{1}{2} \log \frac{\left|\Sigma_{2}\right|}{\left|\Sigma_{1}\right|}-\frac{1}{2} \operatorname{tr}\left\{\Sigma_{1}^{-1} E_{p}\left[\left(x-u_{1}\right)\left(x-u_{1}\right)^{T}\right]\right\}+\frac{1}{2} \operatorname{tr}\left\{E_{p}\left[\Sigma_{2}^{-1}\left(x x^{T}-u_{2} x^{T}-x u_{2}^{T}+u_{2} u_{2}^{T}\right)\right]\right\} \\
    & =\frac{1}{2} \log \frac{\left|\Sigma_{2}\right|}{\left|\Sigma_{1}\right|}-\frac{1}{2} \operatorname{tr}\left\{\Sigma_{1}^{-1} \Sigma_{1}\right\}+\frac{1}{2} \operatorname{tr}\left\{\Sigma_{2}^{-1} E_{p}\left(x x^{T}-u_{2} x^{T}-x u_{2}^{T}+u_{2} u_{2}^{T}\right)\right\} \\
    & =\frac{1}{2} \log \frac{\left|\Sigma_{2}\right|}{\left|\Sigma_{1}\right|}-\frac{1}{2} n+\frac{1}{2} \operatorname{tr}\left\{\Sigma_{2}^{-1}\left(\Sigma_{1}+u_{1} u_{1}^{T}-u_{2} u_{1}^{T}-u_{1} u_{2}^{T}+u_{2} u_{2}^{T}\right)\right\}-- \text { 这里利用了 } E\left[x x^{T}\right]=\Sigma+u u^{T} \\
    & =\frac{1}{2}\left\{\log \frac{\left|\Sigma_{2}\right|}{\left|\Sigma_{1}\right|}-n+\operatorname{tr}\left(\Sigma_{2}^{-1} \Sigma_{1}\right)+\operatorname{tr}\left\{\Sigma_{2}^{-1}\left(u_{1} u_{1}^{T}-u_{2} u_{1}^{T}-u_{1} u_{2}^{T}+u_{2} u_{2}^{T}\right)\right\}\right\} \\
    & =\frac{1}{2}\left\{\log \frac{\left|\Sigma_{2}\right|}{\left|\Sigma_{1}\right|}-n+\operatorname{tr}\left(\Sigma_{2}^{-1} \Sigma_{1}\right)+\operatorname{tr}\left\{\Sigma_{2}^{-1} u_{1} u_{1}^{T}-\Sigma_{2}^{-1} u_{2} u_{1}^{T}-\Sigma_{2}^{-1} u_{1} u_{2}^{T}+\Sigma_{2}^{-1} u_{2} u_{2}^{T}\right\}\right\} \\
    & =\frac{1}{2}\left\{\log \frac{\left|\Sigma_{2}\right|}{\left|\Sigma_{1}\right|}-n+\operatorname{tr}\left(\Sigma_{2}^{-1} \Sigma_{1}\right)+u_{1}^{T} \Sigma_{2}^{-1} u_{1}-2 u_{1}^{T} \Sigma_{2}^{-1} u_{2}+u_{2}^{T} \Sigma_{2}^{-1} u_{2}\right\}-- \text { 这里利用了 } \lambda=\operatorname{tr}\left(\lambda^{T} A \lambda\right)=\operatorname{tr}\left(A \lambda \lambda^{T}\right) \\
    & =\frac{1}{2}\left\{\log \frac{\left|\Sigma_{2}\right|}{\left|\Sigma_{1}\right|}-n+\operatorname{tr}\left(\Sigma_{2}^{-1} \Sigma_{1}\right)+\left(u_{2}-u_{1}\right)^{T} \Sigma_{2}^{-1}\left(u_{2}-u_{1}\right)\right\} \\
\end{aligned}
$$

其中$n=tr(I_{d})$即矩阵阶数、高斯分布的维度。


<!-- ### 加速Diffusion采样和方差的选择(DDIM)
DDPM的高质量生成依赖于较大的$T$(一般为1000或以上)，这就导致diffusion的前向过程非常缓慢。在Denoising Diffusion Implicit Model (DDIM)[2]中提出了一种牺牲多样性来换取更快推断的手段。

根据**特性2**和独立高斯分布的可加性，我们可以得到$\mathbf{x}_{t-1}$为：

$$
\begin{aligned}
    \mathbf{x}_{t-1} & = \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_{0} + \sqrt{1-\bar{\alpha}_{t-1}}\epsilon_{t-1} \\
    & = \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_{0} + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_{t}^{2}}\epsilon_{t} + \sigma_{t}^{2}\bar{\epsilon}_{t} \quad \text{ 独立高斯分布的可加性 }\\
    & = \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_{0} + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_{t}^{2}}\left(\frac{\mathbf{x}_{t} - \sqrt{\bar{\alpha}_{t}}\mathbf{x}_{0}}{\sqrt{1-\bar{\alpha}_{t}}}\right) + \sigma_{t}^{2}\epsilon_{t} \\
    q_{\sigma}(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}, \mathbf{x}_{0}) & = \mathcal{N}(\mathbf{x}_{t-1}; \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_{0} + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_{t}^{2}}\left(\frac{\mathbf{x}_{t} - \sqrt{\bar{\alpha}_{t}}\mathbf{x}_{0}}{\sqrt{1-\bar{\alpha}_{t}}}\right), \sigma_{t}^{2}\mathbf{I}).
\end{aligned}
$$

不同于(5)和(8)，上式将方差$\sigma_{t}^{2}$引入到了均值中，当$\sigma_{t}^{2} = \tilde{\beta}_{t} = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}}\beta_{t}$时，上式与(8)等价。

在DDIM中把由上式经过贝叶斯得到的$q_{\sigma}(\mathbf{x}_{t} \mid \mathbf{x}_{t-1}, \mathbf{x}_{0})$称为非马尔科夫链，因为$\mathbf{x}_{t}$的概率同时依赖于$\mathbf{x}_{t-1}$和$\mathbf{x}_{0}$。DDIM进一步定义了$\sigma_{t}(\eta)^{2} = \eta \cdot \tilde{\beta}_{t}$。当$\eta=0$时，diffusion sample过程会丧失所有随机性从而得到一个deterministic的结果(但是可以改变$\mathbf{x}_{T}$)。而$\eta=1$则DDIM等价于DDPM(使用$\tilde{\beta}_{t}$作为方差的版本)。用随机性换取生成性功能的类似操作在GAN中也可以通过对Latent code操作实现。 -->

---
### 为什么DDPM一定要这么多次采样？
首先我们先思考一下，如何可以加快DDPM的生成效率。最容易想到的两种方法，减小$T$，“跳步”(i.e.不再严格按照$t$到$t-1$的顺序来采样)，下面我们依次讨论：

1. **减小$T$值行不行？**  
   不行，在DDPM中，我们有公式
   $$\mathbf{x}_{t} = \sqrt{\alpha_{t}}\mathbf{x}_{t-1} + \sqrt{1-\alpha_{t}}\epsilon_{t-1} \tag{1}$$
   其中$\alpha_{t} = 1 - \beta_{t}$，由于对于每个$t$，$1 - \alpha_{t}$都接近于0，$\alpha_{t}$都接近于1，这是为了：  
   - 噪声的方差$1 - \alpha_{t}$较小，保证前后分布均为正态分布的假设成立；
   - $\mathbf{x}_{t-1}$的系数$\sqrt{\alpha_{t}}$要尽量接近于1，因为这样才能保证$t$时刻的噪声图尽量保留$t-1$时刻的大体分布，不至于一步就破坏掉原图分布，这样就不好还原了。
  
   同时我们通过推导，可以得到：
   $$\mathbf{x}_{T} = \sqrt{\bar{\alpha}_{T}}\mathbf{x}_{0} + \sqrt{1 - \bar{\alpha}_{T}}\epsilon \tag{2}$$
   其中$\bar{\alpha}_{T} = \prod_{i=1}^{T}\alpha_{i}$，我们希望在$T$时刻，也就是最后时刻，我们的$\mathbf{x}_{T}$是尽量服从标准正态分布的。因此我们希望$\sqrt{\bar{\alpha}_{T}}$尽可能趋向于0，$\sqrt{1 - \bar{\alpha}_{T}}$尽可能趋近于1。在$\alpha_{t}$接近于1的前提下，只能让$T$尽可能的大，才能满足我们最终的需求，这也就是为什么$T$的取值不能太小的原因，因为$\alpha_{t}$不能太小且我们需要$\sqrt{\bar{\alpha}_{t}}$尽可能的趋近于0。
2. **为什么非要一步一步降噪，跳步行不行？**  
   不行，注意，我们的优化目标虽然是噪声$\epsilon$的MSE，但实际这只是重参数的结果。我们的终极优化目标实际上是去拟合概率分布$q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}, \mathbf{x}_{0}\right)$。而我们求得这个分布均值的方法是用的贝叶斯公式：
   $$q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}, \mathbf{x}_{0}\right) = q(\mathbf{x}_{t} \mid \mathbf{x}_{t-1}, \mathbf{x}_{0}) \frac{q(\mathbf{x}_{t-1} \mid \mathbf{x}_{0})}{q(\mathbf{x}_{t} \mid \mathbf{x}_{0})} = q(\mathbf{x}_{t} \mid \mathbf{x}_{t-1}) \frac{q(\mathbf{x}_{t-1} \mid \mathbf{x}_{0})}{q(\mathbf{x}_{t} \mid \mathbf{x}_{0})} \tag{3}$$
   其中，等式右边的三个概率分布，都是通过一个非常重要的假设得到的，那就是我们的**马尔科夫性质**！因此我们的采样必须要严格遵照马尔科夫性质，也就是必须从$\mathbf{x}_{t}$到$\mathbf{x}_{t-1}$一步一步来，不能“跳步”。
3. **由式(2)，如果我们预测出了$\epsilon$，直接移项由$\mathbf{x}_{t}$直接得出$\mathbf{x}_{0}$行不行？**  
   不行，原因同问题2一样。


**既然不能跳步的原因是马尔科夫性质的约束，那假设我们不再让这个过程是马尔科夫的(Non-Markovian)了，现在我们可不可以跳步了？答案是可以的，并且这就是DDIM所做的事情**

---
### DDIM是如何去马尔科夫化的？
由于我们现在假设我们不再有马尔科夫性质，因此式$(3)$要改写成如下形式:

$$q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}, \mathbf{x}_{0}\right) = q(\mathbf{x}_{t} \mid \mathbf{x}_{t-1}, \mathbf{x}_{0}) \frac{q(\mathbf{x}_{t-1} \mid \mathbf{x}_{0})}{q(\mathbf{x}_{t} \mid \mathbf{x}_{0})} \tag{4}$$

由假设不再是马尔科夫链，等号右边的三个分布，我们全都不知道了（因为式$(2)$是通过马尔科夫性质推导出来的），这怎么办？首先我们提出一个大胆的命题：**前向过程$q(\mathbf{x}_{t} \mid \mathbf{x}_{t-1}, \mathbf{x}_{0})$一点都不重要，我们不需要知道。**（因为在DDPM的训练过程中根本没用到$q\left(\mathbf{x}_{t} \mid \mathbf{x}_{t-1}\right)$，而是直接用的$q(\mathbf{x}_{t} \mid \mathbf{x}_{0})$一步到位。）

而对于$q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}, \mathbf{x}_{0}\right)$的计算，我们可以假设一个分布$q_{\sigma}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}, \mathbf{x}_{0}\right)$，并且满足式子$(2)$中的$\mathbf{x}_{T} = \sqrt{\bar{\alpha}_{T}}\mathbf{x}_{0} + \sqrt{1 - \bar{\alpha}_{T}}\epsilon$依然成立。因为我们的前向训练过程是用到了$(2)$式的，所以为了让前向过程不变，我们要确保(2)式依然成立。

我们可以通过待定系数法来求解，假设$q_{\sigma}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}, \mathbf{x}_{0}\right) \sim \mathcal{N}(k\mathbf{x}_{0}+m\mathbf{x}_{t}, \sigma^2\mathbf{I})$，于是我们有$\mathbf{x}_{t-1} = k\mathbf{x}_{0}+m\mathbf{x}_{t}+\sigma\epsilon$，由于我们假设$(2)$式依然成立，于是我们有$\mathbf{x}_{t} = \sqrt{\bar{\alpha}_{t}}\mathbf{x}_{0} + \sqrt{1 - \bar{\alpha}_{t}}\epsilon'$，其中$\epsilon, \epsilon'$都是服从标准正态分布。于是我们可以代入求解，有：

$$
\begin{aligned}
    \mathbf{x}_{t-1} & = k\mathbf{x}_{0} + m\left(\sqrt{\bar{\alpha}_{t}}\mathbf{x}_{0} + \sqrt{1 - \bar{\alpha}_{t}}\epsilon'\right)+\sigma\epsilon \\
    & = (k + m\sqrt{\bar{\alpha}_{t}})\mathbf{x}_{0} + (m\sqrt{1-\bar{\alpha}_{t}})\epsilon' + \sigma\epsilon \\
    & = (k + m\sqrt{\bar{\alpha}_{t}})\mathbf{x}_{0} + \sqrt{m^2(1-\bar{\alpha}_{t})+ \sigma^2}\epsilon-- \text{高斯分布可加性} \tag{5}
\end{aligned}
$$

接下来我们求解$k, m$，因为必须满足式$(2)$，因此我们要满足：

$$
\left\{
    \begin{aligned}
        & k + m\sqrt{\bar{\alpha}_{t}} = \sqrt{\bar{\alpha}_{t-1}} \\
        & m^2(1-\bar{\alpha}_{t})+ \sigma^2 = 1 - \bar{\alpha}_{t-1}
    \end{aligned}
\right. \tag{6}
$$

通过求解上式我们可以得到：

$$
\left\{
    \begin{aligned}
        & m = \frac{\sqrt{1-\bar{\alpha}_{t-1}-\sigma^2}}{\sqrt{1-\bar{\alpha}_{t}}} \\
        & k = \sqrt{\bar{\alpha}_{t-1}} - \sqrt{1-\bar{\alpha}_{t-1}-\sigma^{2}}\frac{\sqrt{\bar{\alpha}_{t}}}{\sqrt{1-\bar{\alpha}_{t}}}
    \end{aligned}
\right.
$$

最终我们可以得到新的$q_{\sigma}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}, \mathbf{x}_{0}\right)$分布，即：

$$
q_{\sigma}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}, \mathbf{x}_{0}\right) = \mathcal{N}\left(\mathbf{x}_{t-1}; \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_{0}+\sqrt{1-\bar{\alpha}_{t-1}-\sigma^{2}}\frac{\mathbf{x}_{t}-\sqrt{\bar{\alpha}_{t}}\mathbf{x}_{0}}{\sqrt{1-\bar{\alpha}_{t}}}, \sigma^{2}\mathbf{I}\right) \tag{7}
$$

这就是我们新的反向生成分布，也就是我们**新的要去拟合的“终极目标”**。当然，有了式$(7)$，通过式$(4)$我们也可以求出$q(\mathbf{x}_{t} \mid \mathbf{x}_{t-1}, \mathbf{x}_{0})$.

---
### DDIM的采样过程
这部分和DDPM是完全一样的（除了采样分布变成了$(7)$）。同样的，对式$(7)$中的均值进行重参数化，用$\mathbf{x}_{t}, \epsilon_{\theta}$来表示，有：

$$
\begin{aligned}
    \mathbf{x}_{t-1} & = \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_{0} + \sqrt{1-\bar{\alpha}_{t-1}-\sigma^{2}}\frac{\mathbf{x}_{t}-\sqrt{\bar{\alpha}_{t}}\mathbf{x}_{0}}{\sqrt{1-\bar{\alpha}_{t}}} + \sigma\epsilon \\
    & = \sqrt{\bar{\alpha}_{t-1}}\left(\frac{\mathbf{x}_{t} - \sqrt{1-\bar{\alpha}_{t}}\epsilon_{\theta}(\mathbf{x}_{t})}{\sqrt{\bar{\alpha}_{t}}}\right) + \sqrt{1-\bar{\alpha}_{t-1}-\sigma^{2}}\epsilon_{\theta}(\mathbf{x}_{t}) + \sigma\epsilon \tag{8}
\end{aligned}
$$

其中，由**特性2**可得：$\mathbf{x}_{t} = \sqrt{\bar{\alpha}_{t}}\mathbf{x}_{0} + \sqrt{1 - \bar{\alpha}_{t}}\epsilon_{\theta}(\mathbf{x}_{t})$

由于我们不再需要上式具有马尔科夫性质，因此我们可以将上式改写为：

$$
\mathbf{x}_{s} = \sqrt{\bar{\alpha}_{s}}\left(\frac{\mathbf{x}_{k} - \sqrt{1-\bar{\alpha}_{k}}\epsilon_{\theta}(\mathbf{x}_{k})}{\sqrt{\bar{\alpha}_{k}}}\right) + \sqrt{1-\bar{\alpha}_{s}-\sigma^{2}}\epsilon_{\theta}(\mathbf{x}_{k}) + \sigma\epsilon \tag{9}
$$

其中严格满足$s < k$。于是，我们就可以从时间序列$\{1, \ldots, T\}$中随机去一个长度为$l$的升序子序列，通过式$(9)$迭代采样$l$次最终得到我们想要的$\mathbf{x}_{0}$。

到目前为止，我们基本完成了DDIM原理的推导，目前还有一个问题没解决，那就是$\sigma$的取值问题。原文的appendixB证明了无论$\sigma$取值为何，都不影响式（2）的成立。因此我们可以较为随意的取值。我们第一时间想到的就是令$\sigma^2=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}}\beta_{t}$，即DDPM中$q(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}, \mathbf{x}_{0})$的方差。

于是作者令$\sigma=\eta\sqrt{\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}}\beta_{t}}$，其中$\eta \in [0, 1]$。考虑两个边界值，

当$\eta = 1$时，我们就回到了DDPM。

- 当$\sigma=\eta\sqrt{\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}}\beta_{t}}$时，DDIM等价于DDPM。  
  证：  
  首先，对比DDPM和DDIM的分布$p(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}, \mathbf{x}_{0})$，其方差相等，因此只需证明均值相等即可，即下式子：
  $$
  \sqrt{\bar{\alpha}_{t-1}}\left(\frac{\mathbf{x}_{t} - \sqrt{1-\bar{\alpha}_{t}}\epsilon_{\theta}(\mathbf{x}_{t})}{\sqrt{\bar{\alpha}_{t}}}\right) + \sqrt{1-\bar{\alpha}_{t-1}-\sigma^{2}}\epsilon_{\theta}(\mathbf{x}_{t})  = \frac{1}{\sqrt{\alpha_{t}}}\left(\mathbf{x}_{t} - \frac{\beta_{t}}{\sqrt{1-\bar{\alpha_{t}}}}\epsilon_{\theta}(\mathbf{x}_{t})\right)
  $$

  下面证明：
  $$
  \begin{aligned}
    & \sqrt{\bar{\alpha}_{t-1}}\left(\frac{\mathbf{x}_{t}-\sqrt{1-\bar{\alpha}_{t}} \epsilon_{\theta}\left(\mathbf{x}_{t}\right)}{\sqrt{\bar{\alpha}_{t}}}\right)+\sqrt{1-\bar{\alpha}_{t-1}-\sigma^{2}} \epsilon_{\theta}\left(x_{t}\right) \\
    = & \frac{1}{\sqrt{\alpha_{t}}} x_{t}-\frac{1}{\sqrt{\alpha_{t}}} \sqrt{1-\bar{\alpha}_{t}} \epsilon_{\theta}\left(x_{t}\right)+\frac{\left(1-\bar{\alpha}_{t-1}\right) \sqrt{\alpha_{t}}}{\sqrt{1-\bar{\alpha}_{t}}} \epsilon_{\theta}\left(x_{t}\right) \\
    = & \frac{1}{\sqrt{\alpha_{t}}} x_{t}-\left(\frac{1}{\sqrt{\alpha_{t}}} \sqrt{1-\bar{\alpha}_{t}}-\frac{\left(1-\bar{\alpha}_{t-1}\right) \sqrt{\alpha_{t}}}{\sqrt{1-\bar{\alpha}_{t}}}\right) \epsilon_{\theta}\left(x_{t}\right) \\
    = & \frac{1}{\sqrt{\alpha_{t}}} x_{t}-\left(\frac{\left(1-\bar{\alpha}_{t}\right)-\left(1-\bar{\alpha}_{t-1}\right) \alpha_{t}}{\sqrt{\alpha_{t}} \sqrt{1-\bar{\alpha}_{t}}}\right) \epsilon_{\theta}\left(x_{t}\right) \\
    = & \frac{1}{\sqrt{\alpha_{t}}} x_{t}-\left(\frac{1-\bar{\alpha}_{t}-\alpha_{t}+\bar{\alpha}_{t}}{\sqrt{\alpha_{t}} \sqrt{1-\bar{\alpha}_{t}}}\right) \epsilon_{\theta}\left(x_{t}\right) \\
    = & \frac{1}{\sqrt{\alpha_{t}}} x_{t}-\left(\frac{1-\alpha_{t}}{\sqrt{\alpha_{t}} \sqrt{1-\bar{\alpha}_{t}}}\right) \epsilon_{\theta}\left(x_{t}\right) \\
    = & \frac{1}{\sqrt{\alpha_{t}}}\left(x_{t}-\frac{\beta_{t}}{\sqrt{1-\bar{\alpha}_{t}}}\right) \epsilon_{\theta}\left(x_{t}\right)
  \end{aligned}
  $$
  
  其中：
  $$
  \begin{aligned}
    \sqrt{1-\bar{\alpha}_{t-1}-\sigma^{2}} & = \frac{\sqrt{1-\bar{\alpha}_{t}}}{\sqrt{1-\bar{\alpha}_{t}}} \sqrt{1-\bar{\alpha}_{t-1}-\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}}\left(1-\alpha_{t}\right)} \\
    & = \frac{\sqrt{\left(1-\bar{\alpha}_{t-1}\right)\left(1-\frac{1}{1-\bar{\alpha}_{t}}\left(1-\alpha_{t}\right)\right)\left(1-\bar{\alpha}_{t}\right)}}{\sqrt{1-\bar{\alpha}_{t}}} \\
    & = \frac{\sqrt{\left(1-\bar{\alpha}_{t-1}\right)\left(1-\bar{\alpha}_{t}-1+\alpha_{t}\right)}}{\sqrt{1-\bar{\alpha}_{t}}} \\
    & = \frac{\sqrt{\left(1-\bar{\alpha}_{t-1}\right)\left(\alpha_{t}-\bar{\alpha}_{t}\right)}}{\sqrt{1-\bar{\alpha}_{t}}} \\
    & = \frac{\sqrt{\left(1-\bar{\alpha}_{t-1}\right)\left(1-\bar{\alpha}_{t-1}\right) \alpha_{t}}}{\sqrt{1-\bar{\alpha}_{t}}} \\
    & = \frac{\left(1-\bar{\alpha}_{t-1}\right) \sqrt{\alpha_{t}}}{\sqrt{1-\bar{\alpha}_{t}}} \\
  \end{aligned}
  $$
  

当$\eta = 0$时，这就是DDIM，因此DDIM其实并不是一个模型，只是一个特殊的采样方式。而当$\eta = 0$时，由于式$(9)$中唯一具有随机性的$\sigma\epsilon$此时亦为0，因此采样过程不再具有随机性，每个$\mathbf{x}_{T}$对应了确定的（deterministic）$\mathbf{x}_{0}$，这就有点类似GAN和VAE了。

当步数$l$很小时，$\eta = 0$效果最好，并且当$\eta = 0$时，20步的生成结果与100步的生成结果一致性很强，这是显然的，因为此时模型变为了确定性模型（deterministic），但是这里面值得关注的是，由于当$\eta = 0$时，每个$\mathbf{x}_{T}$对应唯一的$\mathbf{x}_{0}$，就像我们上面说的，这有点类似GAN和VAE，那我们可以认为此时的$\mathbf{x}_{T}$就是一个high-level的图像编码向量，里面可能蕴涵了大量的信息特征，也许可以用于其他下游任务。最后，作者论述了当$\eta = 0$时，式$(9)$可以写成常微分方程的形式，因此可以理解为模型是在用欧拉法近似从$\mathbf{x}_{0}$到$\mathbf{x}_{T}$的编码函数。

- **DDPM中如果在采样时令$\sigma = 0$，也就是也让它变成deterministic，为什么效果很差？**  
  原因：在DDPM中，$q(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}, \mathbf{x}_{0})$的方差$\sigma$时我们通过式$(3)$贝叶斯公式计算出来的，并不是像DDIM一样假设来的，换句话说，DDIM的$\sigma$取何值也不影响边界分布$\mathbf{x}_{T} = \sqrt{\bar{\alpha}_{T}} + \sqrt{1-\bar{\alpha}_{T}}\epsilon$，但DDPM中是不可以改的，改了就不再遵循前向过程了，也就破坏了原本的分布。

---
### 总结

由于DDPM是基于马尔可夫链建立起来的前向/逆向过程，因此不能“跳步”生成图像，且为了保证$\mathbf{x}_{T} \sim \mathcal{N}(0, \mathbf{I})$，$T$不能过小，因此自然会导致采样慢、出图效率低的缺点。而DDIM这篇文章介绍的方法，巧妙地通过自行设计优化目标$q(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}, \mathbf{x}_{0})$，将马尔可夫的限制取消，在不影响DDPM中的边界分布（i.e. 式$(2)$）的情况下大大缩短了采样步数。这样做的好处是，**训练好的DDPM可以直接拿来通过DDIM的采样方法进行采样，不需要再去训练一次。**

---
## DDIM原理

---
### 概述
用一段话大概描述DDIM做了件什么事情：

DDPM中的推导路线：

$$
q(\mathbf{x}_{t} \mid \mathbf{x}_{t-1}) \xrightarrow{\text{inference}} q(\mathbf{x}_{t} \mid \mathbf{x}_{0}) \xrightarrow{\text{inference}} q(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}, \mathbf{x}_{0}) \xrightarrow{\text{approximate}} p(\mathbf{x}_{t-1} \mid \mathbf{x}_{t})
$$

由DDPM的推导路线可知：
1. 损失函数只依赖于边际分布$q(\mathbf{x}_{t} \mid \mathbf{x}_{0})$
2. 采样过程只依赖于$p(\mathbf{x}_{t-1} \mid \mathbf{x}_{t})$

因此虽然DDPM是从$q(\mathbf{x}_{t} \mid \mathbf{x}_{t-1})$为出发点一步步往前推的，但是最后的结果与它无关，所以可以把它从整个过程中忽略。

在DDPM中，扩散过程（前向过程）定义为一个Markov Chain:

$$ q\left(\mathbf{x}_{1: T} \mid \mathbf{x}_{0}\right)=\prod_{t=1}^{T} q\left(\mathbf{x}_{t} \mid \mathbf{x}_{t-1}\right) \qquad q\left(\mathbf{x}_{t} \mid \mathbf{x}_{t-1}\right)=\mathcal{N}\left(\mathbf{x}_{t} ; \sqrt{\frac{\alpha_{t}}{\alpha_{t-1}}} \mathbf{x}_{t-1},\left(1-\frac{\alpha_{t}}{\alpha_{t-1}}\right) \mathbf{I}\right) $$

注意，在DDIM中，$\alpha_{t}$其实是DDPM中的$\bar{\alpha_{t}}=\prod_{i=1}^{t}\alpha_{i}=\prod_{i=1}^{t}(1-\beta_{t})$，那么DDPM中的前向过程$\beta_{t}$就为:

$$ \beta_{t}=(1-\frac{\alpha_{t}}{\alpha_{t-1}}) $$

扩散过程的一个重要特征是可以直接用$\mathbf{x}_{0}$来对任意的$\mathbf{x}_{t}$进行采样：

$$ q\left(\mathbf{x}_{t} \mid \mathbf{x}_{0}\right)=\mathcal{N}\left(\mathbf{x}_{t} ; \sqrt{\alpha_{t}}\mathbf{x}_{0}, \left(1-\alpha_{t}\right)\mathbf{I}\right) $$

而DDPM的反向过程也定义为一个Markov Chain:

$$ p_\theta\left(\mathbf{x}_{0: T}\right) =p\left(\mathbf{x}_T\right) \prod_{t=1}^T p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right) \qquad p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right) =\mathcal{N}\left(\mathbf{x}_{t-1} ; \boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right), \boldsymbol{\Sigma}_\theta\left(\mathbf{x}_t, t\right)\right) $$

这里用神经网络$p_{\theta}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}\right)$来拟合真实的分布$q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}\right)$。DDPM的前向过程和反向过程如下所示：

<div style="text-align:center">
<img src="./figs/ddpm_forward_reverse_process.png">
</div>  
 
进一步我们发现后验分布$q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}, \mathbf{x}_{0}\right)$是一个可获取的高斯分布：

$$
q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}, \mathbf{x}_{0}\right) = \mathcal{N}\left(\mathbf{x}_{t-1}; \tilde{\mu}\left(\mathbf{x}_{t}, \mathbf{x}_{0}\right), \tilde{\beta_{t}}\mathbf{I}\right)
$$

其中这个高斯分布的方差是定值，而均值是一个依赖$\mathbf{x}_{0}$和$\mathbf{x}_{t}$的组合函数：

$$
\tilde{\boldsymbol{\mu}}_{t}\left(\mathbf{x}_{t}, \mathbf{x}_{0}\right)=\frac{\sqrt{\alpha_{t}}\left(1-\alpha_{t-1}\right)}{\sqrt{\alpha_{t-1}}\left(1-\alpha_{t}\right)} \mathbf{x}_{t}+\frac{\sqrt{\alpha_{t-1}} \beta_{t}}{1-\alpha_{t}} \mathbf{x}_{0}
$$

基于变分法得到如下的优化目标：

 $$
\begin{aligned}
    L_{\mathrm{VLB}} & =\mathbb{E}_{q\left(\mathbf{x}_{0 T T}\right)}\left[\log \frac{q\left(\mathbf{x}_{1: T} \mid \mathbf{x}_0\right)}{p_\theta\left(\mathbf{x}_{0: T}\right)}\right] \\
    & =\mathbb{E}_q\left[\log \frac{\prod_{t=1}^T q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right)}{p_\theta\left(\mathbf{x}_T\right) \prod_{t=1}^T p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)}\right] \\
    & =\mathbb{E}_q\left[-\log p_\theta\left(\mathbf{x}_T\right)+\sum_{t=1}^T \log \frac{q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right)}{p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)}\right] \\
    & =\mathbb{E}_q\left[-\log p_\theta\left(\mathbf{x}_T\right)+\sum_{t=2}^T \log \frac{q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right)}{p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)}+\log \frac{q\left(\mathbf{x}_1 \mid \mathbf{x}_0\right)}{p_\theta\left(\mathbf{x}_0 \mid \mathbf{x}_1\right)}\right] \\
    & =\mathbb{E}_q\left[-\log p_\theta\left(\mathbf{x}_T\right)+\sum_{t=2}^T \log \left(\frac{q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right)}{p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)} \cdot \frac{q\left(\mathbf{x}_t \mid \mathbf{x}_0\right)}{q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_0\right)}\right)+\log \frac{q\left(\mathbf{x}_1 \mid \mathbf{x}_0\right)}{p_\theta\left(\mathbf{x}_0 \mid \mathbf{x}_1\right)}\right] \\
    & =\mathbb{E}_q\left[-\log p_\theta\left(\mathbf{x}_T\right)+\sum_{t=2}^T \log \frac{q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right)}{p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)}+\sum_{t=2}^T \log \frac{q\left(\mathbf{x}_t \mid \mathbf{x}_0\right)}{q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_0\right)}+\log \frac{q\left(\mathbf{x}_1 \mid \mathbf{x}_0\right)}{p_\theta\left(\mathbf{x}_0 \mid \mathbf{x}_1\right)}\right] \\
    & =\mathbb{E}_q\left[-\log p_\theta\left(\mathbf{x}_T\right)+\sum_{t=2}^T \log \frac{q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right)}{p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)}+\log \frac{q\left(\mathbf{x}_T \mid \mathbf{x}_0\right)}{q\left(\mathbf{x}_1 \mid \mathbf{x}_0\right)}+\log \frac{q\left(\mathbf{x}_1 \mid \mathbf{x}_0\right)}{p_\theta\left(\mathbf{x}_0 \mid \mathbf{x}_1\right)}\right] \\
    & =\mathbb{E}_q\left[\log \frac{q\left(\mathbf{x}_T \mid \mathbf{x}_0\right)}{p_\theta\left(\mathbf{x}_T\right)}+\sum_{t=2}^T \log \frac{q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right)}{p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)}-\log p_\theta\left(\mathbf{x}_0 \mid \mathbf{x}_1\right)\right] \\
    & =\mathbb{E}_q[\underbrace{D_{\mathrm{KL}}\left(q\left(\mathbf{x}_T \mid \mathbf{x}_0\right) \| p_\theta\left(\mathbf{x}_T\right)\right)}_{L_{T}} + \sum_{t=2}^T \underbrace{D_{\mathrm{KL}}\left(q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right) \| p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)\right)}_{L_{t-1}}  \underbrace{- \log p_\theta\left(\mathbf{x}_0 \mid \mathbf{x}_1\right)}_{L_0}] \\
    & = \cancel{\underbrace{D_{\mathrm{KL}}\left(q\left(\mathbf{x}_T \mid \mathbf{x}_0\right) \| p_\theta\left(\mathbf{x}_T\right)\right)}_{L_{T}} + \sum_{t=2}^T \underbrace{D_{\mathrm{KL}}\left(q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right) \| p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)\right)}_{L_{t-1}}  \underbrace{- \log p_\theta\left(\mathbf{x}_0 \mid \mathbf{x}_1\right)}_{L_0}}
\end{aligned}
$$
 
根据两个高斯公式的KL公式，我们进一步得到：

$$
L_{t-1} = \mathbb{E}_{q\left(\mathbf{x}_{t} \mid \mathbf{x}_{0}\right)} \left[\frac{1}{2\sigma_{t}^{2}} \left\| \tilde{\mu_{t}}\left(\mathbf{x}_{t}, \mathbf{x}_{0}\right) - \mu_{\theta}\left(\mathbf{x}_{t}, t\right) \right\|^{2}\right]
$$

根据扩散模型的特性，我们通过重参数化可进一步简化上述目标：

$$
L_{t-1}=\mathbb{E}_{\mathbf{x}_{0}, \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})}\left[\frac{\beta_{t}^{2}}{2 \sigma_{t}^{2} \alpha_{t}\left(1-\bar{\alpha}_{t}\right)}\left\|\epsilon-\epsilon_{\theta}\left(\sqrt{\bar{\alpha}_{t}} \mathbf{x}_{0}+\sqrt{1-\bar{\alpha}_{t}} \epsilon, t\right)\right\|^{2}\right]
$$

如果去掉系数，那么我们就能够得到更简化的优化目标：

$$
L_{t-1}^{simple} = \mathbb{E}_{\mathbf{x}_{0}, \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \left[\left\|\epsilon-\epsilon_{\theta}\left(\sqrt{\bar{\alpha}_{t}} \mathbf{x}_{0}+\sqrt{1-\bar{\alpha}_{t}} \epsilon, t\right)\right\|^{2}\right]
$$

由DDPM的优化目标可知，DDPM其实仅仅依赖边缘分布$q\left(\mathbf{x}_{t} \mid \mathbf{x}_{0}\right)$，而并不是直接作用在联合分布$q\left(\mathbf{x}_{1:T} \mid \mathbf{x}_{0} \right)$。这带来的一个启示是：DDPM这个隐变量模型可以有很多推理分布来选择，只要推理分布满足边缘分布条件（扩散过程的特性）即可，而且这些推理过程不一定要是马尔科夫链。但值得注意的是，我们要得到DDPM的优化目标，还需要知道分布$q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}, \mathbf{x}_{0} \right)$，之前我们在根据贝叶斯公式推导这个分布时是知道分布$q\left(\mathbf{x}_{t} \mid \mathbf{x}_{t-1} \right)$的，而且依赖了前向过程的马尔科夫链特性。如果要解除对前向过程的依赖，那么我们就需要直接定义这个分布$q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}, \mathbf{x}_{0} \right)$。基于上述分析，DDIM中将推理分布定义为：

$$
q_{\sigma}\left(\mathbf{x}_{1:T} \mid \mathbf{x}_{0} \right) = q_{\sigma}\left(\mathbf{x}_{T} \mid \mathbf{x}_{0} \right) \prod_{t=2}^{T}q_{\sigma}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}, \mathbf{x}_{0} \right)
$$

这里要同时满足$q_{\sigma}\left(\mathbf{x}_{T} \mid \mathbf{x}_{0} \right) = \mathcal{N}\left(\sqrt{\alpha_{T}}\mathbf{x}_{0}, \left(1-\alpha_{T}\right)\mathbf{I} \right)$以及对于所有$t \geqslant 2$有：

$$
q_{\sigma}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}, \mathbf{x}_{0}\right) = \mathcal{N}\left(\mathbf{x}_{t-1}; \sqrt{\alpha_{t-1}}\mathbf{x}_{0}+\sqrt{1-\alpha_{t-1}-\sigma_{t}^{2}}\frac{\mathbf{x}_{t}-\sqrt{\alpha_{t}}\mathbf{x}_{0}}{\sqrt{1-\alpha_{t}}}, \sigma_{t}^{2}\mathbf{I}\right)
$$

这里的方差$\sigma_{t}^{2}$是一个实数，不同的设置就是不一样的分布，所以$q\left(\mathbf{x}_{1:T} \mid \mathbf{x}_{0} \right)$其实是一系列的推理分布。可以看到这里分布$q_{\sigma}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}, \mathbf{x}_{0} \right)$的均值也定义为一个依赖$\mathbf{x}_{0}$和$\mathbf{x}_{t}$的组合函数，之所以定义为这样的形式，是因为根据$q_{\sigma}\left(\mathbf{x}_{t}, \mathbf{x}_{0} \right)$，我们可以通过数学归纳法证明，对于所有$t$均满足：

$$
q_{\sigma}\left(\mathbf{x}_{t}, \mathbf{x}_{0} \right) = \mathcal{N}\left(\mathbf{x}_{t}; \sqrt{\alpha_{t}}\mathbf{x}_{0}, \left(1-\alpha_{t}\right)\mathbf{I} \right)
$$

这里的证明见DDIM的附录部分，另外博客[生成扩散模型漫谈（四）：DDIM = 高观点DDPM](https://kexue.fm/archives/9181)也从待定系数法来证明了$q_{\sigma}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}, \mathbf{x}_{0}\right)$要构造的形式。可以看到这里定义的推理分布$q_{\sigma}\left(\mathbf{x}_{1:T} \mid \mathbf{x}_{0}\right)$并没有直接定义前向过程，但这里满足了我们要讨论的两个条件：边缘分布$q_{\sigma}\left(\mathbf{x}_{T} \mid \mathbf{x}_{0} \right) = \mathcal{N}\left(\mathbf{x}_{t}; \sqrt{\alpha_{T}}\mathbf{x}_{0}, \left(1-\alpha_{T}\right)\mathbf{I} \right)$，同时已知后验分布$q_{\sigma}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}, \mathbf{x}_{0}\right)$。同样地，我们可以按照和DDPM一样的方式去推导优化目标，最终也会得到同样的$L^{simple}$（虽然VLB的系数不同，论文3.2部分也证明了这个结论）。论文也给出了一个前向过程是非马尔科夫链的示例，如下如所示，这里前向过程是$q_{\sigma}\left(\mathbf{x}_{t} \mid \mathbf{x}_{t-1}, \mathbf{x}_{0}\right)$，由于生成$\mathbf{x}_{t}$不仅依赖于$\mathbf{x}_{t-1}$，而且依赖$\mathbf{x}_{0}$，所以是一个非马尔科夫链：

<div style="text-align:center">
<img src="./figs/ddim_demo.png">
</div>
  
注意，这里只是一个前向过程的示例，而实际上我们上述定义的推理分布并不需要前向过程就可以得到和DDPM一样的优化目标。与DDPM一样，这里也是用神经网络$\epsilon_{\theta}$来预测噪声，那么根据$q_{\sigma}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}, \mathbf{x}_{0} \right)$的形式，我们可以用如下公式来从$\mathbf{x}_{t}$生成$\mathbf{x}_{t-1}$:
  
$$
\mathbf{x}_{t-1}=\sqrt{\alpha_{t-1}}(\underbrace{\frac{\mathbf{x}_{t}-\sqrt{1-\alpha_{t}} \epsilon_{\theta}\left(\mathbf{x}_{t}, t\right)}{\sqrt{\alpha_{t}}}}_{\text {predicted } \mathbf{x}_{0}})+\underbrace{\sqrt{1-\alpha_{t-1}-\sigma_{t}^{2}} \cdot \epsilon_{\theta}\left(\mathbf{x}_{t}, t\right)}_{\text {direction pointing to } \mathbf{x}_{t}}+\underbrace{\sigma_{t} \epsilon_{t}}_{\text {random noise }}
$$

这里将生成过程分成三个部分：一是由预测的$\mathbf{x}_{0}$来产生的，二是由指向$\mathbf{x}_{t}$的部分，三是随机噪声（这里$\epsilon_{t}$是与$\mathbf{x}_{t}无关的噪声$）。论文将$\sigma_{t}^{2}$进一步定义为：

$$
\sigma_{t}^{2} = \eta \cdot \tilde{\beta_{t}} = \eta \cdot \sqrt{\left(1-\alpha_{t-1}\right)/\left(1-\alpha_{t}\right)}\sqrt{\left(1-\alpha_{t}/\alpha_{t-1}\right)}
$$

这里考虑两种情况，一是$\eta = 1$，此时$\sigma_{t}^{2} = \tilde{\beta_{t}}$，此时生成过程就和DDPM一样了，另外情况是$\eta = 0$，这个时候生成过程就没有随机噪声了，是一个确定性的过程，论文将这种情况下的模型成为**DDIM(Denoising Diffusion Implicit Model)**，一旦最初的随机噪声$\mathbf{x}_{T}$确定了，那么DDIM的样本生成就变成了确定的过程。

上面我们得到了DDIM模型，那么我们现在来看如何来加速生成过程。虽然DDIM和DDPM的训练过程一样，但是我们前面已经说了，DDIM并没有明确的前向过程，这意味着我们可以**定义一个更短步数的前向过程**。具体地，这里我们从原始的序列$[1, \ldots, \mathit{T}]$采样一个长度为$S$的子序列$[\mathbf{\tau}_{1}, \ldots, \mathbf{\tau}_{S}]$，我们将$\mathbf{x}_{\tau_{1}}, \ldots, \mathbf{x}_{\tau_{S}}$的前向过程定义为一个马尔科夫链，并且它们满足：$q\left(\mathbf{x}_{\tau_{i}} \mid \mathbf{x}_{0} \right) = \mathcal{N}\left(\mathbf{x}_{t}; \sqrt{\alpha_{\tau_{i}}}\mathbf{x}_{0}, \left(1-\alpha_{\tau_{i}}\right)\mathbf{I}\right)$。下图展示了一个具体的示例：

<div style="text-align:center">
<img src="./figs/ddim_fig2.png">
</div>

那么生成过程也可以用这个子序列的反向马尔科夫链来替代，由于$S$可以设置比原来的步数$L$要小，那么就可以加速生成过程，这里的生成过程变为：

$$
\mathbf{x}_{\tau_{i-1}}=\sqrt{\alpha_{\tau_{i-1}}}\left(\frac{\mathbf{x}_{\tau_{i}}-\sqrt{1-\alpha_{\tau_{i}}}\epsilon_{\theta}\left(\mathbf{x}_{\tau_{i}}, \tau_{i}\right)}{\sqrt{\alpha_{\tau_{i}}}}\right) + \sqrt{1-\alpha_{\tau_{i-1}}-\sigma_{\tau_{i}}^{2}} \cdot \epsilon_{\theta}\left(\mathbf{x}_{\tau_{i}}, \tau_{i}\right) + \sigma_{\tau_{i}}\epsilon
$$

其实上述的加速，我们是将前向过程按如下方式进行了分解：

$$
q_{\sigma, \tau}\left(\mathbf{x}_{1:T} \mid \mathbf{x}_{0}\right) = q_{\sigma, \tau}\left(\mathbf{x}_{T} \mid \mathbf{x}_{0}\right) \prod_{i=1}^{S}q_{\sigma}\left(\mathbf{x}_{\tau_{i-1}} \mid \mathbf{x}_{\tau_{i}}, \mathbf{x}_{0} \right) \prod_{t \in \bar{\tau}}q_{\sigma, \tau}\left(\mathbf{x}_{t} \mid \mathbf{x}_{0} \right)
$$

其中$\bar{\tau} = \{1, \ldots, \mathit{T}\} \backslash \tau$，这包含了两个图：其中一个就是由$\{\mathbf{x}_{\tau_{i}}\}_{i=1}^{S}$组成的马尔科夫链，另外一个是剩余的变量$\{\mathbf{x}_{t}\}_{t \in \bar{\tau}}$组成的星状图。同时生成过程，我们也只用马尔科夫链的那部分来生成：

$$
p_{\theta}\left(\mathbf{x}_{0:T}\right) = p\left(\mathbf{x_{T}}\right) \underbrace{\prod_{i=1}^{S}p_{\theta}\left(\mathbf{x}_{\tau_{i-1}} \mid \mathbf{x}_{\tau_{i}}\right)}_{\text{use to produce sample}} \times \underbrace{\prod_{t \in \tilde{\tau}}p_{\theta}\left(\mathbf{x}_{0} \mid \mathbf{x}_{t} \right)}_{\text{only for VLB}}
$$

论文共设计了两种方案来采样子序列，分别是：
- **Linear**: 采用线性的序列$\tau_{i} = \lfloor ci \rfloor$;
- **Quadratic**: 采用二次方的序列$\tau_{i} = \lfloor ci^{2} \rfloor$;
  
这里$c$是一个定值，它的设定使得$\tau_{-1}$最接近$\mathit{T}$。论文中只对CIFAR10数据集采用**Quadratic**序列，其它数据集均采用**Linear**序列。



## References
[1]: Feller, William. "On the theory of stochastic processes, with particular reference to applications." Selected Papers I. Springer, Cham, 2015. 769-798.  
[2]: Song, Jiaming, Chenlin Meng, and Stefano Ermon. "Denoising diffusion implicit models." arXiv preprint arXiv:2010.02502 (2020)  
[3]: Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." Advances in neural information processing systems 33 (2020): 6840-6851.  
[4]: Weng, Lilian. (Jul 2021). What are diffusion models? Lil’Log. https://lilianweng.github.io/posts/2021-07-11-diffusion-models/.