# Variational Autoencoders and Nonlinear ICA: A Unifying Framework

Ilyes Khemakhem
Gatsby Unit
UCL

Diederik P. Kingma
Google Brain

Ricardo Pio Monti
Gatsby Unit
UCL

Aapo Hyvärinen
Université Paris-Saclay,
Inria, Univ. of Helsinki

# Abstract

The framework of variational autoencoders allows us to efficiently learn deep latent-variable models, such that the model's marginal distribution over observed variables fits the data. Often, we're interested in going a step further, and want to approximate the true joint distribution over observed and latent variables, including the true prior and posterior distributions over latent variables. This is known to be generally impossible due to unidentifiability of the model. We address this issue by showing that for a broad family of deep latent-variable models, identification of the true joint distribution over observed and latent variables is actually possible up to very simple transformations, thus achieving a principled and powerful form of disentanglement. Our result requires a factorized prior distribution over the latent variables that is conditioned on an additionally observed variable, such as a class label or almost any other observation. We build on recent developments in nonlinear ICA, which we extend to the case with noisy or undercomplete observations, integrated in a maximum likelihood framework. The result also trivially contains identifiable flow-based generative models as a special case.

# 1 INTRODUCTION

The framework of variational autoencoders (Kingma and Welling, 2013; Rezende et al., 2014) (VAEs) and its extensions (e.g. Burda et al. (2015); Kingma et al.

(2016); Tucker et al. (2018); Maaløe et al. (2019)) offers a scalable set of techniques for learning deep latent-variable models and corresponding inference models. With VAEs, we can in principle learn flexible models of data such that, after optimization, the model's implicit marginal distribution over the observed variables approximates their true (but unknown) distribution. With VAEs we can also efficiently synthesize pseudodata from the model.

However, we're often interested in going a step further and want to learn the true joint distribution over both observed and latent variables. This is generally a very difficult task, since by definition we only ever observe the observed variables, never the latent variables, therefore we cannot directly estimate their joint distribution. If we could however somehow achieve this task and learn the true joint distribution, this would imply that we have also learned to approximate the true prior and posterior distributions over latent variables. Learning about these distributions can be very interesting for various purposes, for example in order to learn about latent structure behind the data, or in order to infer the latent variables from which the data originated.

Learning the true joint distribution is only possible when the model is identifiable, as we will explain. The original VAE theory doesn't tell us when this is the case; it only tells us how to optimize the model's parameters such that its (marginal) distribution over the observed variables matches the data. The original theory doesn't tell us if or when we learn the correct joint distribution over observed and latent variables.

Almost no literature exists on achieving this goal. A pocket of the VAE literature works towards the related goal of disentanglement, but offers no proofs or theoretic guarantees of identifiability of the model or its latent variables. The most prominent of such models are  $\beta$ -VAEs and their extensions (Burgess et al., 2018; Higgins et al., 2016, 2018; Esmaeili et al., 2018; Kim and Mnih, 2018; Chen et al., 2018), in which the authors introduce adjustable hyperparameters in the VAE objective to encourage disentanglement. Other

Proceedings of the
23^{\mathrm{rd}}
International Conference on Artificial Intelligence and Statistics (AISTATS) 2020, Palermo, Italy. PMLR: Volume 108. Copyright 2020 by the author(s). * This is a slightly updated version of the published manuscript. See Corrigendum at the end of the paper.

---

work attempts to find maximally independent components through the GAN framework *(Brakel and Bengio, 2017)*. However, models in these earlier works are actually non-identifiable due to non-conditional latent priors, as has been seen empirically *(Locatello et al., 2018)*, and we will show formally below.

Recent work in nonlinear Independent Component Analysis (ICA) theory *(Hyvärinen and Morioka, 2016, 2017; Hyvärinen et al., 2019)* provided the first identifiability results for deep latent-variable models. Nonlinear ICA provides a rigorous framework for recovering independent latents that were transformed by some invertible nonlinear transformation into the data. Some special but not very restrictive conditions are necessary, since it is known that when the function from latent to observed variables is nonlinear, the general problem is ill-posed, and one cannot recover the independent latents *(Hyvärinen and Pajunen, 1999)*. However, existing nonlinear ICA methods do not learn to model the data distribution (pdf), nor do they allow us to synthesize pseudo-data.

In this paper we show that under relatively mild conditions the joint distribution over observed and latent variables in VAEs is identifiable and learnable, thus bridging the gap between VAEs and nonlinear ICA. To this end, we establish a principled connection between VAEs and an identifiable nonlinear ICA model, providing a unified view of two complementary methods in unsupervised representation learning. This integration is achieved by using a latent prior that has a factorized distribution that is conditioned on additionally observed variables, such as a class label, time index, or almost any other further observation. Our theoretical results trivially apply to any consistent parameter estimation method for deep latent-variable models, not just the VAE framework. We found the VAE a logical choice since it allows for efficient latent-variable inference and scales to large datasets and models. Finally, we put our theoretical results to a test in experiments. Perhaps most notably, we find that on a synthetic dataset with known ground-truth model, our method with an identifiable VAE indeed learns to closely approximate the true joint distribution over observed and latent variables, in contrast with a baseline non-identifiable model.

## 2 UNIDENTIFIABILITY OF DEEP LATENT VARIABLE MODELS

### 2.1 Deep latent variable models

Consider an observed data variable (random vector) $\mathbf{x}\in\mathbb{R}^{d}$, and a latent random vector $\mathbf{z}\in\mathbb{R}^{n}$. A common deep latent variable model has the following structure:

$p_{\bm{\theta}}(\mathbf{x},\mathbf{z})=p_{\bm{\theta}}(\mathbf{x}|\mathbf{z})p_{\bm{\theta}}(\mathbf{z})$ (1)

where $\bm{\theta}\in\Theta$ is a vector of parameters, $p_{\bm{\theta}}(\mathbf{z})$ is called a prior distribution over the latent variables. The distribution $p_{\bm{\theta}}(\mathbf{x}|\mathbf{z})$, often parameterized with a neural network called the decoder, tells us how the distribution on $\mathbf{x}$ depends on the values of $\mathbf{z}$. The model then gives rise to the observed distribution of the data as:

$p_{\bm{\theta}}(\mathbf{x})=\int p_{\bm{\theta}}(\mathbf{x},\mathbf{z})\mathrm{d}\mathbf{z}$ (2)

Assuming $p_{\bm{\theta}}(\mathbf{x}|\mathbf{z})$ is modelled by a deep neural network, this can model a rich class of data distributions $p_{\bm{\theta}}(\mathbf{x})$.

We assume that we observe data which is generated from an underlying joint distribution $p_{\bm{\theta}^{*}}(\mathbf{x},\mathbf{z})=p_{\bm{\theta}^{*}}(\mathbf{x}|\mathbf{z})p_{\bm{\theta}^{*}}(\mathbf{z})$ where $\bm{\theta}^{*}$ are its true but unknown parameters. We then collect a dataset of observations of $\mathbf{x}$:

$\mathcal{D}=\{\mathbf{x}^{(1)},\ldots,\mathbf{x}^{(N)}\}\;\;\text{where}\;\;\mathbf{z}^{*(i)}\sim p_{\bm{\theta}^{*}}(\mathbf{z})$
$\mathbf{x}^{(i)}\sim p_{\bm{\theta}^{*}}(\mathbf{x}|\mathbf{z}^{*(i)})$

Note that the original values $\mathbf{z}^{*(i)}$ of the latent variables $\mathbf{z}$ are by definition not observed and unknown. The ICA literature, including this work, uses the term sources to refer to $\mathbf{z}^{*(i)}$. Also note that we could just as well have written: $\mathbf{x}^{(i)}\sim p_{\bm{\theta}^{*}}(\mathbf{x})$.

The VAE framework *(Kingma and Welling, 2013; Rezende et al., 2014)* allows us to efficiently optimize the parameters $\bm{\theta}$ of such models towards the (approximate) maximum marginal likelihood objective, such that after optimization:

$p_{\bm{\theta}}(\mathbf{x})\approx p_{\bm{\theta}^{*}}(\mathbf{x})$ (3)

In other words, after optimization we have then estimated the marginal density of $\mathbf{x}$.

### 2.2 Parameter Space vs Function Space

In this work we use slightly non-standard notation and nomenclature: we use $\bm{\theta}\in\Theta$ to refer to the model parameters in function space. In contrast, let $\mathbf{w}\in W$ refer to the space of original neural network parameters (weights, biases, etc.) in which we usually perform gradient ascent.

### 2.3 Identifiability

The VAE model actually learns a full generative model $p_{\bm{\theta}}(\mathbf{x},\mathbf{z})=p_{\bm{\theta}}(\mathbf{x}|\mathbf{z})p_{\bm{\theta}}(\mathbf{z})$ and an inference model $q_{\phi}(\mathbf{z}|\mathbf{x})$ that approximates its posterior $p_{\bm{\theta}}(\mathbf{z}|\mathbf{x})$. The problem is that we generally have no guarantees about what these

---

learned distributions actually are: all we know is that the marginal distribution over $\mathbf{x}$ is meaningful (Eq. 3). The rest of the learned distributions are, generally, quite meaningless.

What we are looking for is models for which the following implication holds for all $(\mathbf{x},\mathbf{z})$:

$\forall(\boldsymbol{\theta},\boldsymbol{\theta}^{\prime}):\ \ p_{\boldsymbol{\theta}}(\mathbf{x})=p_{\boldsymbol{\theta}^{\prime}}(\mathbf{x})\ \ \implies\ \ \boldsymbol{\theta}=\boldsymbol{\theta}^{\prime}$ (4)

That is: if any two different choices of model parameter $\boldsymbol{\theta}$ and $\boldsymbol{\theta}^{\prime}$ lead to the same marginal density $p_{\boldsymbol{\theta}}(\mathbf{x})$, then this would imply that they are equal and thus have matching joint distributions $p_{\boldsymbol{\theta}}(\mathbf{x},\mathbf{z})$. This means that if we learn a parameter $\boldsymbol{\theta}$ that fits the data perfectly: $p_{\boldsymbol{\theta}}(\mathbf{x})=p_{\boldsymbol{\theta}^{*}}(\mathbf{x})$ (the ideal case of Eq. 3), then its joint density also matches perfectly: $p_{\boldsymbol{\theta}}(\mathbf{x},\mathbf{z})=p_{\boldsymbol{\theta}^{*}}(\mathbf{x},\mathbf{z})$. If the joint density matches, this also means that we found the correct prior $p_{\boldsymbol{\theta}}(\mathbf{z})=p_{\boldsymbol{\theta}^{*}}(\mathbf{z})$ and correct posteriors $p_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{x})=p_{\boldsymbol{\theta}^{*}}(\mathbf{z}|\mathbf{x})$. In case of VAEs, we can then also use the inference model $q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x})$ to efficiently perform inference over the sources $\mathbf{z}^{*}$ from which the data originates.

The general problem here is a lack of *identifiability* guarantees of the deep latent-variable model. We illustrate this by showing that any model with unconditional latent distribution $p_{\boldsymbol{\theta}}(\mathbf{z})$ is unidentifiable, i.e. that Eq. (4) does not hold. In this case, we can always find transformations of $\mathbf{z}$ that changes its value but does not change its distribution. For a spherical Gaussian distribution $p_{\boldsymbol{\theta}}(\mathbf{z})$, for example, applying a rotation keeps its distribution the same. We can then incorporate this transformation as the first operation in $p_{\boldsymbol{\theta}}(\mathbf{x}|\mathbf{z})$. This will not change $p_{\boldsymbol{\theta}}(\mathbf{x})$, but it will change $p_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{x})$, since now the values of $\mathbf{x}$ come from different values of $\mathbf{z}$. This is an example of a broad class of commonly used models that are non-identifiable. We show rigorously in Supplementary Material D that, in fact, models with *any* form of unconditional prior $p_{\boldsymbol{\theta}}(\mathbf{z})$ are unidentifiable.

## 3 AN IDENTIFIABLE MODEL BASED ON CONDITIONALLY FACTORIAL PRIORS

In this section, we define a broad family of deep latent-variable models which is identifiable, and we show how to estimate the model and its posterior through the VAE framework. We call this family of models, together with its estimation method, Identifiable VAE, or iVAE for short.

### 3.1 Definition of proposed model

The primary assumption leading to identifiability is a conditionally factorized prior distribution over the latent variables $p_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{u})$, where $\mathbf{u}$ is an additionally observed variable *(Hyvärinen et al., 2019)*. The variable $\mathbf{u}$ could be, for example, the time index in a time series *(Hyvärinen and Morioka, 2016)*, previous data points in a time series, some kind of (possibly noisy) class label, or another concurrently observed variable.

Formally, let $\mathbf{x}\in\mathbb{R}^{d}$, and $\mathbf{u}\in\mathbb{R}^{m}$ be two observed random variables, and $\mathbf{z}\in\mathbb{R}^{n}$ (lower-dimensional, $n\leq d$) a latent variable. Let $\boldsymbol{\theta}=(\mathbf{f},\mathbf{T},\boldsymbol{\lambda})$ be the parameters of the following conditional generative model:

$p_{\boldsymbol{\theta}}(\mathbf{x},\mathbf{z}|\mathbf{u})=p_{\mathbf{f}}(\mathbf{x}|\mathbf{z})p_{\mathbf{T},\boldsymbol{\lambda}}(\mathbf{z}|\mathbf{u})$ (5)

where we first define:

$p_{\mathbf{f}}(\mathbf{x}|\mathbf{z})=p_{\boldsymbol{\varepsilon}}(\mathbf{x}-\mathbf{f}(\mathbf{z}))$ (6)

which means that the value of $\mathbf{x}$ can be decomposed as $\mathbf{x}=\mathbf{f}(\mathbf{z})+\boldsymbol{\varepsilon}$ where $\boldsymbol{\varepsilon}$ is an independent noise variable with probability density function $p_{\boldsymbol{\varepsilon}}(\boldsymbol{\varepsilon})$, i.e. $\boldsymbol{\varepsilon}$ is independent of $\mathbf{z}$ or $\mathbf{f}$. We assume that the function $\mathbf{f}:\mathbb{R}^{n}\rightarrow\mathbb{R}^{d}$ is injective; but apart from injectivity it can be an arbitrarily complicated nonlinear function. For the sake of analysis we treat the function $\mathbf{f}$ itself as a parameter of the model; however in practice we can use flexible function approximators such as neural networks.

We describe the model above with noisy and continuous-valued observations $\mathbf{x}=\mathbf{f}(\mathbf{z})+\boldsymbol{\varepsilon}$. However, our identifiability results also apply to non-noisy observations $\mathbf{x}=\mathbf{f}(\mathbf{z})$, which are a special case of Eq. (6) where $p_{\boldsymbol{\varepsilon}}(\boldsymbol{\varepsilon})$ is Gaussian with infinitesimal variance. For these reasons, we can use flow-based generative models *(Dinh et al., 2014)* for $p_{\boldsymbol{\theta}}(\mathbf{x}|\mathbf{z})$, while maintaining identifiability.

The prior on the latent variables $p_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{u})$ is assumed to be *conditionally* factorial, where each element of $z_{i}\in\mathbf{z}$ has a univariate exponential family distribution given conditioning variable $\mathbf{u}$. The conditioning on $\mathbf{u}$ is through an arbitrary function $\boldsymbol{\lambda}(\mathbf{u})$ (such as a look-up table or neural network) that outputs the individual exponential family parameters $\lambda_{i,j}$. The probability density function is thus given by:

$p_{\mathbf{T},\boldsymbol{\lambda}}(\mathbf{z}|\mathbf{u})=\prod_{i}\frac{Q_{i}(z_{i})}{Z_{i}(\mathbf{u})}\exp\left[\sum_{j=1}^{k}T_{i,j}(z_{i})\lambda_{i,j}(\mathbf{u})\right]$ (7)

where $Q_{i}$ is the base measure, $Z_{i}(\mathbf{u})$ is the normalizing constant and $\mathbf{T}_{i}=(T_{i,1},\ldots,T_{i,k})$ are the sufficient statistics and $\boldsymbol{\lambda}_{i}(\mathbf{u})=(\lambda_{i,1}(\mathbf{u}),\ldots,\lambda_{i,k}(\mathbf{u}))$ the corresponding parameters, crucially depending on $\mathbf{u}$. Finally,

---

$k$, the dimension of each sufficient statistic, is fixed (not estimated). Note that exponential families have universal approximation capabilities, so this assumption is not very restrictive *(Sriperumbudur et al., 2017)*.

### 3.2 Estimation by VAE

Next we propose a practical estimation method for the model introduced above. Consider we have a dataset $\mathcal{D}=\left\{\left(\mathbf{x}^{(1)},\mathbf{u}^{(1)}\right),\ldots,\left(\mathbf{x}^{(N)},\mathbf{u}^{(N)}\right)\right\}$ of observations generated according to the generative model defined in Eq. (5). We propose to use a VAE as a means of learning the true generating parameters $\boldsymbol{\theta}^{*}:=(\mathbf{f}^{*},\mathbf{T}^{*},\boldsymbol{\lambda}^{*})$, up to the indeterminacies discussed below.

VAEs are a framework that simultaneously learns a deep latent generative model and a variational approximation $q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x},\mathbf{u})$ of its true posterior $p_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{x},\mathbf{u})$, the latter being often intractable. Denote by $p_{\boldsymbol{\theta}}(\mathbf{x}|\mathbf{u})=\int p_{\boldsymbol{\theta}}(\mathbf{x},\mathbf{z},|\mathbf{u})\mathrm{d}\mathbf{z}$ the conditional marginal distribution of the observations, and with $q_{\mathcal{D}}(\mathbf{x},\mathbf{u})$ we denote the empirical data distribution given by dataset $\mathcal{D}$. VAEs learn the vector of parameters $(\boldsymbol{\theta},\boldsymbol{\phi})$ by maximizing $\mathcal{L}(\boldsymbol{\theta},\boldsymbol{\phi})$, a lower bound on the data log-likelihood defined by:

$\mathbb{E}_{q_{\mathcal{D}}}\left[\log p_{\boldsymbol{\theta}}(\mathbf{x}|\mathbf{u})\right]\geq\mathcal{L}(\boldsymbol{\theta},\boldsymbol{\phi}):=$
$\mathbb{E}_{q_{\mathcal{D}}}\left[\mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x},\mathbf{u})}\left[\log p_{\boldsymbol{\theta}}(\mathbf{x},\mathbf{z}|\mathbf{u})-\log q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x},\mathbf{u})\right]\right]$ (8)

We use the reparameterization trick *(Kingma and Welling, 2013)* to sample from $q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x},\mathbf{u})$. This trick provides a low-variance stochastic estimator for gradients of the lower bound with respect to $\boldsymbol{\phi}$. The training algorithm is the same as in a regular VAE. Estimates of the latent variables can be obtained by sampling from the variational posterior.

VAEs, like any maximum likelihood estimation method, requires the densities to be normalized. To this end, in practice we choose the prior $p_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{u})$ to be a Gaussian location-scale family, which is widely used with VAEs.

### 3.3 Identifiability and consistency results

As discussed in section 2.3, identifiability as defined by equation (4) is very hard to achieve in deep latent variable models. As a first step towards an identifiable model, we seek to recover the model parameters or the latent variables up to trivial transformations. Here, we state informally our results on this weaker form of identifiability of the model —a rigorous treatment is given in Section 4. Consider for simplicity the case of no noise and sufficient statistics of size $k=1$, and define $T_{i}:=T_{i,1}$. Then we can recover $\mathbf{z}$ which are related to the original $\mathbf{z}^{*}$ as follows:

$(T_{1}^{*}(z_{1}^{*}),\ldots,T_{n}^{*}(z_{n}^{*}))=A(T_{1}(z_{1}),\ldots,T_{n}(z_{n}))$ (9)

for an invertible matrix $A$. That is, we can recover the original latent variables up to a component-wise (point-wise) transformations $T_{i}^{*},T_{i}$, which are defined as the sufficient statistics of exponential families, and up to a subsequent linear transformation $A$. Importantly, the linear transformation $A$ can often be resolved by excluding families where, roughly speaking, only the location (mean) is changing. Then $A$ is simply a permutation matrix, and equation (9) becomes

$T_{i}^{*}(z_{i}^{*})=T_{i^{\prime}}(z_{i^{\prime}})$ (10)

for a permuted index $i^{\prime}$. Thus, the only real indeterminacy is often the component-wise transformations of the latents, which may be inconsequential in many applications.

### 3.4 Interpretation as nonlinear ICA

Now we show how the model above is closely related to previous work on nonlinear ICA. In nonlinear ICA, we assume observations $\mathbf{x}\in\mathbb{R}^{d}$, which are the result of an unknown (but invertible) transformation $\mathbf{f}$ of latent variables $\mathbf{z}\in\mathbb{R}^{d}$:

$\mathbf{x}=\mathbf{f}(\mathbf{z})$ (11)

where $\mathbf{z}$ are assumed to follow a factorized (but typically unknown) distribution $p(\mathbf{z})=\prod_{i=1}^{d}p_{i}(z_{i})$. This model is essentially a deep generative model. The difference to the definition above is mainly in the lack of noise and the equality of the dimensions: The transformation $\mathbf{f}$ is deterministic and invertible. Thus, any posteriors would be degenerate.

The goal is then to recover (identify) $\mathbf{f}^{-1}$, which gives the independent components as $\mathbf{z}=\mathbf{f}^{-1}(\mathbf{x})$, based on a dataset of observations of $\mathbf{x}$ alone. Thus, the goal of nonlinear ICA was always identifiability, which is in general not attained by deep latent variable models, as was discussed in Section 2 above.

To obtain identifiability, we either have to restrict $\mathbf{f}$ (for instance make it linear) and/or we have to introduce some additional constraints on the distribution of the sources $\mathbf{z}$. Recently, three new nonlinear ICA frameworks *(Hyvärinen and Morioka, 2016, 2017; Hyvärinen et al., 2019)* exploring the latter direction were proposed, in which it is possible to recover identifiable sources, up to some trivial transformations.

---

The framework in *Hyvärinen et al. (2019)* is particularly close to what we proposed above. However, there are several important differences. First, here we define a generative model where posteriors are non-degenerate, which allows us to show an explicit connection to VAE. We are thus also able to perform maximum likelihood estimation, in terms of evidence lower bound, while previous nonlinear ICA used more heuristic self-supervised schemes. Computing a lower bound on the likelihood is useful, for example, for model selection and validation. In addition, we can in fact prove a tight link between maximum likelihood estimation and maximization of independence of latents, as discussed in Supplementary Material F. We also learn both the forward and backward models, which allows for recovering independent latents from data, but also generating new data. The forward model is also likely to help investigate the meaning of the latents. At the same time, we are able to provide stronger identifiability results which apply for more general models than earlier theory, and in particular considers the case where the number of latent variables is smaller than the number of observed variables and is corrupted by noise. Given the popularity of VAEs, our current framework should thus be of interest. Further discussion can be found in Supplementary Material G.

## 4 IDENTIFIABILITY THEORY

Now we give our main technical results. The proofs are in Supplementary Material B.

#### Notations

Let $\mathcal{Z}\subset\mathbb{R}^{n}$ and $\mathcal{X}\subset\mathbb{R}^{d}$ be the domain and the image of $\mathbf{f}$ in (6), respectively, and $\mathcal{U}\subset\mathbb{R}^{m}$ the support of the distribution of $\mathbf{u}$. We denote by $\mathbf{f}^{-1}$ the inverse defined from $\mathcal{X}\to\mathcal{Z}$. We suppose that $\mathcal{Z}$, $\mathcal{X}$ and $\mathcal{U}$ are open sets. We denote by $\mathbf{T}(\mathbf{z}):=(\mathbf{T}_{1}(z_{1}),\ldots,\mathbf{T}_{n}(z_{n}))=(T_{1,1}(z_{1})\ldots,T_{n,k}(z_{n}))\in\mathbb{R}^{nk}$ the vector of sufficient statistics of (7), $\bm{\lambda}(\mathbf{u})=(\bm{\lambda}_{1}(\mathbf{u}),\ldots,\bm{\lambda}_{n}(\mathbf{u}))=(\lambda_{1,1}(\mathbf{u}),\ldots,\lambda_{n,k}(\mathbf{u}))\in\mathbb{R}^{nk}$ the vector of its parameters. Finally $\Theta=\{\bm{\theta}:=(\mathbf{f},\mathbf{T},\bm{\lambda})\}$ is the domain of parameters describing (5).

### 4.1 General results

In practice, we are often interested in models that are identifiable up to a class of transformation. Thus, we introduce the following definition:

###### Definition 1

Let $\sim$ be an equivalence relation on $\Theta$. We say that (1) is identifiable up to $\sim$ (or $\sim$-identifiable) if

$p_{\bm{\theta}}(\mathbf{x})=p_{\tilde{\bm{\theta}}}(\mathbf{x})\implies\tilde{\bm{\theta}}\sim\bm{\theta}$ (12)

The elements of the quotient space $\Theta\mathop{/\sim}\limits$ are called the identifiability classes.

We now define two equivalence relations on the set of parameters $\Theta$.

###### Definition 2

Let $\sim$ be the equivalence relation on $\Theta$ defined as follows:

$(\mathbf{f},\mathbf{T},\bm{\lambda})\sim(\tilde{\mathbf{f}},\tilde{\mathbf{T}},\tilde{\bm{\lambda}})\Leftrightarrow$
$\exists A,\mathbf{c}\mid\mathbf{T}(\mathbf{f}^{-1}(\mathbf{x}))=A\tilde{\mathbf{T}}(\tilde{\mathbf{f}}^{-1}(\mathbf{x}))+\mathbf{c},\forall\mathbf{x}\in\mathcal{X}$ (13)

where $A$ is an $nk\times nk$ matrix and $\mathbf{c}$ is a vector

If $A$ is *invertible*, we denote this relation by $\sim_{A}$. If $A$ is a block *permutation* matrix, we denote it by $\sim_{P}$.

Our main result is the following Theorem:

###### Theorem 1

Assume that we observe data sampled from a generative model defined according to (5)-(7), with parameters $(\mathbf{f},\mathbf{T},\bm{\lambda})$. Assume the following holds:

1. The set $\{\mathbf{x}\in\mathcal{X}|\varphi_{\varepsilon}(\mathbf{x})=0\}$ has measure zero, where $\varphi_{\varepsilon}$ is the characteristic function of the density $p_{\varepsilon}$ defined in (6).
2. The mixing function $\mathbf{f}$ in (6) is injective.
3. The sufficient statistics $T_{i,j}$ in (7) are differentiable almost everywhere, and $(T_{i,j})_{1\leq j\leq k}$ are linearly independent on any subset of $\mathcal{X}$ of measure greater than zero.
4. There exist $nk+1$ distinct points $\mathbf{u}^{0},\ldots,\mathbf{u}^{nk}$ such that the matrix

$L=(\bm{\lambda}(\mathbf{u}_{1})-\bm{\lambda}(\mathbf{u}_{0}),\ldots,\bm{\lambda}(\mathbf{u}_{nk})-\bm{\lambda}(\mathbf{u}_{0}))$ (14)

of size $nk\times nk$ is invertible.

then the parameters $(\mathbf{f},\mathbf{T},\bm{\lambda})$ are $\sim_{A}$-identifiable.

This Theorem guarantees a basic form of identifiability of the generative model (5). In fact, suppose the data was generated according to the set of parameters $(\mathbf{f},\mathbf{T},\bm{\lambda})$. And let $(\tilde{\mathbf{f}},\tilde{\mathbf{T}},\tilde{\bm{\lambda}})$ be the parameters obtained from some learning algorithm (supposed consistent in the limit of infinite data) that perfectly approximates the marginal distribution of the observations. Then the Theorem says that necessarily $(\tilde{\mathbf{f}},\tilde{\mathbf{T}},\tilde{\bm{\lambda}})\sim_{A}(\mathbf{f},\mathbf{T},\bm{\lambda})$. If there were no noise, this would mean that the learned transformation $\tilde{\mathbf{f}}$ transforms the observations into latents $\tilde{\mathbf{z}}=\tilde{\mathbf{f}}^{-1}(\mathbf{x})$ that are equal to the true generative latents $\mathbf{z}=\mathbf{f}^{-1}(\mathbf{x})$, up to a linear invertible transformation (the matrix $A$) and point-wise nonlinearities (in the form of $\mathbf{T}$ and $\tilde{\mathbf{T}}$). With noise, we obtain the posteriors of the latents up to an analogous indeterminacy.

---

4.2 Characterization of the linear indeterminacy

The equivalence relation $\sim_{A}$ provides a useful form of identifiability, but it is very desirable to remove the linear indeterminacy $A$, and reduce the equivalence relation to $\sim_{P}$ by analogy with linear ICA where such matrix is resolved up to a *permutation* and *signed scaling*. We present in this section sufficient conditions for such reduction, and special cases to avoid.

We will start by giving two Theorems that provide sufficient conditions. Theorem 2 deals with the more general case $k\geq 2$, while Theorem 3 deals with the special case $k=1$.

###### Theorem 2 ($k\geq 2$)

Assume the hypotheses of Theorem 1 hold, and that $k\geq 2$. Further assume:

1. The sufficient statistics $T_{i,j}$ in (7) are twice differentiable.
2. The mixing function $\mathbf{f}$ has all second order cross derivatives.

then the parameters $(\mathbf{f},\mathbf{T},\boldsymbol{\lambda})$ are $\sim_{P}$-identifiable.

###### Theorem 3 ($k=1$)

Assume the hypotheses of Theorem 1 hold, and that $k=1$. Further assume:

1. The sufficient statistics $T_{i,1}$ are not monotonic.
2. All partial derivatives of $\mathbf{f}$ are continuous.

then the parameters $(\mathbf{f},\mathbf{T},\boldsymbol{\lambda})$ are $\sim_{P}$-identifiable.

These two Theorems imply that in most cases $\tilde{\mathbf{f}}^{-1}\circ\mathbf{f}:\mathcal{Z}\to\mathcal{Z}$ is a pointwise nonlinearity, which essentially means that the estimated latent variables $\tilde{\mathbf{z}}$ are equal to a permutation and a pointwise nonlinearity of the original latents $\mathbf{z}$.

This kind of identifiability is stronger than any previous results in the literature, and considered sufficient in many applications. On the other hand, there are very special cases where a linear indeterminacy cannot be resolved, as shown by the following:

###### Proposition 1

Assume that $k=1$, and that

1. $T_{i,1}(z_{i})=z_{i}$ for all $i$.
2. $Q_{i}(z_{i})=1$ or $Q_{i}(z_{i})=e^{-z_{i}^{2}}$ for all $i$.

Then $A$ can not be reduced to a permutation matrix.

This Proposition stipulates that if the components are Gaussian (or exponential in the case of non-negative components) and only the location is changing, we can’t hope to reduce the matrix $A$ in $\sim_{A}$ to a permutation. In fact, to prove this in the Gaussian case, we simply consider orthogonal transformations of the latent variables, which all give rise to the same observational distribution with a simple adjustment of parameters.

### 4.3 Consistency of Estimation

The theory above further implies a consistency result on the VAE. If the variational distribution $q_{\boldsymbol{\phi}}$ is a broad parametric family that includes the true posterior, then we have the following result.

###### Theorem 4

Assume the following:

1. The family of distributions $q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x},\mathbf{u})$ contains $p_{\mathbf{f},\mathbf{T},\boldsymbol{\lambda}}(\mathbf{z}|\mathbf{x},\mathbf{u})$.
2. We maximize $\mathcal{L}(\boldsymbol{\theta},\boldsymbol{\phi})$ with respect to both $\boldsymbol{\theta}$ and $\boldsymbol{\phi}$.

then in the limit of infinite data, the VAE learns the true parameters $\boldsymbol{\theta}^{*}:=(\mathbf{f}^{*},\mathbf{T}^{*},\boldsymbol{\lambda}^{*})$ up to the equivalence class defined by $\sim$ in (13).

## 5 EXPERIMENTS

### 5.1 Simulations on artifical data

#### Dataset

We run simulations on data used previously in the nonlinear ICA literature *(Hyvärinen and Morioka, 2016; Hyvärinen et al., 2019)*. We generate synthetic datasets where the sources are non-stationary Gaussian time-series: we divide the sources into $M$ segments of $L$ samples each. The conditioning variable $\mathbf{u}$ is the segment label, and its distribution is uniform on the integer set $\llbracket 1,M\rrbracket$. Within each segment, the conditional prior distribution is chosen from the family (7) for small $k$. When $k=2$, we used mean and variance modulated Gaussian distribution. When $k=1$, we used variance modulated Gaussian or Laplace (to fall within the hypotheses of Theorem 3). The true parameters $\lambda_{i}$ were randomly and independently generated across the segments and the components from a non degenerate distributions to satisfy assumption (iv) of Theorem 1. Following *Hyvärinen et al. (2019)*, we mix the sources using a multi-layer perceptron (MLP) and add small Gaussian noise.

#### Model specification

Our estimates of the latent variables are generated from the variational posterior $q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{u},\mathbf{x})$, for which we chose the following form: $q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x},\mathbf{u})=\mathcal{N}\left(\mathbf{z}|\mathbf{g}(\mathbf{x},\mathbf{u};\phi_{\mathbf{g}}),\mathbf{diag}\,\boldsymbol{\sigma}^{2}(\mathbf{x},\mathbf{u};\phi_{\boldsymbol{\sigma}})\right)$, a multivariate Gaussian with a diagonal covariance. The

---

Ilyes Khemakhem, Diederik P. Kingma, Ricardo Pio Monti, Aapo Hyvärinen

![[Variational_Autoencoders_and_Nonlinear_ICA-_A_Unifying_Framework_p7_img1.jpeg]]
Figure 1: Visualization of both observation and latent spaces in the case  $n = d = 2$  and where the number of segments is  $M = 5$  (segments are colour coded). First, data is generated in (a)-(b) as follows: (a) samples from the true distribution of the sources  $p_{\theta^*}(\mathbf{z}|\mathbf{u})$ : Gaussian with non stationary mean and variance, (b) are observations sampled from  $p_{\theta^*}(\mathbf{x}|\mathbf{z})$ . Second, after learning both a vanilla VAE and an iVAE models, we plot in (c) the latent variables sampled from the posterior  $q_{\phi}(\mathbf{z}|\mathbf{x},\mathbf{u})$  of the iVAE and in (d) the latent variables sampled from the posterior of the vanilla VAE.

noise distribution  $p_{\varepsilon}$  is Gaussian with small variance. The functional parameters of the decoder and the inference model, as well as the conditional prior are chosen to be MLPs. We use an Adam optimizer (Kingma and Ba, 2014) to update the parameters of the network by maximizing  $\mathcal{L}(\pmb{\theta}, \pmb{\phi})$  in equation (8). The data generation process as well as hyperparameter choices are detailed in Supplementary Material H.1.

Performance metric To evaluate the performance of the method, we compute the mean correlation coefficient (MCC) between the original sources and the corresponding latents sampled from the learned posterior. To compute this performance metric, we first calculate all pairs of correlation coefficients between source and latent components. We then solve a linear sum assignment problem to assign each latent component to the source component that best correlates with it, thus reversing any permutations in the latent space. A high MCC means that we successfully identified the true parameters and recovered the true sources, up to point-wise transformations. This is a standard measure used in ICA.

Results: 2D example First, we show a visualization of identifiability of iVAE in a 2D case in Figure 1, where we plot the original sources, observed data and the posterior distributions learned by our model, compared to a vanilla VAE. Our method recovers the original sources up to trivial indeterminacies (rotation and sign flip), whereas the VAE fails to do a good separation of the latent variables.

Results: Comparison to VAE variants We compared the performance of iVAE to a vanilla VAE. We used the same network architecture for both models,

![[Variational_Autoencoders_and_Nonlinear_ICA-_A_Unifying_Framework_p7_img2.jpeg]]
(a) Training dynamics

![[Variational_Autoencoders_and_Nonlinear_ICA-_A_Unifying_Framework_p7_img3.jpeg]]
(b) Changing  $n$
Figure 2: Performance of iVAE in recovering the true sources, compared to VAE,  $\beta$ -VAE and  $\beta$ -TC-VAE, for  $M = 40$ ,  $L = 1000$  and  $d = 5$  (and  $n = 5$  for  $(a)$ ).

with the sole exception of the addition of the conditional prior in iVAE. When the data is centered, the VAE prior is Gaussian or Laplace. We also compared the performance to two models from the disentanglement literature, namely a  $\beta$ -VAE (Higgins et al., 2016) and a  $\beta$ -TC-VAE (Chen et al., 2018). The parameter  $\beta$  of the  $\beta$ -VAE and the parameters  $\alpha$ ,  $\beta$  and  $\gamma$  for  $\beta$ -TC-VAE were chosen by following the instructions of their respective authors. We trained these 4 models on the dataset described above, with  $M = 40$ ,  $L = 1000$ ,  $d = 5$  and  $n \in [2,5]$ . Figure 2a compares performances obtained from an optimal choice of parameters achieved by iVAE and the three models discussed above, when the dimension of the latent space equals the dimension of the data ( $n = d = 5$ ). iVAE achieved an MCC score of above  $95\%$ , whereas the other three models fail at finding a good estimation of the true parameters. We further investigated the impact of the latent dimension on the performance in Figure 2b. iVAE has much higher correlations than the three other models, especially as the dimension increases. Further visualization are in Supplementary Material I.4.

Results: Comparison to TCL Next, we compared our method to previous nonlinear ICA methods, namely TCL by Hyvarinen and Morioka (2016), which is based on a self supervised classification task (see Supplementary Material G.1). We run simulations on the same dataset as Figure 2a, where we varied the number of segments from 10 to 50. Our method slightly outperformed TCL in our experiments. The results are reported in Figure 3a. Note that according to Hyvarinen et al. (2019), TCL performs best among previously proposed methods for this kind of data.

Finally, we wanted to show that our method is robust to some failure modes which occur in the context of self-supervised methods. The theory of TCL is premised on the notion that in order to accurately classify observations into their relative segments, the model must learn the true log-densities of sources within each segment. While such theory will hold in the limit of infinite data, we considered here a special case where accurate classi

---

Ilyes Khemakhem, Diederik P. Kingma, Ricardo Pio Monti, Aapo Hyvärinen

![[Variational_Autoencoders_and_Nonlinear_ICA-_A_Unifying_Framework_p8_img4.jpeg]]
(a) Normal

![[Variational_Autoencoders_and_Nonlinear_ICA-_A_Unifying_Framework_p8_img5.jpeg]]
(b) Easy to classify
Figure 3: (a) Performance of iVAE in comparison to TCL in recovering the true sources on normal data (b) Performance of iVAE in comparison to TCL in recovering the true sources on easy to classify data.

fication did not require learning the log-densities very precisely. This was achieved by generating synthetic data where  $x_{2}$  alone contained sufficient information to perform classification, by making the mean of  $x_{2}$  significantly modulated across segments; further details in Supplementary Material H.2. In such a setting, TCL is able to obtain high classification accuracy without unmixing observations, resulting in its failure to recover latent variables as reflected in Figure 3b. In contrast, the proposed iVAE, by virtue of optimizing a maximum likelihood objective, does not suffer from such degenerate behaviour.

Further simulations on hyperparameter selection and discrete data are in Supplementary Material I.

# 5.2 Nonlinear causal discovery in fMRI

An important application of ICA methods is within the domain of causal discovery (Peters et al., 2017). The use of ICA methods in this domain is premised on the equivalence between a (nonlinear) ICA model and the corresponding structural equation model (SEM). Such a connection was initially exploited in the linear case (Shimizu et al., 2006) and extended to the nonlinear case by Monti et al. (2019) who employed TCL.

Briefly, consider data  $\mathbf{x} = (x_{1}, x_{2})$ . The goal is to establish if the causal direction is  $x_{1} \rightarrow x_{2}$ , or  $x_{2} \rightarrow x_{1}$  or conclude that no (acyclic) causal relationship exists. Assuming  $x_{1} \rightarrow x_{2}$ , then the problem can be described by the following SEM:  $x_{1} = f_{1}(n_{1})$ ,  $x_{2} = f_{2}(x_{1}, n_{2})$  where  $\mathbf{f} = (f_{1}, f_{2})$  is a (possibly nonlinear) mapping and  $\mathbf{n} = (n_{1}, n_{2})$  are latent disturbances that are assumed to be independent. The above SEM can be seen as a nonlinear ICA model where latent disturbances,  $\mathbf{n}$ , are the sources. As such, we may perform causal discovery by first recovering latent disturbances (using TCL or iVAE) and then running a series of independence tests. Formally, if  $x_{1} \rightarrow x_{2}$  then, denoting statistical independence by  $\perp$ , it suffices to verify that  $x_{1} \perp n_{2}$  whereas  $x_{1} \not\perp n_{1}$ ,  $x_{2} \not\perp n_{1}$  and  $x_{2} \not\perp n_{2}$ . Such an

![[Variational_Autoencoders_and_Nonlinear_ICA-_A_Unifying_Framework_p8_img6.jpeg]]
(a) iVAE
Figure 4: Estimated causal graph on hippocampal fMRI data unmixing of sources is achieved via iVAE (left) or TCL (right). Blue edges are feasible given anatomical connectivity, red edges are not.

![[Variational_Autoencoders_and_Nonlinear_ICA-_A_Unifying_Framework_p8_img7.jpeg]]
(b) TCL

approach can be extended beyond two-dimensional observations as described in Monti et al. (2019).

To demonstrate the benefits of iVAE as compared to TCL, both algorithms were employed to learn causal structure from fMRI data (details in Supplementary Material I.3). The recovered causal graphs are shown in Figure 4. Blue edges are anatomically feasible whilst red edges are not. There is significant overlap between the estimated causal networks, but in the case of iVAE both anatomically incorrect edges correspond to indirect causal effects. This is in contrast with TCL where incorrect edges are incompatible with anatomical structure and cannot be explained as indirect effects.

# 6 CONCLUSION

Unsupervised learning can have many different goals, such as: (i) approximate the data distribution, (ii) generate new samples, (iii) learn useful features, and above all (iv) learn the original latent code that generated the data (identifiability). Deep latent-variable models typically implemented by VAEs are an excellent framework to achieve (i), and are thus our first building block. The nonlinear ICA model discussed in section 3.4 is the only existing framework to provably achieve (iv). We bring these two pieces together to create our new model termed iVAE. In particular, this is the first rigorous proof of identifiability in the context of VAEs. Our model in fact checks all the four boxes above that are desired in unsupervised learning.

The advantage of the new framework over typical deep latent-variable models used with VAEs is that we actually recover the original latents, thus providing principled disentanglement. On the other hand, the advantages of this algorithm for solving nonlinear ICA over Hyvarinen et al. (2019) are several; briefly, we significantly strengthen the identifiability results, we obtain the likelihood and can use MLE, we learn a forward model as well and can generate new data, and we consider the more general cases of noisy data with fewer components.

---

Corrigendum

The published version of this paper claimed that the identifiability result extends to the case with discrete observations. Unfortunately, after publication, we found an error in the proof for the discrete case, so we can no longer claim that this is the case. We suspect that the discrete case requires a different type of proof, which we leave for future work. To remove incorrect claims, we made minimal changes to the abstract, Section 3.1 and Section 6; we also rewrote Appendix C which contained the erroneous proof. We do still provide experiments in Supplementary Material I that strongly suggest that identifiability is achievable in such setting.