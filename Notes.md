2024-11-29 00:07

Tags:

# Neural Nets

Neural networks are computational models inspired by biological neural systems, composed of layers of interconnected neurons with learnable parameters.  

These networks learn from data by *minimizing* a typically non-convex objective function, leveraging gradient descent or its variants. 

Efficient computation of gradients, facilitated by the chain rule and implemented via *backpropagation*, is enabled by representing the network as a computational graph. This representation integrates naturally with automatic differentiation tools, making it an effective framework for training neural networks.

This is why certain architectures succeed, they are different ways of structuring computational graphs that maintain gradient flow and composability while being highly differentiable.


>[!summary]  Another definition 
>Neural nets are computational graphs / analog circuits that play well with automatic differentiation (Backpropagation in this case).  

## What do we want from these circuits ?


>[!danger] Summary. 
>We want our neural net to:
>1.  Have the ability to approximate any function. 
>2. Approximate a function from data.
>
>This connects to two fundamental theorems: 
> 1. Universal Approximation Theorem (capability)
> 2. Statistical Learning Theory (learnability), read more here: [[Learning Theory]]

### Expressivity: 

We want it to be rich enough to express patterns of interest in data, we don't usually know the patterns of interest, so we want our circuits to express all possible patterns.

We need the neural nets to be *universal approximators*, some of the known universal approximators are Fourier series, polynomials, Piecewise-constant and Piecewise-linear. 

The last category explains this property in neural nets crisply. 

> The fact the neural nets are universal approximators is rooted in the universal approximation theorem, it states that with one hidden layer and sufficient width, one can approximate any function to an arbitrary precision. The theorem guarantees the existence of a set of weights that can approximate the function, but doesn't specify how to find those weights. In practice, finding those exact weights through training is extremely challenging (Optimization in non-convex landscapes, Sample complexity, Training stability, Generalization bound, Scale and computation efficiency)

Piecewise-linear functions are constructed from a sum of *elbow* functions and a constant (To move it around).  Using elbows we can represent piecewise linear functions in a way that is differentiable for a computer. 

![Blue curve represents \(g(x)\), black dots represent noisy data samples.](Pasted%20image%2020241114123400.png)

The elbows are the red lines. 

Now we need to give our computer a *degree of freedom* to be able to draw elbows, with different slopes and at different positions.

![[Pasted image 20241129161624.png]]

To draw these elbows in a way differentiable by the computer and that we can move by changing $w$ and/or $b$ through training, we use ReLUs. 

$$ReLU(x) = \max(0, wx + b)$$

![[Pasted image 20241129162735.png]]


This is another way to write it out.
$$ReLU(wx + b) = \begin{cases}
wx + b & \text{if } wx + b > 0 \\
0 & \text{if } wx + b \leq 0
\end{cases}$$
$$ReLU(wx + b) = \begin{cases}
wx + b & \text{if } w > 0 \text{ and } x > -\frac{b}{w} \\
0 & \text{if } w > 0 \text{ and } x \leq -\frac{b}{w} \\
wx + b & \text{if } w < 0 \text{ and } x < -\frac{b}{w} \\
0 & \text{if } w < 0 \text{ and } x \geq -\frac{b}{w}
\end{cases}$$

Another elbow we might be interested in is that with negative values, to achieve that we multiply the output of each $ReLU$ by a weight in the output layer. We sum the output of each neuron and add a constant to get the final approximation 


![[Pasted image 20241129161427.png]]


>The optimizer is trying to make the final real number (loss) as small as possible across all the training points. To push the loss down, how much do I have to move y, b’s, and w’s? This process goes backward and the elbow changes

The more neurons you have, the more your approximation gets finer since we get more elbows (*breakpoints*) to work with. Increasing the depth of the neural net improves our approximation through a different mechanism: the compositionality of ReLU functions allows each higher-layer elbow to be constructed from combinations of elbows in previous layers, creating more complex nonlinear patterns.

Something one might notice is that there's some *'Redundancy'* in our representation, when we multiply weight matrices across layers, they effectively collapse into equivalent simpler representations. This means the actual degrees of freedom (effective parameters) can be fewer than the raw parameter count. In the example above we have 2 degrees of freedom for slopes and locations.  

Now why make NNs deeper when making them wider is sufficient ? This is rooted in the second goal, *Learnability*.  

### Learnability:

In the early stages of Neural nets, most of the focus was on the expressivity of the net, now much of the focus is on the goal of learnability. 

This is why some non-linearities are preferred in comparison to others, if the goal was only expressivity, we can use any non-linearity. But since we also care about the ability of our neural net to learn we prefer some non-linearities to others (Think about saturation and gradient flow as an example).

Now, the question of what does our neural network learn, is very complicated and we do not have a rigorous, well-understood theoretical framework to explain it. 

But if we can perfectly understand linear models, think linear regression, we can think about and come to understand non-linear deep models, and more importantly understand:
	*what does our optimization algorithm really learn ?* *When we look at the data what patterns come into focus ? and why these exact patterns ?*

We achieve this through a local linearization of our neural networks. 

#### Neural Net Represented as a Generalized Linear Model: 

There are two perspectives to see how a neural net can be represented as $GLM$ and they both revolve around a key question, What are *features* ? 
##### The usage perspective, features are the penultimate layer outputs: 

This perspective is useful when discussing **pretraining** or **representation learning**.

The following is a simplified sketch of a neural network with $l$ hidden layers.

![[Pasted image 20241214184959.png]]


We can think of the output of the penultimate layer $\tilde{X}$ as lifting or distillation of the input $X$ into a nicer feature space. $\tilde{X} = \phi(X)$. 

This featurization is data-driven rather than hand-picked. 

The output of the featurizer will be then fed as a linear function of those learned features to a loss that we are optimizing. 

If we use the Hinge Loss, we are basically constructing a kernel (Derived from data) $SVM$ or kernel regression with squared loss and so on.

the layers essentially find a representation of the data that allows the generalized linear model ($GLM$) to work well, this $GLM$, will be given by: 

$$f(X_k*,w) = \sum_{i=1}^\lambda w_i \tilde{X_{ki}}$$

> This is why **pretraining** (e.g., in models like BERT or GPT or CLIP) is so powerful—it learns a feature space that is highly effective for downstream tasks.



##### The training perspective, local features for learning:

This perspective is useful when discussing **fine-tuning** or **optimization dynamics**.

Our general model is deep and non-linear in $X$ and $\theta$, for example:  

$$f(x; \theta) = W^{(2)}ReLU(W^{(1)}x+b^{(1)})+b^{(2)}
$$

Here, $\theta = [vec(W^{(1)}),vec(b^{(1)}),vec(W^{(2)}),vec(b^{(2)})]$, represents all the learnable parameters.

During training, the optimization algorithm and the model’s *output* can be approximated locally as a linear function of the parameter changes $\Delta \vec{\theta}$:

$$f(x; \theta) = f(x, \theta_0) + \left. \frac{d{f}}{d\theta} \right|_{\theta_0} \cdot \Delta \vec{\theta}
$$

1. Now, $\Delta \vec{\theta}$ are the locally learnable parameters, consider them as our weights. These are the best parameters at the *CURRENT* iteration, imagine it as if you have to optimize a function just once, we will do this at each iteration, have a different function to optimize, in the context of $GD$ optimization,  $\Delta \vec{\theta} =  -\eta \nabla_\theta{L(y, f(x;\theta))}$ now plug $f$ as the $GLM$.  
2. the $d$ feature vector (or feature matrix, in batch training) , $\frac{df}{d\theta} =[\frac{df}{d\theta_0}, \frac{df}{d\theta_1}, ..., \frac{df}{d\theta_d}]$ is the learned non-linear features.
3. $f(x, \theta_0)$, is a constant, consider it as our bias. 

###### Important Technical Clarification on Notation:

For a neural network with output dimension $m$, $f: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian $\frac{df}{d\theta}$ is actually a tensor of shape $(m, p)$ where $p$ is the number of parameters. More precisely:
$$\frac{df}{d\theta} = \left[\frac{\partial f_i}{\partial \theta_j}\right]_{i=1,\ldots,m;\;j=1,\ldots,p}$$

We have successfully simplified the general model into a generalized *linear* model. 

> Remember, this generalized model is *local*, the feature matrix and $\theta_0$ change at every step of the gradient, we have a different GLM to optimize at each step.

From this perspective, one can see that despite the many possible layers within a neural net, the algorithm is working (*at each step of the gradient*) on a generalized linear model (though differently centered) with a feature (encoded in $\frac{f(X, \theta_0)}{d\theta}$) corresponding to every parameter. 

This means that *some* findings that describe the inductive bias of gradient descent and optimization dynamics in a linear model, can be transcribed to a neural net through this view.  

> Try to derive the gradient descent update to find the optimal value of $\Delta \vec{\theta}$ (best weight change for the current iteration) considering the feature matrix as you design matrix, it will look exactly the same as that of linear regression if you use squared loss. 

> [!danger] Limitations of the Linear Approximation:
While many insights from linear models transfer, several important phenomena in neural networks cannot be explained by the local linear approximation:
>
>1. *Long-term dynamics*: The changing feature matrix means long-term behavior can deviate significantly from linear predictions
>2. *Phase transitions*: Sudden changes in network behavior during training
>3. *Representation learning*: The evolution of features over time
>4. *Architecture-specific effects*: Different architectures can lead to qualitatively different training dynamics


*For example, with squared-error loss to optimize*: 

In linear regression we have a design matrix $X$, here we have a feature matrix $\frac{f(X, \theta_0)}{d\theta}$. 

The same way the singular values of $X$ determine the gradient descent direction in linear regression, those of the feature matrix $\frac{f(X, \theta_0)}{d\theta}$, determines our next trusted direction in neural nets. hence early stopping. [[Implicit regularization and linear regression]] 

The same way the conditioning of the hessian is determined by $X$ in linear regression, in neural nets, it's directly influenced by the feature matrix $\frac{f(X, \theta_0)}{d\theta}$. hence  the importance of batch normalization. [[Normalization and linear regression]] 

It's worth noting that batch normalization's role is more complex than just Hessian conditioning 
(Internal covariate shift reduction, Optimization landscape smoothing, Implicit regularization)


>[!important] Implications for Modern Architecture Design:
The local linear approximation perspective has influenced several modern architecture choices:
>
>1. *Skip connections:* Help maintain stable feature matrices during training
>2. *Attention mechanisms:* Create more flexible feature interactions
>3. *Normalization schemes:* Better conditioning of the local optimization problem
# References
[[@CS182]] Lecture 1, 2, 3, 4, 5. 

For more: https://claude.ai/chat/0e4f589f-f573-43d2-87fd-36c1f2ae414b (Talks about other resources and *NTK*)
