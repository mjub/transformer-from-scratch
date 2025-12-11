# Scaling laws

 - $D \approx 20 \times N$

   The number of tokens $D$ in the training set must be 20 times larger than the number $N$ of trainable parameters. This is the "Chinchilla ratio"

 - $d_k = 64$

   The dimension of a single attention head. It is usually set to 64.

 - $H = \frac{d_{\text{model}}}{d_k}$

    The number of heads (*focus points*) $H$ should be equal to the model width $d_{\text{model}}$ divided by the dimension of a single head.

 - $\mathcal{L}_0 \approx -\log{\frac{1}{V}}$

   The expected cross-entropy when starting training should be approximatively equal to $-\log{\frac{1}{V}}$, that is $\log{V}$, where $V$ is the size of the vocabulary.

 - $d_{ff} = 4 \times d_{\text{model}}$

   The feed forward size should be 4 times superior to the model width.

 - $d_{\text{model}} \approx 20 \times L$

   The model width $d_{\text{model}}$ should be approximatively equal to 20 times the number of layers (the *model depth*) $L$.

   Some use a factor of 10 instead of 20.

 - $P \approx 12 \times L \times d_{\text{model}}^2$

   The number of trainable parameters $P$ is roughly equal to $12 \times L \times d_{\text{model}}^2$.
