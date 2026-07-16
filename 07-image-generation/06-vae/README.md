# VAE

[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

## Generated digits

- MLP version which is closer to the one presented in the paper.
![](./logs/mlp-vae--20260712-165845/visualizations/024.jpg)

- Convolution layers verion
![](./logs/conv-vae--20260712-165941/visualizations/024.jpg)

## Losses

| | |
|:---:|:---:|
| ![](./mlp-vae-vs-conv-vae/perstep-elbo.jpg) | ![](./mlp-vae-vs-conv-vae/perstep-log_p.jpg) |
| ![](./mlp-vae-vs-conv-vae/perstep-kl_div.jpg) | |

## Resources
- The original paper
- [Tutorial on Diffusion Models for Imaging and Vision](https://arxiv.org/abs/2403.18103)
- [Understanding Deep Learning](https://udlbook.github.io/udlbook/)

