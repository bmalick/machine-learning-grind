# VAE

[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

## Generated digits

- MLP version which is closer to the one presented in the paper.
![](./logs/mlp_auto_enc--20260630-214405/visualizations/024.jpg)

- Convolution layers verion
![](./logs/conv_auto_enc--20260630-214456/visualizations/024.jpg)

## Losses

![](./mlp_auto_enc--20260630-214405-vs-conv_auto_enc--20260630-214456/perstep-elbo.jpg)
![](./mlp_auto_enc--20260630-214405-vs-conv_auto_enc--20260630-214456/perstep-log_p.jpg)
![](./mlp_auto_enc--20260630-214405-vs-conv_auto_enc--20260630-214456/perstep-kl_div.jpg)
