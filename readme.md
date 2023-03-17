# Rethinking the editing of generative adversary networks: a method to estimate editing vectors based on dimension reduction
## 内容:
本文件夹中包含了除 StyleGAN-Human 外全部的实验材料，StyleGAN-Human 的使用方法请参考原仓库（https://github.com/stylegan-human/StyleGAN-Human.git）。

generate_with_latent 目录中存储了 StyleGAN-Human 生成的 1000 组隐向量及其对应图片（由于右键附件容量有限，未附在压缩包内）。

res.txt 中逐行存储了对应图片的上装长度和纹理有无。

本文提出的 Binary Feature Editing 方法实现见于 binary_feature.py。

本文提出的 Continuous Feature Editing 方法实现见于 continuous_feature.py。

## 说明
如欲复现本文的编辑效果，需要将我们得到的编辑向量导入 StyleGAN-Human 的 latent_direction/sefa 目录中，并在 edit_config.py 中加入相应信息。

## 贡献
完成人： 蒋浩然、俞政宏

蒋浩然：实验设计；Continuous Feature Editing 方法的推导和实现；论文撰写（Methods 及以后）。

俞政宏：负责 EditGAN 结果的复现；Binary Feature Editing 方法的实现；论文撰写（Existing works 及以前）。
