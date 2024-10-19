# GAN

# DDPM

# StrDiffusion


## Idea
1. HOSVD分解后的张量重建似乎无法补全空白，需要设计网络增加平滑性(e.g.,TV损失)
2. 如何归一化HOSVD分解后的\delta core使其从正负无穷归一化到[0, 1]
3. X * M的tucker形式？其中M的最后一个mode是重复的
4. 和tensor sketching技术的关系？
5. Tucker分解core tensor一定是稀疏的，可能是对角的？可能元素是高斯分布的？约束生成结果的低秩性？
6. mask在需要的时候需要二值化? 
7. 搭建一个tensor function和SDE的判别器，让SDE进行反复采样？like StrDiffusion
8. vanilla SDE从LQ->GT后 再使用一次Tensor function进行后处理，平滑地填平未补全的部分？
9. ~~每次变换mu会使得相同图像的两次随机状态不一样，因此无法收敛？但是增加数据增广就可以？~~**数据量太大需要加长训练时间**
10. 在相邻状态间添加Tucker core tensor的最优解和预测值的mse损失后无法收敛，并且lowrank损失下降到0.0021左右不再变换
11. 计算一个低秩的最优解代替yt_1_optimum？设计一个time-dependent的低秩函数，随着采样时间从0到∞，最优解倾向于从F范数转向到核范数
12. 将LQ和HQ进行正交分解，然后学习核张量G的扩散过程？或者控制G不变，轮流更新factor matrices？由分解带来的误差又怎么解决？
13. 可以将在张量元素中的扩散终点视为一个随机矩阵？研究这个随机矩阵满秩的概率为多大？以证明图像退化是一个秩增过程？
14. 张量低秩分解-重建过程导致图像失真，这是导致低秩约束失效原因？

## Log
1. 训练隐式tuckerNet级联SDE太消耗时间
2. 200epoch vanilla SDE with T=100: 
        PSNR:24.66, SSIM: 0.8886
    使用原始mask: 
        PSNR: 23.737318436620267, SSIM: 0.8717837802868211
    使用截取的256*256，进行测试：
        PSNR: 26.17640709728353, SSIM: 0.8988711440913995
    使用截取的退化图像256*256& 
        PSNR: 26.644616516209325, SSIM: 0.9035967934292346
    训练使用随机截取的掩码，测试使用截取的退化图像：
        PSNR: 28.594873487207213, SSIM: 0.9181528240175935
        PSNR: 27.99784054964565, SSIM: 0.915652642487878
        PSNR: 27.81815542971622, SSIM: 0.9161213248744948
        PSNR: 28.137314051914736, SSIM: 0.9179463687704654
        PSNR: 27.4387945115071, SSIM: 0.902100797548644
    继续训练300epoch ir=5e-5
        PSNR: 30.342365379032483, SSIM: 0.9408874248049109
    全图验证:
        PSNR: 29.69959093120466, SSIM: 0.938167336712829

3. Tensor Function: PSNR: 24.36, SSIM: 0.8741
4. 加mask训练：
    PSNR: 27.978176811823545, SSIM: 0.9177896625697921 损失下降到0.0018左右
    继续训练300epoch 测试性能提升显著：
        PSNR: 31.3691298897155, SSIM: 0.9459931772643316
        PSNR: 31.313967784838393, SSIM: 0.9520706849204292
5. 加Low ramk loss，全图验证
    PSNR: 26.54908389190119, SSIM: 0.9107776230896276
    PSNR: 28.346947395680903, SSIM: 0.9271333907347883

## Dataset Bug
mask size is wrong: (765, 768) /home/chengzy/mural-completion/dataset/muralV3/Masks/079.jpg
GT size is wrong: (765, 768, 3) 079.jpg
LQ size is wrong: (765, 768, 3) 079.jpg
mask size is wrong: (1557, 1681) /home/chengzy/mural-completion/dataset/muralV3/Masks/012.jpg
GT size is wrong: (1557, 1681, 3) 012.jpg
LQ size is wrong: (1557, 1681, 3) 012.jpg
mask size is wrong: (767, 767) /home/chengzy/mural-completion/dataset/muralV3/Masks/268.jpg
GT size is wrong: (767, 767, 3) 268.jpg
LQ size is wrong: (767, 767, 3) 268.jpg
