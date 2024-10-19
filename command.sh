#train structure denoising
python train/structure/config/inpainting/train.py
# train texture denoising
python train/texture/config/inpainting/train.py
# train discriminator
python train/discriminator/config/inpainting/train.py

#train tdm
python tdm/train.py
python tdm/test.py