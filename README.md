# muralCompletion

## Introduction
Mural Image Completion with Tesnor Decomposition and Diffusion

## Overview

## Installation

1. Download source code:
    
    `git clone https://gitee.com/czy-codelib/mural-completion.git`

2.  Pip install dependencies:
    * OS: Ubuntu 20.04.6
    * nvidia :
        - cuda: 12.1
        - cudnn: 8.5.0
    * python == 3.9.18
    * pytorch >= 2.1.0
    * Python packages: `pip install -r requirements.txt`

3.  Dataset Preparation:

    * You can set the mask/LQ/GT path in [tdm/options/test/ir-sde-td.yml](https://gitee.com/czy-codelib/mural-completion/blob/master/tdm/options/test/ir-sde-td.yml#L26)

4. Download the weight of network from the [link](https://drive.google.com/file/d/1lD1IAkwXbQP9ifum_3loldC-EBtYNQ2Q/view?usp=drive_link) and move it into the path which setted in [tdm/options/test/ir-sde-td.yml](https://gitee.com/czy-codelib/mural-completion/blob/master/tdm/options/test/ir-sde-td.yml#L52)

5. Run the following command to test performance:

    `python tdm/test.py`
    
## Acknowledgement
1. XXXX