# Low-rank Structure Guided Diffusion for Mural Restoration

## Installation

1. Download source code and dataset:
    
    * `git clone https://github.com/CZY-Code/LRDiff.git`
    * Download the dataset of from the [link](https://drive.google.com/file/d/1Twzrkkb9jEInpsrdrabB6RAcHagwZCVP/view?usp=drive_link)
   

3.  Pip install dependencies:
    * OS: Ubuntu 20.04.6
    * nvidia :
        - cuda: 12.1
        - cudnn: 8.5.0
    * python == 3.9.18
    * pytorch >= 2.1.0
    * Python packages: `pip install -r requirements.txt`

4.  Dataset Preparation:

    * You can set the mask/LQ/GT path in [tdm/options/test/ir-sde-td.yml](https://github.com/CZY-Code/LRDiff/blob/a950d090ff1ce918910198205630b24207eb28eb/tdm/options/test/ir-sde-td.yml#L25)

5. Download the weight of network from the [link](https://drive.google.com/file/d/1lD1IAkwXbQP9ifum_3loldC-EBtYNQ2Q/view?usp=drive_link) and move it into the path which setted in [tdm/options/test/ir-sde-td.yml](https://github.com/CZY-Code/LRDiff/blob/530432b32b39e26db4c9c8f18ccf845f0ffd57eb/tdm/options/test/ir-sde-td.yml#L52)

6. Run the following command to test performance:

    `python tdm/test.py`
    
## Acknowledgement
This implementation is based on / inspired by:

* [https://github.com/Algolzw/image-restoration-sde](https://github.com/Algolzw/image-restoration-sde) (Image Restoration SDE)
* [https://github.com/andreas128/RePaint](https://github.com/andreas128/RePaint) (RePaint)
* [https://github.com/htyjers/StrDiffusion](https://github.com/htyjers/StrDiffusion) (StrDiffusion)
