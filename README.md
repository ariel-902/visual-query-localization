# visual-query-localization
visual query localization, a project aimed at pinpointing the final appearance of specific objects within egocentric video footage.

## Installation

`conda create --name vqloc python=3.8`

`conda activate vqloc`

`# Install pytorch or use your own torch version`
`conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge`

在 DLCV-Fall-2023-Final-2-jokoandherfriends 資料夾底下跑

`pip install -r requirements.txt`

`pip install loralib`

## Download **Pre-trained Weights**

We used the model weights trained on [here](https://utexas.box.com/shared/static/3j3q9qsc1kovpwfxtnsful7pvdy234q6.tar).

請下載到 DLCV-Fall-2023-Final-2-jokoandherfriends 資料夾

## Training 

在 DLCV-Fall-2023-Final-2-jokoandherfriends/config/train.yaml 中 clip_dir 修改為助教提供的 DLCV_vq2d_data/clips 路徑，meta_dir 修改為助教提供的 DLCV_vq2d_data 路徑

bash train.sh
