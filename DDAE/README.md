# DDAE

## Requirements:
- simple use: `pip install -r requirements.txt`
- or install these:
    - torch==1.9.0
    - torchaudio==0.9.0
    - torchvision==0.10.0
    - scipy==1.7.0
    - h5py==3.3.0
    - librosa==0.8.1

## Run:
STFT preprocessing:
`python src/preprocessing.py data.h5 list_noisy list_clean`
Train model with pytorch (will store model at src/model/DDAE.pt)
`python src/train_DNN.py v2_train_DNN.py data.h5`
Use the model to test result (result will store at src/enhanced_voice/)
`python src/run.py DDAE.pt list_noisy`


## References:
@inproceedings{lu2013speech,
  title={Speech enhancement based on deep denoising autoencoder.},
  author={Lu, Xugang and Tsao, Yu and Matsuda, Shigeki and Hori, Chiori},
  booktitle={Interspeech},
  volume={2013},
  pages={436--440},
  year={2013}
}
