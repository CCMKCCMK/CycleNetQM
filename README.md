# CycleNetQM: Enhanced Time Series Forecasting

An improved implementation of [CycleNet](https://github.com/ACAT-SCUT/CycleNet) developed as part of the COMP4434 course project.

## Contributors

- [**Wenchang CHAI** (CCMKCCMK)](https://github.com/CCMKCCMK)
- [**Zitong SHEN** (Qween0fPandora)](https://github.com/Qween0fPandora)
- [**Beichen GUO** (TechnicolorGUO)](https://github.com/TechnicolorGUO)
- [**Honghe DING** (DavidDING21)](https://github.com/DavidDING21)

## Key Improvements

Our enhanced version introduces two major improvements to the original CycleNet architecture:

1. **Seasonal Scalar** added to the Cycle component Q
2. **Step-wise Training Strategy** replacing the original joint training approach

Detailed implementation and analysis can be found in our group report.

## Performance

Our model **CycleNetQM** achieves state-of-the-art results on the [electricity dataset](https://drive.google.com/file/d/1bNbw1y8VYp-8pkRTqbjoW-TA-G8T0EQf/view), outperforming several baseline models including:

- Linear
- LSTM
- GRU
- CycleNet+Linear
- CycleNet+MLP

## Getting Started

### Prerequisites

The conda environment requirements are identical to the original [CycleNet repository](https://github.com/ACAT-SCUT/CycleNet).

### Pre-trained Models

Pre-trained model checkpoints are available on [Google Drive](https://drive.google.com/file/d/1hQkVfMomVv1VSVVTVM2ZX1Cqo3Ve0Qe1/view?usp=sharing).

### Train & Run
To train or test the model:

```sh run_main.sh```

Modify run_main.sh to:

Change models you want to train using model_names
Toggle training/testing with is_training (1 for training & testing, 0 for testing)
Change the backbone of CycleNet using model_types

### Model variants available:

 - CycleNetQQ: Seasonal scalar only
 - CycleNetMM: Step-wise training only
 - CycleNetQM: Combined approach (our main model)
 - CycleNet
 - Linear
 - LSTM
 - GRU

## Acknowledgments

This work builds upon the [CycleNet](https://github.com/ACAT-SCUT/CycleNet) architecture developed by ACAT-SCUT. We thank the original authors for their foundational work in time series forecasting.
