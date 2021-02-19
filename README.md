# Uncertainty-Aware Boosted Ensembling in Multi-Modal Settings

> Uncertainty-Aware Boosted Ensembling in Multi-Modal Settings

> Utkarsh Sarawgi\*, Rishab Khincha\*, Wazeer Zulfikar\*, Satrajit Ghosh and Pattie Maes  
> Under review 

\* Equal contribution

## Abstract

Reliability of machine learning (ML) systems is crucial in safety-critical applications such as healthcare, and uncertainty estimation is a widely researched method to highlight the confidence of ML systems in deployment. Sequential and parallel ensemble techniques have shown improved performance of ML systems in multi-modal settings by leveraging the feature sets together. We propose an uncertainty-aware boosting technique for multi-modal ensembling in order to focus on the data points with higher associated uncertainty estimates, rather than the ones with higher loss values. We evaluate this method on healthcare tasks related to Dementia and Parkinson's disease which involve real-world multi-modal speech and text data, wherein our method shows an improved performance. Additional analysis suggests that introducing uncertainty-awareness into the boosted ensembles decreases the overall entropy of the system, making it more robust to heteroscedasticity in the data, as well as better calibrating each of the modalities along with high quality prediction intervals.


## Usage 

### Dataset Download

1. Alzheimer's Dementia: Request access from [DementiaBank](https://dementia.talkbank.org/)
2. Parkinson's Disease: Available on UCI datasets [Parkinson's Telemonitoring Dataset](https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring)

### Setup

1. Install dependencies using `pip install -r requirements.txt`
2. Install and setup OpenSmile for Compare features extraction following [COMPARE.md](https://github.com/wazeerzulfikar/ad-mmse/blob/master/COMPARE.md)
3. Extract compare features

### Run

1. Alzheimer's Dementia - Set config parameters in `config.py` and run `python main.py`
* Vanilla Ensemble - `boosting_type`: `rmse` and `voting_type`: `hard_voting`
* UA Ensemble - `boosting_type`: `stddev` and `voting_type`: `hard_voting`
* UA Ensemble (weighted) - `boosting_type`: `stddev` and `voting_type`: `uncertainty_voting`

2. Parkinson's Telemonitoring Dataset - Set config parameters in `main.py` and run `python main.py`
* Vanilla Ensemble - `ua_ensemble`: `False`
* UA Ensemble - `ua_ensemble`: `True`
* UA Ensemble (weighted) - `ua_ensemble`: `True`

## License

This code is released under the MIT License (refer to the [LICENSE](https://github.com/usarawgi911/Uncertainty-aware-boosting/blob/master/LICENSE) for details).

## Citation

If you find this project useful for your research, please use the following BibTeX entries.
Uncertainty-Aware Boosted Ensembling in Multi-Modal Settings

    @article{sarawgi2021uaensemble,
      title={Uncertainty-Aware Boosted Ensembling in Multi-Modal Settings},
      author={Sarawgi\*, Utkarsh and Khincha\*, Rishab and Zulfikar\*, Wazeer and Ghosh, Satrajit and Maes, Pattie},
      journal={},
      year={2021}
    }
