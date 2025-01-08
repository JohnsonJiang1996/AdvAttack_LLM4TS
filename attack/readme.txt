# Time Series Forecasting Adversarial Attacks

This project implements and evaluates various adversarial attacks against time series forecasting models, with a particular focus on TimeGPT and other neural forecasting architectures.

## Project Structure

- `attack.py`: Contains implementations of different adversarial attack methods:
  - GWN (Gaussian White Noise)
  - DGA (Directional Gradient Attack)
  - SPSA (Simultaneous Perturbation Stochastic Approximation)
  - ITE (Iterative Attack)

- `main.py`: Demonstrates the implementation of adversarial attacks against TimeGPT using the exchange rate dataset.

- `neural_test.py`: Benchmarks various neural forecasting models against TimeGPT, including:
  - LSTM
  - NHITS
  - Autoformer
  - iTransformer
  - PatchTST
  - TimesNet
  - NLinear
  - Informer
  - TimeGPT

## Requirements

- Python 3.8+
- PyTorch
- pandas
- numpy
- nixtla
- neuralforecast
- sklearn

## Setup

1. Install the required packages:

2. Get your TimeGPT API key from https://www.nixtla.io/

3. Set your API key in both main.py and neural_test.py

## Usage

### Running Adversarial Attacks

python main.py
This will:
- Load the exchange rate dataset
- Apply both clean predictions and adversarial attacks
- Compare prediction errors between clean and attacked data

### Benchmarking Different Models
python neural_test.py
This will:
- Load the exchange rate dataset
- Train and predict using various neural forecasting models
- Compare prediction errors between different models
- Save results to 'exchange_results_with_timegpt_48_new.csv'

python neural_test.py

This will:
- Compare different neural forecasting models
- Generate performance metrics (MAE and MSE)
- Save results to 'exchange_results_with_timegpt_48_new.csv'

## Data

The project uses the exchange rate dataset by default. The data should be in CSV format with:
- A date column ('ds')
- A target column ('y')

## Attack Methods

1. **GWN (Gaussian White Noise)**
   - Adds random Gaussian noise to the input data

2. **DGA (Directional Gradient Attack)**
   - Uses gradient information to generate adversarial perturbations
   - Optimized for maximum prediction error

3. **SPSA (Simultaneous Perturbation Stochastic Approximation)**
   - Gradient-free optimization method
   - Useful for black-box attacks

4. **ITE (Iterative Attack)**
   - Iterative optimization process
   - Allows for controlled perturbation magnitude

## Notes

- The scale of attacks can be adjusted using the 'scale' parameter
- Default forecast horizon is 48 steps
- Input sequence length is 96 steps by default