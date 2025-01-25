# Adversarial Attack on Time Series Forecasting Models

This project implements and evaluates adversarial attacks against various time series forecasting models, with a focus on comparing TimeGPT with other neural forecasting architectures.

## Project Structure

- `attack_time_model.py`: Focuses on evaluating TimeGPT's robustness against DGA (Directional Gradient Attack)
  - Uses standardized data
  - Implements sliding window prediction (step size = horizon)
  - Evaluates both clean and attacked performance

- `attack_different_models.py`: Comprehensive evaluation of multiple models
  - Evaluates TimeGPT and neural forecasting models
  - Uses DGA for TimeGPT and GWN for other models
  - Maintains consistent evaluation metrics

## Attack Methods

1. **DGA (Directional Gradient Attack)**
   - Used specifically for TimeGPT
   - Calculates gradient direction through API queries
   - Applies perturbation in original data space
   - Scale of attack: 2% of data mean value

2. **GWN (Gaussian White Noise)**
   - Used for neural forecasting models
   - Adds random Gaussian noise
   - Same scale as DGA for fair comparison

## Data Processing

- Data split: 60% training, 20% validation, 20% testing
- Input window: 96 timesteps
- Prediction horizon: 48 timesteps
- Standardization applied before model input
- Sliding window evaluation with horizon-length steps

## Models Evaluated

1. TimeGPT (with DGA attack)
2. LSTM
3. NHITS
4. iTransformer
5. PatchTST
6. TimesNet
7. NLinear

## Requirements

- Python 3.8+
- PyTorch
- pandas
- numpy
- nixtla (for TimeGPT)
- neuralforecast
- sklearn

## Usage

1. Set up TimeGPT API key in both main.py and neural_test.py

2. Run TimeGPT evaluation:
```bash
python main.py
```
- Evaluates TimeGPT with DGA attack
- Outputs clean and attacked performance metrics

3. Run comprehensive model evaluation:
```bash
python neural_test.py
```
- Evaluates all models
- Uses appropriate attack method for each model
- Saves results in CSV format

## Output Format

Results are saved in CSV files with columns:
- Model: Model name
- Clean_MAE: Mean Absolute Error without attack
- Clean_MSE: Mean Squared Error without attack
- Attack_MAE: Mean Absolute Error under attack
- Attack_MSE: Mean Squared Error under attack

## Notes

- TimeGPT requires API key from nixtla.io
- Different attack methods (DGA vs GWN) are used based on model type
- All evaluations use consistent metrics and data processing
- Results are saved with dataset name, input length, and horizon in filename
