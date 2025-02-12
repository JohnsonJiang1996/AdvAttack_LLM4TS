# Adversarial Attack on Time Series Forecasting Models

This project implements and evaluates adversarial attacks against various time series forecasting models, with a focus on comparing TimeGPT with other neural forecasting architectures.

## Project Structure

- `example_attack_timegpt.py`: provides an end-to-end example of manipulating [TimeGPT](https://docs.nixtla.io/)
  - Evaluate TimeGPT's robustness against DGA (Directional Gradient Attack)
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

3. **SPSA (Simultaneous Perturbation Stochastic Approximation)**
   - Gradient-free optimization method
   - Uses random perturbations to estimate gradients
   - Implemented but not included in main paper results

4. **ITA (Iterative Attack)**
   - Iterative optimization process
   - Allows for controlled perturbation magnitude
   - Implemented but not included in main paper results

Note: While SPSA and ITA are implemented in our codebase (see attack.py), we focus on DGA and GWN in our paper experiments due to their better performance and efficiency in the API query setting. Users can still experiment with these additional attack methods using our implementation. You can use our attack demo to choose different attack types.

## LLMTime Attack Integration

We provide integration with the LLMTime framework for zero-shot time series forecasting:

1. Clone the LLMTime repository:
```bash
git clone https://github.com/ngruver/llmtime.git
```

2. Install our attack files:
   - Copy all files from `llmtime_attack` folder to the cloned LLMTime repository
   - Replace the original demo files with our attack implementation

3. Set up OpenAI API:
   - You'll need an OpenAI API key for LLMTime testing
   - Multiple API calls are made during the query phase
   - We recommend starting with small-scale tests to manage API usage
   - You can use our attack demo to choose different attack types

4. API Usage Notes:
   - Each attack iteration requires multiple API calls
   - Start with a small number of test cases to estimate API consumption
   - Monitor your API usage to avoid unexpected costs

## Data Processing

- Data split: 60% training, 20% validation, 20% testing
- Input window: 96 timesteps
- Prediction horizon: 48 timesteps
- Data Standardization:
  * Calculate mean and std from training data
  * Apply (x - mean)/std standardization
  * Attack scale set to 2% of original data mean
  * Standardization applied before model input and after attacks
- Sliding window evaluation with horizon-length steps

## Models Evaluated

1. TimeGPT (with DGA attack)
2. LSTM
3. NHITS
4. iTransformer
5. PatchTST
6. TimesNet
7. NLinear

## LLM Models (tested in LLMTime framework)

These models require API access and are tested using the LLMTime repository:

1. GPT-3 (requires OpenAI API)
2. GPT-4 (requires OpenAI API)
3. Mistral (requires Mistral API)

Note: For LLM models testing:
- Set up respective API keys in the LLMTime environment
- Add to your ~/.bashrc:
  ```bash
  export OPENAI_API_KEY=<your OpenAI key>
  export MISTRAL_KEY=<your Mistral key>
  ```
- These models are tested using LLMTime's zero-shot forecasting approach
- API calls are made for both model predictions and attack evaluations
- Consider API usage costs when planning experiments

## Requirements

- Python 3.8+
- PyTorch
- pandas
- numpy
- nixtla (for TimeGPT)
- neuralforecast
- sklearn

## API Requirements
- TimeGPT API key (from nixtla.io)
- OpenAI API key (for GPT-3 and GPT-4)
- Mistral API key (for Mistral models)

Note: All API keys should be properly set in your environment before running experiments. Free credits are available for initial testing with TimeGPT.

## Usage

1. Set up API keys:
   - Register at https://www.nixtla.io/ for TimeGPT
   - TimeGPT provides free API credits for initial testing
   - Set up OpenAI API key for LLMTime integration
   - API keys need to be set in both attack_different_models and attack_time_model.py

2. Run TimeGPT evaluation:
```bash
python attack_time_model.py
```
- Evaluates TimeGPT with DGA attack
- Outputs clean and attacked performance metrics

3. Run comprehensive model evaluation:
```bash
python attack_different_models.py
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
  - Free API credits are available for initial testing
  - Each API call counts towards your credit limit
  - DGA attack requires multiple API calls per prediction
- LLMTime integration requires OpenAI API:
  - Start with small-scale tests to manage API usage
  - Each attack iteration makes multiple API calls
  - Monitor API consumption carefully
- Different attack methods (DGA vs GWN) are used based on model type
- All evaluations use consistent metrics and data processing
- Results are saved with dataset name, input length, and horizon in filename
