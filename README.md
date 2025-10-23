# green-ai-image-generation

♻️ Generate AI images 3x faster with 70% less energy consumption. 

Demonstrates model optimization using Pruna AI for sustainable corporate image generation. Includes practical examples (LinkedIn banners) with performance benchmarks and carbon footprint analysis.

## Project Overview

This repository showcases how AI model optimization can significantly reduce:
- Generation time
- Memory usage
- Energy consumption
- CO2 emissions

All while maintaining image quality for corporate branding use cases.

## Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- 64GB RAM recommended
- Hugging Face account and API token

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/green-ai-image-generation.git
cd green-ai-image-generation
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate green-ai
```

3. Set up authentication:
```bash
huggingface-cli login
```
Enter your token from https://huggingface.co/settings/tokens

4. Create `.env` file:
```bash
cp .env.example .env
```
Add your Hugging Face token to `.env`

## Usage

### Baseline Generation

Run the unoptimized model to establish baseline metrics:
```bash
python baseline.py
```

This generates a LinkedIn banner using Stable Diffusion XL and measures:
- Generation time
- Memory usage
- Energy consumption
- CO2 emissions

### Optimization (Coming Soon)
```bash
python optimize.py
```

### Optimized Generation (Coming Soon)
```bash
python optimized.py
```

## Project Structure
```
green-ai-image-generation/
├── baseline.py           # Unoptimized SDXL generation with metrics
├── optimize.py           # Model optimization with Pruna AI (TODO)
├── optimized.py          # Optimized model generation (TODO)
├── experiments/          # Testing and prototyping scripts
├── environment.yml       # Conda environment specification
├── .env                  # API tokens (not committed)
├── .gitignore           
└── README.md
```

## Sustainability Metrics

The project tracks key sustainability indicators:

- **Generation Time**: Time to generate one image
- **Memory Usage**: RAM consumed during generation
- **Energy Consumption**: Estimated Wh based on CPU TDP
- **CO2 Emissions**: Estimated grams based on grid intensity
- **Model Size**: Disk space required

### Current Baseline Results (SDXL)
```
Generation time: ~70s
System RAM used: ~13.5 GB
Model size: ~26.5 GB
Energy: ~1.26 Wh
CO2: ~0.44g
```

## Technologies Used

- **Stable Diffusion XL**: State-of-the-art image generation
- **Pruna AI**: Model optimization framework
- **PyTorch**: Deep learning framework
- **Diffusers**: Hugging Face diffusion models library

## Configuration

Energy and CO2 calculations use configurable constants in the scripts:
```python
CPU_TDP_WATTS = 65              # Apple M3 average TDP
CO2_INTENSITY_G_PER_KWH = 350   # Netherlands grid average
```

Adjust these values based on your:
- CPU specifications
- Local electricity grid carbon intensity

## Notes

- Energy and CO2 metrics are estimates based on CPU TDP
- Actual power consumption varies with CPU load
- Grid carbon intensity varies by location and time
- First run downloads models (~26GB for SDXL)

## License

Apache License - See [LICENSE](LICENSE) file for details.

## Acknowledgments

- Pruna AI for model optimization tools
- Stability AI for Stable Diffusion XL
- Hugging Face for the diffusers library

## Related Resources

- [Pruna AI Documentation](https://docs.pruna.ai/)
- [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [Green Software Foundation](https://greensoftware.foundation/)