# sdr-eval

A terminal tool for calculating SDR (Signal-to-Distortion Ratio), designed to quickly evaluate the performance of music separation models.

## Introduction

This tool provides a convenient way to assess the quality of audio separation models by calculating the SDR between reference and estimated audio files.

## Installation

```bash
git clone https://github.com/AliceNavigator/sdr-eval.git
cd sdr-eval
conda create -n sdr-eval python=3.10
conda activate sdr-eval
pip install -r requirements.txt
```

## Usage

```
python sdr.py [-h] [--reference_folder REFERENCE_FOLDER] [--estimated_folder ESTIMATED_FOLDER] [--num_workers NUM_WORKERS]
```

### Options:

- `-h, --help`: Show this help message and exit
- `--reference_folder REFERENCE_FOLDER`: Folder containing reference audio files to process
- `--estimated_folder ESTIMATED_FOLDER`: Folder containing estimated audio files to process
- `--num_workers NUM_WORKERS`: Number of workers used for SDR computation

## Example

```bash
python sdr.py --reference_folder ./ref_audio --estimated_folder ./est_audio --num_workers 4
```

## License

[MIT License](LICENSE)
