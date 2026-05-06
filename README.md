# Setup

## Installation

1. Set up a virtual environment. This project is developed using Python 3.8:
   ```bash
   conda create -n PFNav python==3.8
   conda activate PFNav
   ```

2. For machines with multiple GPUs or without a connected display (e.g. a compute cluster), install `habitat-sim-v0.1.7`:
   ```bash
   git clone https://github.com/facebookresearch/habitat-sim.git
   cd habitat-sim
   git checkout tags/v0.1.7
   pip install -r requirements.txt
   python setup.py install --headless
   ```

3. Install `habitat-lab-v0.1.7`:
   ```bash
   git clone https://github.com/facebookresearch/habitat-lab.git
   cd habitat-lab
   git checkout tags/v0.1.7
   cd habitat_baselines/rl
   vi requirements.txt # remove the line: tensorflow==1.13.1
   cd ../../ # return to the habitat-lab root directory
   
   pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
   
   pip install -r requirements.txt
   python setup.py develop --all # installs both habitat and habitat_baselines; if it fails, retry — failures are usually caused by network issues
   ```
   If you run into difficulties during installation, please refer to the [Official Habitat Installation Guide](https://github.com/facebookresearch/habitat-lab#installation) for step-by-step instructions on setting up [`habitat-lab`](https://github.com/facebookresearch/habitat-lab) and [`habitat-sim`](https://github.com/facebookresearch/habitat-sim). Our experiments use version [`v0.1.7`](https://github.com/facebookresearch/habitat-lab/releases/tag/v0.1.7), consistent with VLN-CE. See the [VLN-CE repository](https://github.com/jacobkrantz/VLN-CE) for further details.

4. Install Grounded-SAM:
   ```bash
   git clone https://github.com/IDEA-Research/GroundingDINO.git
   cd GroundingDINO
   git checkout -q 57535c5a79791cb76e36fdb64975271354f10251
   pip install -q -e .
   pip install 'git+https://github.com/facebookresearch/segment-anything.git'
   ```

5. Install the remaining dependencies:
   ```bash
   git clone https://github.com/LiLingFeng-ydmy/PFNav.git
   cd PFNav-code
   pip install requirements.txt
   pip install requirements2.txt
   ```

---

## Datasets

1. **R2R-CE**
   - **Instructions:** Download the preprocessed `R2R_VLNCE_v1-3_preprocessed` instructions from the [VLN-CE repository](https://github.com/jacobkrantz/VLN-CE).
   - **Scenes:** This project relies on Matterport3D (MP3D) scene reconstructions. To obtain the official download script (`download_mp.py`), follow the instructions on the [Matterport3D project page](https://niessner.github.io/Matterport/). Then download the scene data with:
   ```bash
   # must be run with Python 2.7
   python download_mp.py --task habitat -o data/scene_datasets/mp3d/
   ```
   After extraction, ensure the directory follows the structure `scene_datasets/mp3d/{scene}/{scene}.glb`. There should be 90 scenes in total. Place the `scene_datasets` folder inside `data/`.

2. **PFNav LLM Replies / BLIP2-ITM / BLIP2-VQA / Grounded-SAM:** Download all required files from [Google Drive](https://drive.google.com/drive/folders/1fHUDDnK-gNNABrcb5u_F93mAQhu8tC8z?usp=sharing).

The overall data directory should be organized as follows:

```
PFNav-code
├── data
│   ├── blip2
│   ├── datasets
│       ├── LLM_REPLYS_VAL_UNSEEN
│       ├── R2R_VLNCE_v1-3_preprocessed
│   ├── grounded_sam
│   ├── logs
│   ├── scene_datasets
│   └── vqa
└── ...
```

---

## Running

```bash
cd PFNav-code
sh run_r2r/main.sh
```
