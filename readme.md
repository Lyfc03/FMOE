# Missing Pieces, Complete Picture: Navigating Micro-Video Popularity with Flexible Mixture of Modality Experts

> This is the source code for *Missing Pieces, Complete Picture: Navigating Micro-Video Popularity with Flexible Mixture of Modality Experts*.

## Configure the Environment

Create the running environment and install pytorch:

```bash
conda create -n FMOE python=3.9

conda activate FMOE

pip3 install torch --index-url https://download.pytorch.org/whl/cu121
```

Install other relevant Python libraries:

```bash
pip install -r requirements.txt
```

## Run the Code

Firstly, set the parameters under the **config** folder. 

Then modify the operating configuration to set the working directory as the **root directory of the entire project**: FMOE/

Finally, just run the code!

```bash
# clone the repo and go into it:
cd FMOE

# training, validing and testing 
python run.py
```
