<h1 align="center">Situational Dimensions that drive Event Boundaries</h1>

The code for the 2-pager.

## Setup

1. Download Michelman data from https://zenodo.org/records/10055827 and unzip into [data/datasets](data/datasets/)

You should have a folder `data/datasets/GPT_event_share` now. Alternativelyl specify `GPT_EVENT_SHARE_DIR` in `.env`.

2. Install dependencies

```bash
# create conda environment (optional)
conda create -n sitdim python=3.12
conda activate sitdim

# install dependencies
pip install -r requirements.txt
```

3. Download `punkt` tokenizer from nltk, in your console type `python` or open an interactive session.

```py
import nltk

nltk.download('punkt')
```

4. Run & adapt config in the end of the code to run different files!

```sh
# run code
python single_story_evaluation.py
```
You can see the available ratings in [data/ratings/](data/ratings/)
