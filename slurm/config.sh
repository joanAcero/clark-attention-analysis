# Paths used by all scripts in slurm/. Edit to match your setup.
# Submit every job from the repository root, e.g.:  sbatch -A <project> slurm/00_tests.sh

CONDA_BASE=/idiap/temp/jacero/miniconda3
CONDA_ENV=bert-norm-analysis
PROJECT=/idiap/temp/jacero/projects/norm_attention_analysis
REPO=$PROJECT/clark-kobayashi-attention-analysis
DATA=$PROJECT/data                 # inputs and extracted maps (large)
RESULTS=$PROJECT/results           # executed notebooks
BERT=$DATA/models/bert-base-uncased
KOBAYASHI_SRC=/idiap/temp/jacero/kobayashi-norm-analysis-of-transformer/emnlp2020/transformers/src

export HF_HOME=/idiap/temp/jacero/cache/huggingface
export PIP_CACHE_DIR=/idiap/temp/jacero/cache/pip
export KOBAYASHI_SRC

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate $CONDA_ENV
cd $REPO

echo "host: $(hostname)  python: $(which python)"
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| transformers', transformers.__version__)"
