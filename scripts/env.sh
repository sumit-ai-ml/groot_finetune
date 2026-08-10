# Activate the GR00T environment. Source this, never execute it.
#
#   source /home/sushi/Documents/groot_finetune/scripts/env.sh
#
# Two things are mandatory and easy to forget:
#   - the venv itself
#   - activate_spark.sh, which puts NVPL / torch / CUDA libs on the linker path.
#     Without it `import torch` dies on libnvpl_lapack_lp64_gomp.so.0.
#
# Never run training through `uv run python` here: uv would re-sync against the
# repo-root pyproject.toml, which targets x86_64, and destroy this environment.
# Activate, then use plain `python`.

export GROOT_PROJECT=/home/sushi/Documents/groot_finetune
export GROOT_REPO="$GROOT_PROJECT/Isaac-GR00T"

source "$GROOT_REPO/.venv/bin/activate"
source "$GROOT_REPO/scripts/activate_spark.sh" > /dev/null

# So `import eval_lib` works from any directory.
export PYTHONPATH="$GROOT_PROJECT/scripts${PYTHONPATH:+:$PYTHONPATH}"

cd "$GROOT_REPO"
