- install uv by wget:
`wget -qO- https://astral.sh/uv/install.sh | sh`
- create python venv by uv
`uv venv --python 3.12 --seed mimic_basline_2`
- activate python env
`source mimic_baseline_2/bin/activate`
- install isaacsim6.0 by uv
`uv pip install --upgrade pip`
`uv pip install "isaacsim[all,extscache]==6.0.0" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow`
- install isaaclab
  ```
  git clone https://github.com/isaac-sim/IsaacLab.git
  cd IsaacLab
  ./isaaclab.sh --install
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Cartpole-Direct-v0 --num_envs=16384 --max_iterations=100 presets=newton
  ```
- install rsl-rl
  ```
  cd ..
  git clone https://github.com/leggedrobotics/rsl_rl.git -b v5.0.1
  cd rsl_rl/
  pip install -e .
  ```
