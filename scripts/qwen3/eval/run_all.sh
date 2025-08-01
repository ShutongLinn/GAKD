set -e
bash ./eval_all_dolly.sh
bash ./eval_all_self_inst.sh
bash ./eval_all_vicuna.sh

bash ./eval_all_sinst.sh
# bash ./eval_all_uinst.sh