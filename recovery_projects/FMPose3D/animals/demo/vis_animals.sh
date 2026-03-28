#Test
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

layers=5
gpu_id=0
sample_steps=3
batch_size=1
sh_file='vis_animals.sh'
# n_joints=26
# out_joints=26

model_type='fmpose3d_animals'
# model_path=''  # set to a local file path to override the registry
saved_model_path='../pre_trained_models/fmpose3d_animals/fmpose3d_animals_pretrained_weights.pth'

# path='./images/image_00068.jpg'  # single image
input_images_folder="${REPO_ROOT}/pipeline_data/input/videos/AI_PlayBow.mp4"
output_root="${REPO_ROOT}/pipeline_data/intermediate/fmpose3d"

hypothesis_num=10
aggregation='rpea'     # 'rpea' (paper Eq.10-11) or 'mean'
rpea_topk=5            # top-K filtering for RPEA
rpea_alpha=50.0        # temperature for RPEA weights
bone_norm=True         # enforce median bone lengths across video frames

python vis_animals.py \
 --type 'video' \
 --path ${input_images_folder} \
 --output_root "${output_root}" \
 --saved_model_path "${saved_model_path}" \
 ${model_path:+--model_path "$model_path"} \
 --model_type "${model_type}" \
 --sample_steps ${sample_steps} \
 --batch_size ${batch_size} \
 --layers ${layers} \
 --dataset animal3d \
 --gpu ${gpu_id} \
 --sh_file ${sh_file} \
 --hypothesis_num ${hypothesis_num} \
 --aggregation ${aggregation} \
 --topk ${rpea_topk} \
 --rpea_alpha ${rpea_alpha} \
 --bone_norm ${bone_norm}