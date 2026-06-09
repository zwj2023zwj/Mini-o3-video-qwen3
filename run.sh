cd /newdisk/maodawei/ZhangWeijie/Model/Mini-o3-video-main_qwen3

export BASE_IMAGE_DIR="/newdisk/maodawei/ZhangWeijie/Dataset/Seeker-173K/"
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1

CACHE_ROOT=/newdisk/maodawei/ZhangWeijie/rc
RAY_CACHE_ROOT=/newdisk/maodawei/ZhangWeijie/r
TMP_CACHE_ROOT=/newdisk/maodawei/ZhangWeijie/t
mkdir -p "${CACHE_ROOT}"/{xdg,vllm,torchinductor,triton,wandb} "${RAY_CACHE_ROOT}" "${TMP_CACHE_ROOT}"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TMPDIR="${TMP_CACHE_ROOT}"
export RAY_TMPDIR="${RAY_CACHE_ROOT}"
export WANDB_DIR="${CACHE_ROOT}/wandb"

export SELF_SET_OVERVIEW_FPS=1
export SELF_SET_FPS_MAX_FRAMES=16
RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR=./logs
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run_${RUN_TIMESTAMP}.log"

echo "Full log will be saved to: ${LOG_FILE}"
echo "Terminal only shows key progress, scores, warnings, and errors."

TRAIN_DATA=/newdisk/maodawei/ZhangWeijie/Dataset/Seeker-173K/videos/youtube_video_2024_rl_subset__tool_penalty_single_turn.json
VAL_DATA=/newdisk/maodawei/ZhangWeijie/Dataset/Seeker-173K/videos/youtube_video_2024_rl_subset__tool_penalty_single_turn.json

PYTHON_BIN=/newdisk/maodawei/ZhangWeijie/miniconda_envs/video-o3-qwen3vl-py312/bin/python

CUDA_VISIBLE_DEVICES=0,2,5,7 "${PYTHON_BIN}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.system_prompt="tool_crop" \
  data.train_files=[${TRAIN_DATA}] \
  data.val_files=[${VAL_DATA}] \
  data.train_batch_size=4 \
  trainer.val_before_train=False \
  data.val_batch_size=4 \
  trainer.num_workers=4 \
  data.max_prompt_length=8192 \
  data.max_response_length=2048 \
  data.image_key=images \
  data.video_key=video \
  data.answer_key=solution \
  data.mask_blank=False \
  data.acc_reward_weight=1.0 \
  data.format_reward_weight=1.0 \
  data.tool_call_penalty=0 \
  data.general_qa_reward_fn="general_qa_tool_mc" \
  data.gpt_general_qa_reward_fn="general_qa_tool" \
  data.gpt_extract_answer=False \
  data.extract_answer_tags="strict" \
  data.return_raw_chat=True \
  data.gpt_threads=300 \
  data.tool_call="crop" \
  data.use_tgt_size=False \
  data.max_pixels=16384 \
  data.min_pixels=512 \
  reward_model.reward_manager=naive_multithreads_tool \
  reward_model.use_hybrid_reward_manager=False \
  actor_rollout_ref.actor.ignore_exceed=True \
  actor_rollout_ref.model.path=/newdisk/maodawei/ZhangWeijie/BaseModel/Qwen3-VL-4B-Instruct \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.entropy_coeff=0.000 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.use_multi_turn_response_mask=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
  actor_rollout_ref.rollout.name=vllm_multi_turn_tool_call \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.max_generation_round=6 \
  'actor_rollout_ref.rollout.limit_mm_per_prompt={video:3}' \
  actor_rollout_ref.rollout.val_max_generation_round=6 \
  'actor_rollout_ref.rollout.val_limit_mm_per_prompt={video:3}' \
  actor_rollout_ref.rollout.use_raw_image=True \
  actor_rollout_ref.rollout.multi_turn_prompt_type="v2" \
  actor_rollout_ref.rollout.vllm_infer_batch_size=1 \
  actor_rollout_ref.rollout.mode="async" \
  actor_rollout_ref.rollout.use_relative_coordinates=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.critic_warmup=0 \
  trainer.logger=['console','wandb'] \
  trainer.project_name='Mini-o3-video' \
  trainer.experiment_name='seek-RL' \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=200 \
  trainer.default_local_dir=./save_seek_${RUN_TIMESTAMP} \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  trainer.log_training_rollouts_freq=10 \
  trainer.train_generations_to_log_to_wandb=5 \
  trainer.val_generations_to_log_to_wandb=0 \
  trainer.use_3drope=True \
  trainer.rejection_sample=True \
  trainer.rejection_sample_multiplier=1 \
  2>&1 | tee "${LOG_FILE}" | grep --line-buffered -E "TRAINING_MARK|<<< \\[score\\]|WARNING|ERROR|Traceback|Exception|global_steps|test_score|train/|val/|Saving|saved"

PY_STATUS=${PIPESTATUS[0]}
echo "Full log saved to: ${LOG_FILE}"
exit "${PY_STATUS}"
