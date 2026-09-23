#!/bin/bash

set -euo pipefail

# ===================== 参数区 =====================

# 全局固定参数
NUM_GPUS=8
NUM_MICROBATCHES=16

# 模型清单（控制输出顺序）
MODELS=(1p7b 8b 20b 32b 235b)

# 调度器清单（控制输出顺序）
SCHEDULERS=(1f1b interleaved-1f1b gpipe looped-bfs roundpipe)

# 模型基础参数：模型名、输出前缀、层数、时间参数
declare -A MODEL_NAME=(
    [1p7b]="Qwen3-1.7B"
    [8b]="Llama3-8B"
    [20b]="GPTOSS-20B"
    [32b]="Qwen3-32B"
    [235b]="Qwen3-235B"
)

declare -A MODEL_OUTPUT_PREFIX=(
    [1p7b]="1.7b"
    [8b]="8b"
    [20b]="20b"
    [32b]="32b"
    [235b]="235b"
)

declare -A NUM_LAYERS=(
    [1p7b]="28"
    [8b]="32"
    [20b]="24"
    [32b]="64"
    [235b]="94"
)

declare -A FWD_TIME_PER_LAYER=(
    [1p7b]="7.429"
    [8b]="18.50370313"
    [20b]="21.6445625"
    [32b]="40.5610625"
    [235b]="43.016"
)

declare -A BWD_TIME_PER_LAYER=(
    [1p7b]="15.831"
    [8b]="41.20659375"
    [20b]="81.780875"
    [32b]="91.17429688"
    [235b]="58.376"
)

declare -A LMHEAD_FWD_TIME=(
    [1p7b]="22.0655"
    [8b]="37.2255"
    [20b]="44.2925"
    [32b]="54.928"
    [235b]="42.444"
)

declare -A COMM_TIME=(
    [1p7b]="1.6"
    [8b]="3.2"
    [20b]="2.25"
    [32b]="4"
    [235b]="3.2"
)

declare -A LMHEAD_BWD_TIME=(
    [1p7b]="43.707"
    [8b]="74.501"
    [20b]="86.264"
    [32b]="108.294"
    [235b]="43.294"
)

# 调度器后缀参数
declare -A SCHED_OUTPUT_SUFFIX=(
    [1f1b]="1f1b"
    [interleaved-1f1b]="i1f1b"
    [gpipe]="gpipe"
    [looped-bfs]="looped-bfs"
    [roundpipe]="roundpipe"
)

declare -A SCHED_TITLE_SUFFIX=(
    [1f1b]="1F1B"
    [interleaved-1f1b]="Interleaved-1F1B"
    [gpipe]="GPipe"
    [looped-bfs]="Looped-BFS"
    [roundpipe]="RoundPipe"
)

# stage-partition 与“模型 + 调度器”绑定
declare -A STAGE_PARTITION=(
    ["1f1b|1p7b"]="3,3,4,4,4,4,4,2"
    ["1f1b|8b"]="4,4,4,4,4,4,4,4"
    ["1f1b|20b"]="3,3,3,3,3,3,3,3"
    ["1f1b|32b"]="8,8,8,8,8,8,8,8"
    ["1f1b|235b"]="11,12,12,12,12,12,12,11"

    ["interleaved-1f1b|1p7b"]="1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,0"
    ["interleaved-1f1b|8b"]="1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
    ["interleaved-1f1b|20b"]="1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
    ["interleaved-1f1b|32b"]="1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
    ["interleaved-1f1b|235b"]="0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0"

    ["gpipe|1p7b"]="3,3,4,4,4,4,4,2"
    ["gpipe|8b"]="4,4,4,4,4,4,4,4"
    ["gpipe|20b"]="3,3,3,3,3,3,3,3"
    ["gpipe|32b"]="8,8,8,8,8,8,8,8"
    ["gpipe|235b"]="11,12,12,12,12,12,12,11"

    ["looped-bfs|1p7b"]="1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,0"
    ["looped-bfs|8b"]="1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
    ["looped-bfs|20b"]="1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
    ["looped-bfs|32b"]="1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
    ["looped-bfs|235b"]="0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0"

    ["roundpipe|1p7b"]="7,7,7,7|0,3,3,3,3,3,3,3,3,2,2"
    ["roundpipe|8b"]="5,5,5,5,6,6|0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
    ["roundpipe|20b"]="4,5,5,5,5|0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
    ["roundpipe|32b"]="2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3|0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
    ["roundpipe|235b"]="2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2|0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
)

# ===================== 执行区 =====================

# 统计输出自动写入 ${LOG}（默认 ./log），内容与以前的 `bash run.sh > log` 完全一致：
# plot_bubble.py / plot_ideal.py / plot_ideal-course.py 都默认读取名为 log 的文件，
# 且解析时会跳过不含 "=" 的行并以 scheduler= 分隔记录，所以这里不往 stdout 加任何
# 装饰性内容。进度信息一律走 stderr，因此永远不会混进 log。
LOG="${LOG:-log}"
exec > "${LOG}"

total=0
for scheduler in "${SCHEDULERS[@]}"; do
    for model in "${MODELS[@]}"; do
        [[ -n "${STAGE_PARTITION[${scheduler}|${model}]:-}" ]] && total=$((total + 1))
    done
done
printf 'writing simulator output to %s (%d runs)\n' "${LOG}" "${total}" >&2

done_n=0
for scheduler in "${SCHEDULERS[@]}"; do
    for model in "${MODELS[@]}"; do
        key="${scheduler}|${model}"
        stage_partition="${STAGE_PARTITION[$key]:-}"

        if [[ -z "${stage_partition}" ]]; then
            continue
        fi

        done_n=$((done_n + 1))
        printf '[%2d/%2d] %-18s %-5s -> %s\n' "${done_n}" "${total}" \
            "${scheduler}" "${model}" \
            "${MODEL_OUTPUT_PREFIX[$model]}_${SCHED_OUTPUT_SUFFIX[$scheduler]}.pdf" >&2

        python simulator.py \
            --scheduler "${scheduler}" \
            --num-gpus "${NUM_GPUS}" \
            --num-microbatches "${NUM_MICROBATCHES}" \
            --stage-partition "${stage_partition}" \
            --num-layers "${NUM_LAYERS[$model]}" \
            --fwd-time-per-layer "${FWD_TIME_PER_LAYER[$model]}" \
            --bwd-time-per-layer "${BWD_TIME_PER_LAYER[$model]}" \
            --lmhead-fwd-time "${LMHEAD_FWD_TIME[$model]}" \
            --lmhead-bwd-time "${LMHEAD_BWD_TIME[$model]}" \
            --comm-time "${COMM_TIME[$model]}" \
            --output "${MODEL_OUTPUT_PREFIX[$model]}_${SCHED_OUTPUT_SUFFIX[$scheduler]}.pdf" \
            --title "${MODEL_NAME[$model]} ${SCHED_TITLE_SUFFIX[$scheduler]}"
    done
done

printf 'done: %d runs -> %s\n' "${done_n}" "${LOG}" >&2
