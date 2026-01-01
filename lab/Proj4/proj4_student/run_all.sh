#!/bin/bash

TAGS=("bark" "coffee" "rose" "stucco" "water" "beehive")
LAYERS=("1" "2" "3")


LOG_DIR="log"
if [ ! -d "${LOG_DIR}" ]; then
    mkdir -p "${LOG_DIR}"
fi

# 双层循环：遍历所有 tag + layer 组合
for TAG in "${TAGS[@]}"; do
    echo "====================================="
    echo "当前处理 ${TAG}"
    echo "====================================="
    
    TAG_CLEAN=$(echo "${TAG}" | sed 's/\.jpg//g')
    
    for LAYER in "${LAYERS[@]}"; do

        LOG_FILE="${LOG_DIR}/layer_${LAYER}_${TAG_CLEAN}_log.txt"
        
        echo -e "layer=${LAYER} | tag=${TAG}"
        
        
        python deep_frame.py --layer ${LAYER} --tag ${TAG} > "${LOG_FILE}" 2>&1
        
    done
    
    echo -e "\n-------------------------------------\n"
done

echo "====================================="
echo "所有实验执行完毕！"