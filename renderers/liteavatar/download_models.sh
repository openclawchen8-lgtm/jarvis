#!/bin/bash
# LiteAvatar 模型下載腳本
# 從 ModelScope 下載 audio2mouth ONNX 模型 + Paraformer 語言模型
#
# 用法:
#   bash download_models.sh
#
# Avatar 資產另需手動下載:
#   modelscope download --model HumanAIGC-Engineering/LiteAvatarGallery <avatar_id> --local_dir ./avatars/<avatar_name>

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$SCRIPT_DIR/../liteavatar-src"
WEIGHTS_DIR="$SRC_DIR/weights"

echo "=== LiteAvatar 模型下載 ==="
echo ""

# 1. 安裝 modelscope（如未安裝）
if ! python3 -c "import modelscope" 2>/dev/null; then
    echo "[1/3] 安裝 modelscope..."
    pip3 install modelscope
fi

# 2. 下載模型權重
echo "[2/3] 下載 LiteAvatar 模型權重（audio2mouth ONNX + Paraformer）..."
cd "$SRC_DIR"
python3 -c "
from modelscope import snapshot_download
import os, shutil

print('下載中（約 1.2GB），請稍候...')
model_dir = snapshot_download(
    'HumanAIGC-Engineering/LiteAvatarGallery',
    cache_dir='./cache_modelscope',
    allow_patterns=['lite_avatar_weights/*'],
)

src = os.path.join(model_dir, 'lite_avatar_weights')
for f in os.listdir(src):
    dst = os.path.join('weights', f)
    fp_src = os.path.join(src, f)
    if not os.path.exists(dst):
        if os.path.isfile(fp_src):
            shutil.copy2(fp_src, dst)
            size = os.path.getsize(dst) / 1024 / 1024
            print(f'  ✓ {f} ({size:.1f}MB)')
        elif os.path.isdir(fp_src):
            shutil.copytree(fp_src, dst)
            print(f'  ✓ {f}/')

# Move files to correct locations
if os.path.exists('weights/lm.pb'):
    os.makedirs('weights/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/lm', exist_ok=True)
    shutil.move('weights/lm.pb', 'weights/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/lm/lm.pb')
    print('  ✓ lm.pb → paraformer/lm/')
if os.path.exists('weights/model.pb'):
    shutil.move('weights/model.pb', 'weights/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/model.pb')
    print('  ✓ model.pb → paraformer/')
if os.path.exists('weights/model_1.onnx'):
    print('  ✓ model_1.onnx já no local correto')
"

echo ""
echo "[3/3] 清理快取..."
rm -rf "$SRC_DIR/cache_modelscope"

echo ""
echo "=== 完成 ==="
echo ""
echo "下一步：下載 Avatar 資產"
echo "  modelscope download --model HumanAIGC-Engineering/LiteAvatarGallery \\"
echo "    <avatar_id> --local_dir ./renderers/liteavatar/avatars/<name>"
echo ""
echo "可用 Avatar Gallery: https://modelscope.cn/models/HumanAIGC-Engineering/LiteAvatarGallery/summary"
