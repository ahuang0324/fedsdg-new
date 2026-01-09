如果要运行python 代码，请先激活conda环境
conda activate moqy_py310


优先安装torch

pip install -r requirement.txt


python -c "import torch; print(f'GPU是否可用: {torch.cuda.is_available()}'); print(f'当前显卡: {torch.cuda.get_device_name(0)}')"