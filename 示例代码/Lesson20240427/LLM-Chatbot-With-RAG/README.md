# 须知

## env 文件

要使用某些 LLM 模型，需要创建一个包含 `ACCESS_TOKEN=<your Hugging Face token>` 的 `.env` 文件

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行项目

```bash
streamlit run src/app.py
```

## 确保在你拥有GPU的条件下使用量化工具

- 使用如下函数验证GPU是否可用

```python
import torch
print(torch.cuda.is_available())
```

- 如果你没有GPU环境，修改项目中的代码,并移除相关量化的代码

```phython
device="cpu"
```