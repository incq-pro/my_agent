# My Agent

大模型Agent

## 开发环境搭建

```bash
# 创建虚拟环境
python3 -m venv .venv

# 激活虚拟环境
source .venv/bin/activate

# 安装开发依赖
pip install -e ".[dev]"
```

## 运行测试

```bash
pytest
```

## 代码格式化

```bash
ruff format .
ruff check .
```
