# 多模态视频检索系统

## 项目简介

本项目是一个基于深度学习的多模态视频检索系统，支持文本到视频、音频到视频、图像到视频的检索功能。系统采用最新的预训练模型，包括CLIP、BERT、Wav2Vec2和ResNet，实现高精度的跨模态检索。

### 功能特色

- **多模态检索**：支持文本、音频、图像三种模态的视频检索
- **GPU加速**：充分利用CUDA进行模型推理和特征计算
- **高精度模型**：使用CLIP、BERT、Wav2Vec2、ResNet等先进模型
- **特征缓存**：智能缓存机制，提升重复查询效率
- **混合精度计算**：优化GPU内存使用和计算速度
- **多模态融合**：智能权重分配，提升检索准确率

## 系统要求

### 硬件要求
- GPU：NVIDIA GPU（推荐8GB+显存）
- 内存：16GB+ RAM
- 存储：10GB+ 可用空间

### 软件要求
- Python 3.8+
- CUDA 11.0+
- PyTorch 1.12+

## 安装步骤

### 1. 克隆项目
```bash
git clone <repository-url>
cd 语音与图像多模态检索挑战赛
```

### 2. 创建虚拟环境
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 验证安装
```bash
python test_system.py
```

## 使用方法

### 基本使用
```bash
python main.py
```

### 命令行参数
```bash
python main.py [选项]

选项：
  --config CONFIG_FILE    指定配置文件路径（默认：config.py）
  --device DEVICE         指定设备（cuda/cpu，默认：auto）
  --batch-size BATCH_SIZE 批处理大小（默认：32）
  --cache-features        启用特征缓存
  --verbose               详细输出模式
  --eval                  运行性能评估
```

### 配置文件
主要配置项在 `config.py` 中：
```python
# 设备配置
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 数据路径
VIDEO_DATASET_PATH = './video_dataset'
QUERY_DATASET_PATH = './query_dataset'
QUERY_CSV_PATH = './query.csv'

# 模型配置
CLIP_MODEL_NAME = 'openai/clip-vit-base-patch32'
BERT_MODEL_NAME = 'bert-base-uncased'
WAV2VEC2_MODEL_NAME = 'facebook/wav2vec2-base-960h'
RESNET_MODEL_NAME = 'resnet50'
```

## 数据格式说明

### 输入数据

#### query.csv
包含查询数据，格式如下：
```csv
text_search,audio_search,image_search,raw_video
"描述文本","audio_001.mp3","image_001.png","target_video.mp4"
```

#### 视频数据集
- 位置：`./video_dataset/`
- 格式：MP4文件
- 命名：任意有效文件名

#### 查询数据集
- 位置：`./query_dataset/`
- 音频文件：MP3格式
- 图像文件：PNG格式

### 输出数据

#### submission.csv
检索结果文件，格式如下：
```csv
raw_video
video_001.mp4
video_002.mp4
...
```

## 项目结构

```
语音与图像多模态检索挑战赛/
├── main.py                 # 主程序入口
├── config.py              # 配置文件
├── data_loader.py         # 数据加载模块
├── feature_extractor.py   # 特征提取模块
├── retrieval_engine.py    # 检索引擎
├── utils.py               # 工具函数
├── test_system.py         # 系统测试
├── requirements.txt       # 依赖包列表
├── README.md             # 项目说明
├── query.csv             # 查询数据
├── sample_submit.csv     # 提交样例
├── video_dataset/        # 视频数据集
├── query_dataset/        # 查询数据集
│   ├── *.mp3            # 音频文件
│   └── *.png            # 图像文件
└── cache/               # 特征缓存目录
```

## 技术架构

### 核心模块

1. **特征提取器（FeatureExtractor）**
   - 文本特征：CLIP + BERT
   - 音频特征：Wav2Vec2
   - 图像特征：CLIP + ResNet
   - 视频特征：CLIP（视频帧采样）

2. **检索引擎（RetrievalEngine）**
   - 余弦相似度计算
   - 多模态特征融合
   - Top-K检索算法

3. **数据加载器（DataLoader）**
   - 批量数据处理
   - 内存优化
   - 数据预处理

### 算法流程

1. **数据预处理**：加载查询数据和视频数据集
2. **特征提取**：提取各模态特征向量
3. **相似度计算**：计算查询与视频的相似度
4. **结果排序**：按相似度排序并输出Top-1结果
5. **结果保存**：生成submission.csv文件

## 性能优化建议

### GPU优化
- 使用混合精度训练（AMP）
- 批量处理减少GPU调用
- 合理设置batch_size

### 内存优化
- 启用特征缓存
- 分批处理大数据集
- 及时释放不用的变量

### 速度优化
- 预计算视频特征
- 使用多进程数据加载
- 优化模型推理

## 常见问题解答

### Q: CUDA内存不足怎么办？
A: 减小batch_size，或使用CPU模式：`python main.py --device cpu`

### Q: 模型下载失败？
A: 检查网络连接，或手动下载模型到本地缓存目录

### Q: 特征提取速度慢？
A: 启用特征缓存：`python main.py --cache-features`

### Q: 检索精度不高？
A: 调整config.py中的融合权重参数

### Q: 提交文件格式错误？
A: 确保submission.csv与sample_submit.csv格式一致

## 模型说明

### 预训练模型
- **CLIP**: 用于图像-文本跨模态理解
- **BERT**: 增强文本语义理解
- **Wav2Vec2**: 音频特征提取
- **ResNet**: 图像特征提取

### 特征维度
- 文本特征：768维
- 音频特征：768维
- 图像特征：2048维
- 视频特征：512维

## 评估指标

系统使用Top-1准确率作为主要评估指标：
```
Accuracy = 正确检索数量 / 总查询数量
```

## 贡献指南

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue
- 发送邮件
- 项目讨论区

## 更新日志

### v1.0.0
- 初始版本发布
- 支持多模态视频检索
- GPU加速优化
- 特征缓存机制

---

**注意**：使用前请确保已正确安装所有依赖包，并具备足够的GPU资源。