# -*- coding: utf-8 -*-
"""
多模态视频检索系统配置文件
"""

import torch
import os

# 基础配置
class Config:
    # 设备配置
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 数据路径配置
    VIDEO_DATASET_PATH = './video_dataset/video_dataset'
    QUERY_DATASET_PATH = './query_dataset/query_dataset'
    QUERY_CSV_PATH = './query.csv'
    SAMPLE_SUBMIT_PATH = './sample_submit.csv'
    OUTPUT_PATH = './submission.csv'
    SUBMISSION_PATH = './submission.csv'
    
    # 特征缓存配置
    FEATURE_CACHE_DIR = './cache/features'
    FEATURE_CACHE_PATH = './cache/features'
    VIDEO_FEATURES_CACHE = os.path.join(FEATURE_CACHE_DIR, 'video_features.pkl')
    
    # 模型配置
    CLIP_MODEL_NAME = 'openai/clip-vit-base-patch32'
    RESNET_MODEL_NAME = 'resnet50'
    WAV2VEC2_MODEL_NAME = 'facebook/wav2vec2-base'
    BERT_MODEL_NAME = 'bert-base-uncased'
    
    # 特征维度配置
    FEATURE_DIM = 512
    TEXT_FEATURE_DIM = 512
    VISUAL_FEATURE_DIM = 512
    AUDIO_FEATURE_DIM = 512
    VIDEO_FEATURE_DIM = 512
    
    # 处理参数配置
    BATCH_SIZE = 32
    VIDEO_FPS = 1  # 视频采样帧率
    AUDIO_SAMPLE_RATE = 16000  # 音频采样率
    IMAGE_SIZE = (224, 224)  # 图像尺寸
    MAX_TEXT_LENGTH = 512  # 文本最大长度
    
    # 特征融合权重
    FUSION_WEIGHTS = {
        'text': 0.4,
        'visual': 0.4,
        'audio': 0.2
    }
    
    # 模态权重配置
    TEXT_WEIGHT = 0.4
    VISUAL_WEIGHT = 0.4
    AUDIO_WEIGHT = 0.2
    
    # 性能优化配置
    MIXED_PRECISION = True
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # 日志配置
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 创建必要的目录
    @staticmethod
    def create_dirs():
        os.makedirs(Config.FEATURE_CACHE_DIR, exist_ok=True)
        os.makedirs('./logs', exist_ok=True)