# -*- coding: utf-8 -*-
"""
多模态视频检索系统特征提取模块
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import (
    CLIPProcessor, CLIPModel,
    Wav2Vec2Processor, Wav2Vec2Model,
    BertTokenizer, BertModel
)
import cv2
import librosa
import numpy as np
from PIL import Image
import logging
from typing import Dict, List, Optional, Union, Tuple
from config import Config
from utils import (
    save_features, load_features, normalize_features,
    to_tensor, manage_gpu_memory, create_progress_bar
)

class FeatureExtractor:
    """多模态特征提取器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = Config.DEVICE
        
        # 初始化模型
        self._init_models()
        
        # 图像预处理
        self.image_transform = transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def _init_models(self):
        """初始化所有模型"""
        self.logger.info("正在初始化特征提取模型...")
        
        try:
            # CLIP模型 - 用于图像和文本特征提取
            self.clip_processor = CLIPProcessor.from_pretrained(Config.CLIP_MODEL_NAME)
            self.clip_model = CLIPModel.from_pretrained(Config.CLIP_MODEL_NAME).to(self.device)
            self.clip_model.eval()
            
            # Wav2Vec2模型 - 用于音频特征提取
            self.wav2vec2_model = Wav2Vec2Model.from_pretrained(Config.WAV2VEC2_MODEL_NAME)
            self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained(Config.WAV2VEC2_MODEL_NAME)
            self.wav2vec2_model.to(self.device)
            self.wav2vec2_model.eval()
            
            # BERT模型 - 用于文本特征提取
            self.bert_model = BertModel.from_pretrained(Config.BERT_MODEL_NAME)
            self.bert_tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
            self.bert_model.to(self.device)
            self.bert_model.eval()
            
            # ResNet模型 - 用于视频帧特征提取
            from torchvision import models
            try:
                self.resnet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            except:
                # 如果下载失败，使用未预训练的模型
                self.resnet_model = models.resnet50(weights=None)
                self.logger.warning("使用未预训练的ResNet模型")
            
            self.resnet_model = torch.nn.Sequential(*list(self.resnet_model.children())[:-1])
            self.resnet_model.eval()
            self.resnet_model.to(self.device)
            
            self.logger.info("所有模型初始化完成")
            
        except Exception as e:
            self.logger.error(f"模型初始化失败: {e}")
            raise
    
    def extract_text_features(self, text: str) -> torch.Tensor:
        """提取文本特征
        
        Args:
            text: 输入文本
            
        Returns:
            torch.Tensor: 文本特征向量
        """
        try:
            # 使用CLIP进行文本编码
            inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            return text_features.squeeze(0)
            
        except Exception as e:
            self.logger.error(f"文本特征提取失败: {e}")
            return torch.zeros(Config.TEXT_FEATURE_DIM, device=self.device)
    
    def extract_audio_features(self, audio_path: str) -> torch.Tensor:
        """提取音频特征
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            torch.Tensor: 音频特征向量
        """
        if not audio_path or not os.path.exists(audio_path):
            return torch.zeros(Config.AUDIO_FEATURE_DIM, device=self.device)
        
        try:
            # 加载音频文件
            audio, sr = librosa.load(audio_path, sr=Config.AUDIO_SAMPLE_RATE)
            
            # 如果音频太短，进行填充
            min_length = Config.AUDIO_SAMPLE_RATE  # 至少1秒
            if len(audio) < min_length:
                audio = np.pad(audio, (0, min_length - len(audio)), mode='constant')
            
            with torch.no_grad():
                # 使用Wav2Vec2提取音频特征
                inputs = self.wav2vec2_processor(audio, sampling_rate=sr, 
                                               return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.wav2vec2_model(**inputs)
                audio_features = outputs.last_hidden_state.mean(dim=1)  # 平均池化
                audio_features = normalize_features(audio_features)
                
                return audio_features.squeeze(0)
                
        except Exception as e:
            self.logger.error(f"音频特征提取失败 {audio_path}: {e}")
            return torch.zeros(Config.AUDIO_FEATURE_DIM, device=self.device)
    
    def extract_image_features(self, image_path: str) -> torch.Tensor:
        """提取图像特征
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            torch.Tensor: 图像特征向量
        """
        if not image_path or not os.path.exists(image_path):
            return torch.zeros(Config.IMAGE_FEATURE_DIM, device=self.device)
        
        try:
            # 加载和预处理图像
            image = Image.open(image_path).convert('RGB')
            
            with torch.no_grad():
                # 使用CLIP提取图像特征
                inputs = self.clip_processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                image_features = self.clip_model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                return image_features.squeeze(0)
                
        except Exception as e:
            self.logger.error(f"图像特征提取失败 {image_path}: {e}")
            return torch.zeros(Config.IMAGE_FEATURE_DIM, device=self.device)
    
    def extract_video_features(self, video_path: str) -> torch.Tensor:
        """提取视频特征
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            torch.Tensor: 视频特征向量
        """
        if not video_path or not os.path.exists(video_path):
            return torch.zeros(Config.VIDEO_FEATURE_DIM, device=self.device)
        
        try:
            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"无法打开视频文件: {video_path}")
                return torch.zeros(Config.VIDEO_FEATURE_DIM, device=self.device)
            
            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 计算采样间隔
            sample_interval = max(1, int(fps / Config.VIDEO_FPS))
            
            frame_features = []
            frame_count = 0
            
            with torch.no_grad():
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # 按间隔采样帧
                    if frame_count % sample_interval == 0:
                        # 转换为RGB格式
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_pil = Image.fromarray(frame_rgb)
                        
                        # 预处理帧
                        frame_tensor = self.image_transform(frame_pil).unsqueeze(0).to(self.device)
                        
                        # 使用ResNet提取帧特征
                        frame_feature = self.resnet_model(frame_tensor)
                        frame_features.append(frame_feature)
                        
                        # 限制最大帧数以控制内存使用
                        if len(frame_features) >= Config.MAX_FRAMES_PER_VIDEO:
                            break
                    
                    frame_count += 1
            
            cap.release()
            
            if not frame_features:
                return torch.zeros(Config.VIDEO_FEATURE_DIM, device=self.device)
            
            # 聚合帧特征
            video_features = torch.stack(frame_features).mean(dim=0)  # 平均池化
            video_features = normalize_features(video_features)
            
            return video_features.squeeze(0)
            
        except Exception as e:
            self.logger.error(f"视频特征提取失败 {video_path}: {e}")
            return torch.zeros(Config.VIDEO_FEATURE_DIM, device=self.device)
    
    def extract_multimodal_features(self, query: Dict) -> torch.Tensor:
        """提取多模态查询特征
        
        Args:
            query: 查询字典，包含text_search, audio_path, image_path等
            
        Returns:
            torch.Tensor: 融合的多模态特征向量
        """
        features = []
        
        # 提取文本特征
        if query.get('text_search'):
            text_features = self.extract_text_features(query['text_search'])
            features.append(text_features * Config.TEXT_WEIGHT)
        
        # 提取音频特征
        if query.get('audio_path'):
            audio_features = self.extract_audio_features(query['audio_path'])
            features.append(audio_features * Config.AUDIO_WEIGHT)
        
        # 提取图像特征
        if query.get('image_path'):
            image_features = self.extract_image_features(query['image_path'])
            features.append(image_features * Config.IMAGE_WEIGHT)
        
        if not features:
            return torch.zeros(Config.MULTIMODAL_FEATURE_DIM, device=self.device)
        
        # 特征融合
        if len(features) == 1:
            multimodal_features = features[0]
        else:
            # 简单的特征拼接
            multimodal_features = torch.cat(features, dim=0)
            
            # 如果需要，可以添加更复杂的融合策略
            if Config.USE_ATTENTION_FUSION:
                multimodal_features = self._attention_fusion(features)
        
        return normalize_features(multimodal_features.unsqueeze(0)).squeeze(0)
    
    def _attention_fusion(self, features: List[torch.Tensor]) -> torch.Tensor:
        """注意力机制特征融合
        
        Args:
            features: 特征列表
            
        Returns:
            torch.Tensor: 融合后的特征
        """
        # 简单的注意力权重计算
        feature_stack = torch.stack(features)
        attention_weights = torch.softmax(feature_stack.norm(dim=-1), dim=0)
        
        # 加权融合
        fused_features = torch.sum(feature_stack * attention_weights.unsqueeze(-1), dim=0)
        return fused_features
    
    def batch_extract_video_features(self, video_paths: List[str], 
                                   cache_dir: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """批量提取视频特征
        
        Args:
            video_paths: 视频路径列表
            cache_dir: 缓存目录
            
        Returns:
            Dict[str, torch.Tensor]: 视频ID到特征的映射
        """
        self.logger.info(f"开始批量提取 {len(video_paths)} 个视频的特征...")
        
        video_features = {}
        progress_bar = create_progress_bar(len(video_paths), "提取视频特征")
        
        for video_path in video_paths:
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            
            # 检查缓存
            if cache_dir:
                cache_path = os.path.join(cache_dir, f"{video_id}_video.pt")
                cached_features = load_features(cache_path)
                if cached_features is not None:
                    video_features[video_id] = cached_features
                    progress_bar.update(1)
                    continue
            
            # 提取特征
            features = self.extract_video_features(video_path)
            video_features[video_id] = features
            
            # 保存缓存
            if cache_dir:
                cache_path = os.path.join(cache_dir, f"{video_id}_video.pt")
                save_features(features, cache_path)
            
            progress_bar.update(1)
            
            # 管理GPU内存
            if len(video_features) % 10 == 0:
                manage_gpu_memory()
        
        progress_bar.close()
        self.logger.info(f"批量特征提取完成，共提取 {len(video_features)} 个视频特征")
        
        return video_features
    
    def batch_extract_query_features(self, queries: List[Dict], 
                                   cache_dir: Optional[str] = None) -> List[torch.Tensor]:
        """批量提取查询特征
        
        Args:
            queries: 查询列表
            cache_dir: 缓存目录
            
        Returns:
            List[torch.Tensor]: 查询特征列表
        """
        self.logger.info(f"开始批量提取 {len(queries)} 个查询的特征...")
        
        query_features = []
        progress_bar = create_progress_bar(len(queries), "提取查询特征")
        
        for i, query in enumerate(queries):
            # 检查缓存
            if cache_dir:
                cache_path = os.path.join(cache_dir, f"query_{i}.pt")
                cached_features = load_features(cache_path)
                if cached_features is not None:
                    query_features.append(cached_features)
                    progress_bar.update(1)
                    continue
            
            # 提取特征
            features = self.extract_multimodal_features(query)
            query_features.append(features)
            
            # 保存缓存
            if cache_dir:
                cache_path = os.path.join(cache_dir, f"query_{i}.pt")
                save_features(features, cache_path)
            
            progress_bar.update(1)
            
            # 管理GPU内存
            if i % 10 == 0:
                manage_gpu_memory()
        
        progress_bar.close()
        self.logger.info(f"批量查询特征提取完成，共提取 {len(query_features)} 个查询特征")
        
        return query_features

def main():
    """测试特征提取功能"""
    from utils import setup_logging
    
    # 设置日志
    setup_logging()
    
    # 创建特征提取器
    extractor = FeatureExtractor()
    
    # 测试文本特征提取
    text = "一个人在跑步"
    text_features = extractor.extract_text_features(text)
    print(f"文本特征维度: {text_features.shape}")
    
    print("特征提取器初始化完成！")

if __name__ == "__main__":
    main()