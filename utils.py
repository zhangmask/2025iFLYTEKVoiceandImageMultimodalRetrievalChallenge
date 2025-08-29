# -*- coding: utf-8 -*-
"""
多模态视频检索系统工具函数
"""

import os
import logging
import pickle
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional
from config import Config

def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format=Config.LOG_FORMAT,
        handlers=[
            logging.FileHandler('./logs/system.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def save_features(features: Dict[str, Any], filepath: str):
    """保存特征到文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(features, f)

def load_features(filepath: str) -> Optional[Dict[str, Any]]:
    """从文件加载特征"""
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None

def normalize_features(features: torch.Tensor) -> torch.Tensor:
    """特征归一化"""
    return F.normalize(features, dim=-1)

def cosine_similarity(query_features: torch.Tensor, video_features: torch.Tensor) -> torch.Tensor:
    """计算余弦相似度"""
    query_norm = normalize_features(query_features)
    video_norm = normalize_features(video_features)
    return torch.mm(query_norm, video_norm.t())

def cosine_similarity_numpy(features1: np.ndarray, features2: np.ndarray) -> np.ndarray:
    """计算余弦相似度
    
    Args:
        features1: 特征矩阵1 (N, D)
        features2: 特征矩阵2 (M, D)
        
    Returns:
        np.ndarray: 相似度矩阵 (N, M)
    """
    # 归一化特征
    features1_norm = features1 / np.linalg.norm(features1, axis=1, keepdims=True)
    features2_norm = features2 / np.linalg.norm(features2, axis=1, keepdims=True)
    
    # 计算余弦相似度
    similarity = np.dot(features1_norm, features2_norm.T)
    
    return similarity

def cosine_similarity_torch(features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
    """使用PyTorch计算余弦相似度
    
    Args:
        features1: 特征矩阵1 (N, D)
        features2: 特征矩阵2 (M, D)
        
    Returns:
        torch.Tensor: 相似度矩阵 (N, M)
    """
    # 归一化特征
    features1_norm = torch.nn.functional.normalize(features1, p=2, dim=1)
    features2_norm = torch.nn.functional.normalize(features2, p=2, dim=1)
    
    # 计算余弦相似度
    similarity = torch.mm(features1_norm, features2_norm.t())
    
    return similarity

def multimodal_similarity(query_features: Dict[str, torch.Tensor], 
                         video_features: Dict[str, torch.Tensor],
                         weights: Dict[str, float] = None) -> float:
    """计算多模态相似度"""
    if weights is None:
        weights = Config.FUSION_WEIGHTS
    
    total_similarity = 0.0
    total_weight = 0.0
    
    for modality in ['text', 'visual', 'audio']:
        if modality in query_features and modality in video_features:
            if query_features[modality] is not None and video_features[modality] is not None:
                sim = cosine_similarity(
                    query_features[modality].unsqueeze(0),
                    video_features[modality].unsqueeze(0)
                ).item()
                total_similarity += weights[modality] * sim
                total_weight += weights[modality]
    
    return total_similarity / total_weight if total_weight > 0 else 0.0

def get_video_files(video_dir: str) -> List[str]:
    """获取视频文件列表"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    video_files = []
    
    for file in os.listdir(video_dir):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(file)
    
    return sorted(video_files)

def create_progress_bar(total: int, desc: str = ""):
    """创建进度条"""
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=desc)
    except ImportError:
        # 如果没有tqdm，返回一个简单的计数器
        class SimpleProgress:
            def __init__(self, total, desc=""):
                self.total = total
                self.current = 0
                self.desc = desc
            
            def update(self, n=1):
                self.current += n
                if self.current % max(1, self.total // 10) == 0:
                    print(f"{self.desc}: {self.current}/{self.total} ({100*self.current/self.total:.1f}%)")
            
            def close(self):
                print(f"{self.desc}: 完成 {self.current}/{self.total}")
        
        return SimpleProgress(total, desc)

def ensure_tensor(data: Any, device: str = None) -> torch.Tensor:
    """确保数据是tensor格式"""
    if device is None:
        device = Config.DEVICE
    
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device)
    else:
        return torch.tensor(data).to(device)

def batch_process(data_list: List, batch_size: int, process_func: callable) -> List:
    """批量处理数据
    
    Args:
        data_list: 数据列表
        batch_size: 批次大小
        process_func: 处理函数
        
    Returns:
        List: 处理结果列表
    """
    results = []
    
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i + batch_size]
        batch_results = process_func(batch)
        results.extend(batch_results)
    
    return results

def to_tensor(data, device=None):
    """将数据转换为tensor
    
    Args:
        data: 输入数据
        device: 目标设备
        
    Returns:
        torch.Tensor: 转换后的tensor
    """
    import torch
    
    if isinstance(data, torch.Tensor):
        tensor = data
    elif isinstance(data, (list, tuple)):
        tensor = torch.tensor(data)
    elif isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    else:
        tensor = torch.tensor(data)
    
    if device is not None:
        tensor = tensor.to(device)
    
    return tensor

def check_gpu_memory():
    """检查GPU内存使用情况"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_cached = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: 已分配 {memory_allocated:.2f}GB, 已缓存 {memory_cached:.2f}GB")

def manage_gpu_memory():
    """管理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def clear_gpu_cache():
    """清理GPU缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger = logging.getLogger(__name__)
        logger.info("GPU缓存已清理")