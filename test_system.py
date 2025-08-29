#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态视频检索系统测试脚本
"""

import os
import torch
import pandas as pd
from pathlib import Path

from config import Config
from utils import setup_logging
from data_loader import DataLoader
from feature_extractor import FeatureExtractor
from retrieval_engine import RetrievalEngine
from main import MultiModalVideoRetrieval

def test_config():
    """测试配置模块"""
    print("\n=== 测试配置模块 ===")
    print(f"设备: {Config.DEVICE}")
    print(f"视频数据路径: {Config.VIDEO_DATASET_PATH}")
    print(f"查询数据路径: {Config.QUERY_CSV_PATH}")
    print(f"特征缓存路径: {Config.FEATURE_CACHE_PATH}")
    
    # 创建必要的目录
    Config.create_dirs()
    print("目录创建完成")
    
    return True

def test_data_loader():
    """测试数据加载模块"""
    print("\n=== 测试数据加载模块 ===")
    
    # 创建测试查询数据
    test_query_file = "test_query.csv"
    test_data = {
        'query_id': ['Q001', 'Q002', 'Q003'],
        'text': ['测试文本1', '测试文本2', '测试文本3'],
        'audio_path': ['audio1.wav', 'audio2.wav', 'audio3.wav'],
        'image_path': ['image1.jpg', 'image2.jpg', 'image3.jpg']
    }
    
    df = pd.DataFrame(test_data)
    df.to_csv(test_query_file, index=False)
    
    try:
        data_loader = DataLoader()
        queries = data_loader.load_queries()
        print(f"成功加载 {len(queries)} 个查询")
        
        # 清理测试文件
        os.remove(test_query_file)
        
        return True
    except Exception as e:
        print(f"数据加载测试失败: {e}")
        if os.path.exists(test_query_file):
            os.remove(test_query_file)
        return False

def test_feature_extractor():
    """测试特征提取模块"""
    print("\n=== 测试特征提取模块 ===")
    
    try:
        extractor = FeatureExtractor()
        print("特征提取器初始化成功")
        
        # 测试文本特征提取
        text = "这是一个测试文本"
        text_features = extractor.extract_text_features(text)
        print(f"文本特征维度: {text_features.shape}")
        
        return True
    except Exception as e:
        print(f"特征提取测试失败: {e}")
        return False

def test_retrieval_engine():
    """测试检索引擎模块"""
    print("\n=== 测试检索引擎模块 ===")
    
    try:
        engine = RetrievalEngine()
        print("检索引擎初始化成功")
        
        # 创建模拟特征数据
        video_features = {
            'video1': torch.randn(512),
            'video2': torch.randn(512),
            'video3': torch.randn(512)
        }
        
        engine.load_video_features(video_features)
        print("视频特征加载成功")
        
        # 测试单个查询检索
        query_feature = torch.randn(512)
        results = engine.retrieve_single_query(query_feature, top_k=2)
        print(f"检索结果: {len(results)} 个")
        
        return True
    except Exception as e:
        print(f"检索引擎测试失败: {e}")
        return False

def test_gpu_availability():
    """测试GPU可用性"""
    print("\n=== 测试GPU可用性 ===")
    
    if torch.cuda.is_available():
        print(f"GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
        
        # 测试GPU计算
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.mm(x, y)
        print("GPU计算测试通过")
        
        return True
    else:
        print("GPU不可用，将使用CPU")
        return False

def run_all_tests():
    """运行所有测试"""
    print("开始系统功能测试...")
    
    # 设置日志
    setup_logging()
    
    tests = [
        ("配置模块", test_config),
        ("数据加载模块", test_data_loader),
        ("特征提取模块", test_feature_extractor),
        ("检索引擎模块", test_retrieval_engine),
        ("GPU可用性", test_gpu_availability)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
            status = "✓ 通过" if result else "✗ 失败"
            print(f"{test_name}: {status}")
        except Exception as e:
            results[test_name] = False
            print(f"{test_name}: ✗ 失败 - {e}")
    
    print("\n=== 测试总结 ===")
    passed = sum(results.values())
    total = len(results)
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！系统功能正常")
    else:
        print("⚠️ 部分测试失败，请检查相关模块")
    
    return results

if __name__ == "__main__":
    run_all_tests()