# -*- coding: utf-8 -*-
"""
多模态视频检索系统数据加载模块
"""

import os
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from config import Config
from utils import get_video_files, setup_logging

class DataLoader:
    """数据加载器类"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.video_files = []
        self.query_data = None
        
    def load_video_dataset(self) -> Dict[str, str]:
        """加载视频数据集
        
        Returns:
            Dict[str, str]: 视频ID到文件路径的映射
        """
        self.logger.info("开始加载视频数据集...")
        
        if not os.path.exists(Config.VIDEO_DATASET_PATH):
            raise FileNotFoundError(f"视频数据集路径不存在: {Config.VIDEO_DATASET_PATH}")
        
        # 获取所有视频文件
        self.video_files = get_video_files(Config.VIDEO_DATASET_PATH)
        
        if not self.video_files:
            raise ValueError("未找到任何视频文件")
        
        # 创建视频ID到路径的映射
        video_dataset = {}
        for video_file in self.video_files:
            video_id = os.path.splitext(video_file)[0]  # 去掉扩展名作为ID
            video_path = os.path.join(Config.VIDEO_DATASET_PATH, video_file)
            video_dataset[video_id] = video_path
        
        self.logger.info(f"成功加载 {len(video_dataset)} 个视频文件")
        return video_dataset
    
    def load_query_data(self) -> pd.DataFrame:
        """加载查询数据
        
        Returns:
            pd.DataFrame: 查询数据
        """
        self.logger.info("开始加载查询数据...")
        
        if not os.path.exists(Config.QUERY_CSV_PATH):
            raise FileNotFoundError(f"查询文件不存在: {Config.QUERY_CSV_PATH}")
        
        # 读取查询CSV文件
        self.query_data = pd.read_csv(Config.QUERY_CSV_PATH)
        
        # 检查必要的列
        required_columns = ['text_search', 'audio_search', 'image_search', 'raw_video']
        missing_columns = [col for col in required_columns if col not in self.query_data.columns]
        
        if missing_columns:
            raise ValueError(f"查询文件缺少必要的列: {missing_columns}")
        
        self.logger.info(f"成功加载 {len(self.query_data)} 条查询数据")
        return self.query_data
    
    def load_query_dataset_files(self) -> Dict[str, str]:
        """加载查询数据集中的音频和图像文件
        
        Returns:
            Dict[str, str]: 文件ID到路径的映射
        """
        self.logger.info("开始加载查询数据集文件...")
        
        if not os.path.exists(Config.QUERY_DATASET_PATH):
            raise FileNotFoundError(f"查询数据集路径不存在: {Config.QUERY_DATASET_PATH}")
        
        query_files = {}
        
        # 遍历查询数据集目录
        for file in os.listdir(Config.QUERY_DATASET_PATH):
            file_path = os.path.join(Config.QUERY_DATASET_PATH, file)
            if os.path.isfile(file_path):
                file_id = os.path.splitext(file)[0]  # 去掉扩展名作为ID
                query_files[file_id] = file_path
        
        self.logger.info(f"成功加载 {len(query_files)} 个查询数据集文件")
        return query_files
    
    def preprocess_queries(self, query_data: pd.DataFrame, query_files: Dict[str, str]) -> List[Dict]:
        """预处理查询数据
        
        Args:
            query_data: 查询数据DataFrame
            query_files: 查询文件映射
            
        Returns:
            List[Dict]: 预处理后的查询列表
        """
        self.logger.info("开始预处理查询数据...")
        
        processed_queries = []
        
        for idx, row in query_data.iterrows():
            query = {
                'query_id': idx,
                'text_search': row['text_search'] if pd.notna(row['text_search']) else None,
                'audio_search': row['audio_search'] if pd.notna(row['audio_search']) else None,
                'image_search': row['image_search'] if pd.notna(row['image_search']) else None,
                'raw_video': row['raw_video'] if pd.notna(row['raw_video']) else None,
                'audio_path': None,
                'image_path': None
            }
            
            # 获取音频文件路径
            if query['audio_search'] and query['audio_search'] in query_files:
                query['audio_path'] = query_files[query['audio_search']]
            
            # 获取图像文件路径
            if query['image_search'] and query['image_search'] in query_files:
                query['image_path'] = query_files[query['image_search']]
            
            processed_queries.append(query)
        
        self.logger.info(f"成功预处理 {len(processed_queries)} 条查询")
        return processed_queries
    
    def load_sample_submit(self) -> pd.DataFrame:
        """加载样本提交文件
        
        Returns:
            pd.DataFrame: 样本提交数据
        """
        if not os.path.exists(Config.SAMPLE_SUBMIT_PATH):
            self.logger.warning(f"样本提交文件不存在: {Config.SAMPLE_SUBMIT_PATH}")
            return None
        
        sample_submit = pd.read_csv(Config.SAMPLE_SUBMIT_PATH)
        self.logger.info(f"加载样本提交文件，包含 {len(sample_submit)} 条记录")
        return sample_submit
    
    def validate_data_consistency(self, video_dataset: Dict[str, str], 
                                query_data: pd.DataFrame) -> bool:
        """验证数据一致性
        
        Args:
            video_dataset: 视频数据集
            query_data: 查询数据
            
        Returns:
            bool: 数据是否一致
        """
        self.logger.info("开始验证数据一致性...")
        
        # 检查查询中引用的视频是否存在
        missing_videos = []
        for idx, row in query_data.iterrows():
            if pd.notna(row['raw_video']):
                video_id = os.path.splitext(row['raw_video'])[0]
                if video_id not in video_dataset:
                    missing_videos.append(row['raw_video'])
        
        if missing_videos:
            self.logger.warning(f"查询中引用的视频文件不存在: {missing_videos[:10]}...")
            return False
        
        self.logger.info("数据一致性验证通过")
        return True
    
    def get_data_statistics(self, query_data: pd.DataFrame) -> Dict:
        """获取数据统计信息
        
        Args:
            query_data: 查询数据
            
        Returns:
            Dict: 统计信息
        """
        stats = {
            'total_queries': len(query_data),
            'text_queries': query_data['text_search'].notna().sum(),
            'audio_queries': query_data['audio_search'].notna().sum(),
            'image_queries': query_data['image_search'].notna().sum(),
            'video_count': len(self.video_files)
        }
        
        # 计算多模态查询统计
        multimodal_count = 0
        for idx, row in query_data.iterrows():
            modalities = sum([
                pd.notna(row['text_search']),
                pd.notna(row['audio_search']),
                pd.notna(row['image_search'])
            ])
            if modalities > 1:
                multimodal_count += 1
        
        stats['multimodal_queries'] = multimodal_count
        
        self.logger.info(f"数据统计: {stats}")
        return stats
    
    def load_queries(self) -> List[Dict]:
        """加载并预处理所有查询数据
        
        Returns:
            List[Dict]: 预处理后的查询列表
        """
        # 加载查询数据
        query_data = self.load_query_data()
        
        # 加载查询数据集文件
        query_files = self.load_query_dataset_files()
        
        # 预处理查询
        processed_queries = self.preprocess_queries(query_data, query_files)
        
        return processed_queries

def main():
    """测试数据加载功能"""
    # 设置日志
    setup_logging()
    
    # 创建数据加载器
    data_loader = DataLoader()
    
    try:
        # 加载数据
        video_dataset = data_loader.load_video_dataset()
        query_data = data_loader.load_query_data()
        query_files = data_loader.load_query_dataset_files()
        
        # 预处理查询
        processed_queries = data_loader.preprocess_queries(query_data, query_files)
        
        # 验证数据一致性
        data_loader.validate_data_consistency(video_dataset, query_data)
        
        # 获取统计信息
        stats = data_loader.get_data_statistics(query_data)
        
        print("数据加载完成！")
        print(f"视频数量: {len(video_dataset)}")
        print(f"查询数量: {len(processed_queries)}")
        print(f"查询文件数量: {len(query_files)}")
        
    except Exception as e:
        logging.error(f"数据加载失败: {e}")
        raise

if __name__ == "__main__":
    main()