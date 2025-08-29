# -*- coding: utf-8 -*-
"""
多模态视频检索系统主程序
"""

import os
import time
import argparse
import logging
from typing import Dict, List, Optional

import torch
import pandas as pd

from config import Config
from utils import setup_logging, manage_gpu_memory
from data_loader import DataLoader
from feature_extractor import FeatureExtractor
from retrieval_engine import RetrievalEngine

class MultiModalVideoRetrieval:
    """多模态视频检索系统主类"""
    
    def __init__(self, config_override: Optional[Dict] = None):
        """初始化检索系统
        
        Args:
            config_override: 配置覆盖参数
        """
        # 设置日志
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 应用配置覆盖
        if config_override:
            for key, value in config_override.items():
                if hasattr(Config, key):
                    setattr(Config, key, value)
        
        # 创建必要目录
        Config.create_dirs()
        
        # 初始化组件
        self.data_loader = DataLoader()
        self.feature_extractor = FeatureExtractor()
        self.retrieval_engine = RetrievalEngine()
        
        # 数据存储
        self.video_dataset = {}
        self.query_data = None
        self.processed_queries = []
        self.video_features = {}
        self.query_features = []
        
        self.logger.info("多模态视频检索系统初始化完成")
    
    def load_data(self):
        """加载所有数据"""
        self.logger.info("开始加载数据...")
        
        try:
            # 加载视频数据集
            self.video_dataset = self.data_loader.load_video_dataset()
            
            # 加载查询数据
            self.query_data = self.data_loader.load_query_data()
            
            # 加载查询数据集文件
            query_files = self.data_loader.load_query_dataset_files()
            
            # 预处理查询
            self.processed_queries = self.data_loader.preprocess_queries(
                self.query_data, query_files
            )
            
            # 验证数据一致性
            self.data_loader.validate_data_consistency(
                self.video_dataset, self.query_data
            )
            
            # 获取数据统计
            stats = self.data_loader.get_data_statistics(self.query_data)
            self.logger.info(f"数据加载完成: {stats}")
            
        except Exception as e:
            self.logger.error(f"数据加载失败: {e}")
            raise
    
    def extract_features(self, use_cache: bool = True):
        """提取所有特征
        
        Args:
            use_cache: 是否使用特征缓存
        """
        self.logger.info("开始特征提取...")
        
        cache_dir = Config.FEATURE_CACHE_DIR if use_cache else None
        
        try:
            # 提取视频特征
            self.logger.info("提取视频特征...")
            video_paths = list(self.video_dataset.values())
            self.video_features = self.feature_extractor.batch_extract_video_features(
                video_paths, cache_dir
            )
            
            # 提取查询特征
            self.logger.info("提取查询特征...")
            self.query_features = self.feature_extractor.batch_extract_query_features(
                self.processed_queries, cache_dir
            )
            
            self.logger.info(f"特征提取完成 - 视频: {len(self.video_features)}, 查询: {len(self.query_features)}")
            
            # 清理GPU内存
            manage_gpu_memory()
            
        except Exception as e:
            self.logger.error(f"特征提取失败: {e}")
            raise
    
    def perform_retrieval(self, method: str = 'standard') -> List[List[tuple]]:
        """执行视频检索
        
        Args:
            method: 检索方法 ('standard', 'reranking', 'adaptive')
            
        Returns:
            List[List[tuple]]: 检索结果
        """
        self.logger.info(f"开始执行检索，方法: {method}")
        
        try:
            # 加载特征到检索引擎
            self.retrieval_engine.load_video_features(self.video_features)
            self.retrieval_engine.load_query_features(self.query_features)
            
            # 根据方法执行检索
            if method == 'standard':
                results = self.retrieval_engine.retrieve_batch_queries(top_k=1)
            elif method == 'reranking':
                results = self.retrieval_engine.retrieve_with_reranking(
                    self.query_features, initial_top_k=10, final_top_k=1
                )
            elif method == 'adaptive':
                results = self.retrieval_engine.adaptive_retrieval(self.processed_queries)
            else:
                raise ValueError(f"未知的检索方法: {method}")
            
            self.logger.info(f"检索完成，共处理 {len(results)} 个查询")
            
            # 获取检索统计
            stats = self.retrieval_engine.get_retrieval_statistics(results)
            self.logger.info(f"检索统计: {stats}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"检索执行失败: {e}")
            raise
    
    def generate_submission(self, retrieval_results: List[List[tuple]], 
                          output_path: Optional[str] = None):
        """生成提交文件
        
        Args:
            retrieval_results: 检索结果
            output_path: 输出文件路径
        """
        if output_path is None:
            output_path = Config.SUBMISSION_PATH
        
        try:
            # 加载sample_submit.csv以获取正确的行数
            sample_submit = self.data_loader.load_sample_submit()
            if sample_submit is not None:
                expected_rows = len(sample_submit)
                self.logger.info(f"根据sample_submit.csv生成 {expected_rows} 行结果")
                self.retrieval_engine.generate_submission_with_expected_rows(
                    retrieval_results, output_path, expected_rows
                )
            else:
                self.retrieval_engine.generate_submission(retrieval_results, output_path)
            
            self.logger.info(f"提交文件已生成: {output_path}")
            
        except Exception as e:
            self.logger.error(f"提交文件生成失败: {e}")
            raise
    
    def evaluate_performance(self, retrieval_results: List[List[tuple]]) -> Dict[str, float]:
        """评估系统性能
        
        Args:
            retrieval_results: 检索结果
            
        Returns:
            Dict[str, float]: 评估指标
        """
        try:
            # 从查询数据中提取真实标签
            ground_truth = []
            for query in self.processed_queries:
                if query.get('raw_video'):
                    ground_truth.append(query['raw_video'])
                else:
                    ground_truth.append('')  # 默认值
            
            # 执行评估
            metrics = self.retrieval_engine.evaluate_retrieval(
                retrieval_results, ground_truth
            )
            
            self.logger.info(f"性能评估完成: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"性能评估失败: {e}")
            return {}
    
    def run_full_pipeline(self, retrieval_method: str = 'standard', 
                         use_cache: bool = True, 
                         evaluate: bool = True) -> Dict:
        """运行完整的检索流程
        
        Args:
            retrieval_method: 检索方法
            use_cache: 是否使用缓存
            evaluate: 是否进行评估
            
        Returns:
            Dict: 运行结果和统计信息
        """
        start_time = time.time()
        
        try:
            # 1. 加载数据
            self.load_data()
            
            # 2. 提取特征
            self.extract_features(use_cache=use_cache)
            
            # 3. 执行检索
            retrieval_results = self.perform_retrieval(method=retrieval_method)
            
            # 4. 生成提交文件
            self.generate_submission(retrieval_results)
            
            # 5. 评估性能（可选）
            metrics = {}
            if evaluate:
                metrics = self.evaluate_performance(retrieval_results)
            
            # 计算总耗时
            total_time = time.time() - start_time
            
            # 汇总结果
            result_summary = {
                'status': 'success',
                'total_time': total_time,
                'video_count': len(self.video_dataset),
                'query_count': len(self.processed_queries),
                'retrieval_method': retrieval_method,
                'metrics': metrics,
                'submission_path': Config.SUBMISSION_PATH
            }
            
            self.logger.info(f"完整流程执行完成: {result_summary}")
            return result_summary
            
        except Exception as e:
            self.logger.error(f"完整流程执行失败: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'total_time': time.time() - start_time
            }
    
    def optimize_system(self, validation_split: float = 0.2):
        """优化系统参数
        
        Args:
            validation_split: 验证集比例
        """
        self.logger.info("开始系统参数优化...")
        
        try:
            # 分割验证集
            val_size = int(len(self.processed_queries) * validation_split)
            val_queries = self.processed_queries[:val_size]
            val_ground_truth = [q.get('raw_video', '') for q in val_queries]
            
            # 优化检索参数
            optimal_params = self.retrieval_engine.optimize_retrieval_parameters(
                val_queries, val_ground_truth
            )
            
            # 应用优化参数
            Config.TEXT_WEIGHT = optimal_params.get('text', Config.TEXT_WEIGHT)
            Config.AUDIO_WEIGHT = optimal_params.get('audio', Config.AUDIO_WEIGHT)
            Config.IMAGE_WEIGHT = optimal_params.get('image', Config.IMAGE_WEIGHT)
            
            self.logger.info(f"参数优化完成: {optimal_params}")
            
        except Exception as e:
            self.logger.error(f"参数优化失败: {e}")
    
    def get_system_info(self) -> Dict:
        """获取系统信息
        
        Returns:
            Dict: 系统信息
        """
        return {
            'device': str(Config.DEVICE),
            'cuda_available': torch.cuda.is_available(),
            'gpu_memory': torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
            'video_dataset_path': Config.VIDEO_DATASET_PATH,
            'query_csv_path': Config.QUERY_CSV_PATH,
            'feature_cache_dir': Config.FEATURE_CACHE_DIR,
            'models': {
                'clip': Config.CLIP_MODEL_NAME,
                'wav2vec2': Config.WAV2VEC2_MODEL_NAME,
                'bert': Config.BERT_MODEL_NAME
            }
        }

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='多模态视频检索系统')
    
    parser.add_argument('--method', type=str, default='standard',
                       choices=['standard', 'reranking', 'adaptive'],
                       help='检索方法')
    
    parser.add_argument('--no-cache', action='store_true',
                       help='不使用特征缓存')
    
    parser.add_argument('--no-eval', action='store_true',
                       help='不进行性能评估')
    
    parser.add_argument('--optimize', action='store_true',
                       help='执行参数优化')
    
    parser.add_argument('--output', type=str,
                       help='输出文件路径')
    
    parser.add_argument('--config', type=str,
                       help='配置文件路径')
    
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    
    return parser.parse_args()

def load_config_from_file(config_path: str) -> Dict:
    """从文件加载配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        Dict: 配置字典
    """
    import json
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logging.error(f"配置文件加载失败: {e}")
        return {}

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 加载配置覆盖
    config_override = {}
    if args.config:
        config_override = load_config_from_file(args.config)
    
    if args.output:
        config_override['SUBMISSION_PATH'] = args.output
    
    try:
        # 创建检索系统
        retrieval_system = MultiModalVideoRetrieval(config_override)
        
        # 打印系统信息
        system_info = retrieval_system.get_system_info()
        print("=" * 60)
        print("多模态视频检索系统")
        print("=" * 60)
        print(f"设备: {system_info['device']}")
        print(f"CUDA可用: {system_info['cuda_available']}")
        print(f"视频数据集: {system_info['video_dataset_path']}")
        print(f"查询文件: {system_info['query_csv_path']}")
        print("=" * 60)
        
        # 执行参数优化（可选）
        if args.optimize:
            retrieval_system.load_data()
            retrieval_system.extract_features(use_cache=not args.no_cache)
            retrieval_system.optimize_system()
        
        # 运行完整流程
        result = retrieval_system.run_full_pipeline(
            retrieval_method=args.method,
            use_cache=not args.no_cache,
            evaluate=not args.no_eval
        )
        
        # 打印结果
        print("\n" + "=" * 60)
        print("执行结果")
        print("=" * 60)
        print(f"状态: {result['status']}")
        print(f"总耗时: {result.get('total_time', 0):.2f}秒")
        print(f"视频数量: {result.get('video_count', 0)}")
        print(f"查询数量: {result.get('query_count', 0)}")
        print(f"检索方法: {result.get('retrieval_method', 'unknown')}")
        
        if result.get('metrics'):
            metrics = result['metrics']
            print(f"准确率: {metrics.get('accuracy', 0):.4f}")
            print(f"正确数量: {metrics.get('correct', 0)}")
        
        print(f"提交文件: {result.get('submission_path', 'unknown')}")
        print("=" * 60)
        
        if result['status'] == 'success':
            print("\n✅ 检索系统执行成功！")
        else:
            print(f"\n❌ 检索系统执行失败: {result.get('error', 'unknown')}")
            return 1
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断执行")
        return 1
    except Exception as e:
        print(f"\n❌ 系统执行失败: {e}")
        logging.exception("系统执行异常")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())