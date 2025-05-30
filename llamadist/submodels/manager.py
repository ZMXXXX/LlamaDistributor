"""
子模型管理器

提供对分层后子模型的统一管理接口，包括：
1. 子模型的保存和加载
2. 子模型信息管理
3. 子模型验证
4. 版本控制
"""

import os
import json
import torch
from typing import List, Dict, Optional, Any
from pathlib import Path
import shutil
from datetime import datetime

from ..partitioner.splitter import LlamaSubModel, LlamaPartitioner
from ..partitioner.strategies import PartitionStrategy


class SubModelManager:
    """
    子模型管理器
    
    提供分层模型的统一管理功能
    """
    
    def __init__(self, base_dir: str = "./models"):
        """
        初始化子模型管理器
        
        Args:
            base_dir: 基础目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 元数据文件
        self.metadata_file = self.base_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """加载元数据"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"models": {}, "version": "1.0"}
    
    def _save_metadata(self):
        """保存元数据"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def save_partitioned_model(
        self,
        submodels: List[LlamaSubModel],
        model_name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        overwrite: bool = False
    ) -> str:
        """
        保存分层模型
        
        Args:
            submodels: 子模型列表
            model_name: 模型名称
            description: 模型描述
            tags: 标签
            overwrite: 是否覆盖已存在的模型
            
        Returns:
            str: 模型ID
        """
        if model_name in self.metadata["models"] and not overwrite:
            raise ValueError(f"模型 {model_name} 已存在，使用 overwrite=True 覆盖")
        
        # 创建模型目录
        model_dir = self.base_dir / model_name
        if model_dir.exists() and overwrite:
            shutil.rmtree(model_dir)
        
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存子模型
        partitioner = LlamaPartitioner()
        partitioner.save_partitioned_models(
            submodels=submodels,
            output_dir=str(model_dir),
            save_config=True
        )
        
        # 创建模型信息
        model_info = {
            "name": model_name,
            "description": description,
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "num_partitions": len(submodels),
            "partitions": [
                {
                    "partition_idx": sm.partition_idx,
                    "layer_start": sm.layer_start,
                    "layer_end": sm.layer_end,
                    "device": sm.partition_config.device,
                    "memory_usage": sm.get_memory_usage()
                }
                for sm in submodels
            ],
            "total_memory": sum(sm.get_memory_usage() for sm in submodels),
            "model_config": submodels[0].config.to_dict() if submodels else {}
        }
        
        # 保存模型信息
        with open(model_dir / "model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # 更新元数据
        self.metadata["models"][model_name] = model_info
        self._save_metadata()
        
        print(f"分层模型 '{model_name}' 已保存到 {model_dir}")
        return model_name
    
    def load_partitioned_model(
        self,
        model_name: str,
        devices: Optional[List[str]] = None
    ) -> List[LlamaSubModel]:
        """
        加载分层模型
        
        Args:
            model_name: 模型名称
            devices: 目标设备列表
            
        Returns:
            List[LlamaSubModel]: 子模型列表
        """
        if model_name not in self.metadata["models"]:
            raise ValueError(f"模型 {model_name} 不存在")
        
        model_dir = self.base_dir / model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"模型目录不存在: {model_dir}")
        
        # 使用分层器加载
        partitioner = LlamaPartitioner()
        submodels = partitioner.load_partitioned_models(
            input_dir=str(model_dir),
            devices=devices
        )
        
        print(f"分层模型 '{model_name}' 已加载")
        return submodels
    
    def list_models(self) -> List[Dict]:
        """
        列出所有模型
        
        Returns:
            List[Dict]: 模型信息列表
        """
        return list(self.metadata["models"].values())
    
    def get_model_info(self, model_name: str) -> Dict:
        """
        获取模型信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            Dict: 模型信息
        """
        if model_name not in self.metadata["models"]:
            raise ValueError(f"模型 {model_name} 不存在")
        
        return self.metadata["models"][model_name]
    
    def delete_model(self, model_name: str) -> bool:
        """
        删除模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            bool: 是否成功删除
        """
        if model_name not in self.metadata["models"]:
            raise ValueError(f"模型 {model_name} 不存在")
        
        model_dir = self.base_dir / model_name
        if model_dir.exists():
            shutil.rmtree(model_dir)
        
        del self.metadata["models"][model_name]
        self._save_metadata()
        
        print(f"模型 '{model_name}' 已删除")
        return True
    
    def validate_model(self, model_name: str) -> Dict[str, Any]:
        """
        验证模型完整性
        
        Args:
            model_name: 模型名称
            
        Returns:
            Dict: 验证结果
        """
        if model_name not in self.metadata["models"]:
            raise ValueError(f"模型 {model_name} 不存在")
        
        model_dir = self.base_dir / model_name
        model_info = self.metadata["models"][model_name]
        
        validation_result = {
            "model_name": model_name,
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # 检查模型目录
        if not model_dir.exists():
            validation_result["valid"] = False
            validation_result["errors"].append(f"模型目录不存在: {model_dir}")
            return validation_result
        
        # 检查配置文件
        config_file = model_dir / "partition_config.json"
        if not config_file.exists():
            validation_result["valid"] = False
            validation_result["errors"].append("缺少分层配置文件")
        
        # 检查子模型文件
        expected_partitions = model_info["num_partitions"]
        for i in range(expected_partitions):
            submodel_dir = model_dir / f"submodel_{i}"
            if not submodel_dir.exists():
                validation_result["valid"] = False
                validation_result["errors"].append(f"缺少子模型目录: submodel_{i}")
                continue
            
            # 检查权重文件
            weight_file = submodel_dir / "pytorch_model.bin"
            if not weight_file.exists():
                validation_result["valid"] = False
                validation_result["errors"].append(f"缺少权重文件: submodel_{i}/pytorch_model.bin")
        
        # 尝试加载模型进行功能性验证
        try:
            submodels = self.load_partitioned_model(model_name, devices=["cpu"] * expected_partitions)
            
            # 简单的功能性测试
            partitioner = LlamaPartitioner()
            test_input = torch.randint(1, 1000, (1, 8))
            
            if not partitioner.validate_partitioned_models(submodels, test_input):
                validation_result["valid"] = False
                validation_result["errors"].append("功能性验证失败")
                
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"加载验证失败: {str(e)}")
        
        return validation_result
    
    def export_model(
        self,
        model_name: str,
        export_path: str,
        format: str = "tar"
    ) -> str:
        """
        导出模型
        
        Args:
            model_name: 模型名称
            export_path: 导出路径
            format: 导出格式 (tar, zip)
            
        Returns:
            str: 导出文件路径
        """
        if model_name not in self.metadata["models"]:
            raise ValueError(f"模型 {model_name} 不存在")
        
        model_dir = self.base_dir / model_name
        export_path = Path(export_path)
        
        if format == "tar":
            import tarfile
            with tarfile.open(export_path, "w:gz") as tar:
                tar.add(model_dir, arcname=model_name)
        elif format == "zip":
            import zipfile
            with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in model_dir.rglob("*"):
                    if file_path.is_file():
                        arcname = Path(model_name) / file_path.relative_to(model_dir)
                        zipf.write(file_path, arcname)
        else:
            raise ValueError(f"不支持的导出格式: {format}")
        
        print(f"模型 '{model_name}' 已导出到 {export_path}")
        return str(export_path)
    
    def import_model(
        self,
        import_path: str,
        model_name: Optional[str] = None,
        overwrite: bool = False
    ) -> str:
        """
        导入模型
        
        Args:
            import_path: 导入文件路径
            model_name: 模型名称（如果为None，从文件推断）
            overwrite: 是否覆盖已存在的模型
            
        Returns:
            str: 模型名称
        """
        import_path = Path(import_path)
        
        if not import_path.exists():
            raise FileNotFoundError(f"导入文件不存在: {import_path}")
        
        # 推断模型名称
        if model_name is None:
            model_name = import_path.stem
        
        if model_name in self.metadata["models"] and not overwrite:
            raise ValueError(f"模型 {model_name} 已存在，使用 overwrite=True 覆盖")
        
        # 创建临时目录
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 解压文件
            if import_path.suffix == ".gz" or import_path.name.endswith(".tar.gz"):
                import tarfile
                with tarfile.open(import_path, "r:gz") as tar:
                    tar.extractall(temp_path)
            elif import_path.suffix == ".zip":
                import zipfile
                with zipfile.ZipFile(import_path, 'r') as zipf:
                    zipf.extractall(temp_path)
            else:
                raise ValueError(f"不支持的导入格式: {import_path.suffix}")
            
            # 查找模型目录
            extracted_dirs = [d for d in temp_path.iterdir() if d.is_dir()]
            if len(extracted_dirs) != 1:
                raise ValueError("导入文件应包含单个模型目录")
            
            source_dir = extracted_dirs[0]
            target_dir = self.base_dir / model_name
            
            # 移动模型目录
            if target_dir.exists():
                shutil.rmtree(target_dir)
            
            shutil.move(str(source_dir), str(target_dir))
            
            # 加载模型信息
            model_info_file = target_dir / "model_info.json"
            if model_info_file.exists():
                with open(model_info_file, 'r') as f:
                    model_info = json.load(f)
                model_info["name"] = model_name  # 更新名称
                
                # 更新元数据
                self.metadata["models"][model_name] = model_info
                self._save_metadata()
            
        print(f"模型 '{model_name}' 已导入")
        return model_name
    
    def get_storage_info(self) -> Dict[str, Any]:
        """
        获取存储信息
        
        Returns:
            Dict: 存储信息
        """
        total_size = 0
        model_sizes = {}
        
        for model_name in self.metadata["models"]:
            model_dir = self.base_dir / model_name
            if model_dir.exists():
                size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
                model_sizes[model_name] = size
                total_size += size
        
        return {
            "base_dir": str(self.base_dir),
            "total_models": len(self.metadata["models"]),
            "total_size": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "model_sizes": model_sizes
        } 