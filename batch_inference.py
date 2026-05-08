#!/usr/bin/env python3
"""
批量执行多个checkpoint的推理和评估
"""
import os
import yaml
import argparse
import subprocess
import pathlib
from tqdm import tqdm
import logging
import uuid

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="批量执行多个checkpoint的推理")
    parser.add_argument("--config", type=str, default="configs/batch_inference.yaml", help="批量推理配置文件")
    parser.add_argument("--resume", action="store_true", help="跳过已完成的checkpoint")
    return parser.parse_args()

def get_checkpoint_number(checkpoint_path):
    """从checkpoint路径提取数字用于排序"""
    return int(checkpoint_path.stem.split('-')[1])

def run_inference(checkpoint_path, config, inference_config_path):
    """为单个checkpoint运行推理"""
    checkpoint_name = checkpoint_path.name
    output_dir = pathlib.Path(config["base_output_dir"]).joinpath(checkpoint_name)

    # 创建临时配置文件
    temp_config = config["inference_config"].copy()
    temp_config["checkpoint"] = str(checkpoint_path)
    temp_config["output_dir"] = str(output_dir)

    temp_config_path = f"configs/temp_inference_{checkpoint_name}_{str(uuid.uuid4())}.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(temp_config, f)

    try:
        # 运行推理
        cmd = f"accelerate launch --multi_gpu --mixed_precision=bf16 inference.py --config {temp_config_path}"
        logger.info(f"开始推理 {checkpoint_name}: {cmd}")

        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"推理失败 {checkpoint_name}: {result.stderr}")
            return False, str(output_dir)
        else:
            logger.info(f"推理完成 {checkpoint_name}")
            return True, str(output_dir)

    finally:
        # 清理临时配置文件
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

def run_evaluation(output_dir, config):
    """为推理结果运行评估"""
    logger.info(f"开始评估 {output_dir}")

    # 构建评估命令
    metrics = " ".join(config["evaluation"]["open_ended_metrics"])
    cmd = f"python benchmark_eval.py --submission_dir {output_dir} --open_ended_metrics {metrics}"

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"评估失败 {output_dir}: {result.stderr}")
        return False
    else:
        logger.info(f"评估完成 {output_dir}")
        return True

def main():
    args = parse_args()

    # 加载配置
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # 获取所有checkpoint
    checkpoint_base_dir = pathlib.Path(config["checkpoint_base_dir"])
    checkpoints = sorted([p for p in checkpoint_base_dir.glob("checkpoint-*") if p.is_dir()],
                        key=get_checkpoint_number)

    if not checkpoints:
        logger.error(f"未找到checkpoint在目录: {checkpoint_base_dir}")
        return

    logger.info(f"找到 {len(checkpoints)} 个checkpoint")

    results = []
    failed_checkpoints = []

    # 遍历处理每个checkpoint
    for checkpoint_path in tqdm(checkpoints, desc="处理checkpoint"):
        checkpoint_name = checkpoint_path.name
        output_dir = pathlib.Path(config["base_output_dir"]).joinpath(checkpoint_name)

        # 检查是否需要跳过
        if args.resume and output_dir.exists():
            # 检查是否有评估结果文件
            csv_files = list(output_dir.glob("*.csv"))
            if csv_files:
                logger.info(f"跳过已完成的checkpoint: {checkpoint_name}")
                results.append({
                    "checkpoint": checkpoint_name,
                    "status": "skipped",
                    "output_dir": str(output_dir)
                })
                continue

        logger.info(f"处理checkpoint: {checkpoint_name}")

        # 运行推理
        inference_success, result_output_dir = run_inference(checkpoint_path, config, args.config)

        if not inference_success:
            failed_checkpoints.append(checkpoint_name)
            results.append({
                "checkpoint": checkpoint_name,
                "status": "inference_failed",
                "output_dir": result_output_dir
            })
            continue

        # 运行评估
        evaluation_success = run_evaluation(result_output_dir, config)

        if not evaluation_success:
            failed_checkpoints.append(f"{checkpoint_name} (evaluation)")
            results.append({
                "checkpoint": checkpoint_name,
                "status": "evaluation_failed",
                "output_dir": result_output_dir
            })
        else:
            results.append({
                "checkpoint": checkpoint_name,
                "status": "success",
                "output_dir": result_output_dir
            })

    # 输出总结
    successful = len([r for r in results if r["status"] == "success"])
    skipped = len([r for r in results if r["status"] == "skipped"])
    failed = len([r for r in results if "failed" in r["status"]])

    logger.info(f"\n处理完成!")
    logger.info(f"成功: {successful}")
    logger.info(f"跳过: {skipped}")
    logger.info(f"失败: {failed}")

    if failed_checkpoints:
        logger.info(f"失败的checkpoint: {', '.join(failed_checkpoints)}")

    # 保存处理结果
    results_file = pathlib.Path(config["base_output_dir"]) / config["version"] / pathlib.Path(config["inference_config"]["dataset_path"]).name / "batch_processing_results.yaml"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        yaml.dump({
            "summary": {
                "total": len(checkpoints),
                "successful": successful,
                "skipped": skipped,
                "failed": failed
            },
            "details": results
        }, f)

    logger.info(f"处理结果保存至: {results_file}")

if __name__ == "__main__":
    main()