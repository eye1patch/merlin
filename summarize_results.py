#!/usr/bin/env python3
"""
自动扫描所有checkpoint目录，对每个目录内的JSON输出进行评测，
并最终将所有checkpoint的性能指标汇总到一个Excel文件中。
"""
import argparse
import json
import logging
import pandas as pd
from pathlib import Path
from collections import defaultdict

# 评测所需的库
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# --- 配置区域 ---

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 根据您的脚本，定义哪些任务是开放式问题
# 文件名（不含.json）在此集合中的任务将被视为开放题
OPEN_ENDED_TASKS = {"Anti_CJ", "Anti_RJ", "CJS", "RJS"}

# 评测开放题所使用的指标
OPEN_ENDED_METRICS = ['rouge-l', 'bleu']

# --- 核心功能函数 ---

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="汇总所有checkpoint的Benchmark评测结果并生成Excel报告。")
    parser.add_argument("--results_dir", type=str,
                       help="包含所有checkpoint子文件夹的根目录路径。")
    parser.add_argument("--output_file", type=str, default=None,
                       help="输出Excel文件的路径 (默认: <results_dir>/benchmark_summary.xlsx)")
    return parser.parse_args()

def evaluate_json_file(json_path):
    """
    对单个JSON文件进行评测，返回一个包含分数的字典。
    此函数的核心逻辑与您提供的脚本完全一致。
    """
    task_name = json_path.stem
    is_open_ended = task_name in OPEN_ENDED_TASKS

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            submissions = json.load(f)
    except Exception as e:
        logger.error(f"无法读取或解析JSON文件: {json_path}. 错误: {e}")
        return None

    if not submissions:
        logger.warning(f"文件为空，跳过评测: {json_path}")
        return None

    # 初始化评测工具
    rouge_tool = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    bleu_smoother = SmoothingFunction()
    
    # 存储每个样本的分数
    scores = defaultdict(list)

    for item in submissions:
        model_response = item.get("model_response", "")
        correct_answer = item.get("answer", "")
        
        # model_response 或 answer 为空时跳过
        if not model_response or not correct_answer:
            continue
            
        if is_open_ended:
            # 开放题评测
            if 'rouge-l' in OPEN_ENDED_METRICS:
                rouge_scores = rouge_tool.score(correct_answer, model_response)
                scores['rouge-l'].append(rouge_scores['rougeL'].fmeasure)
            if 'bleu' in OPEN_ENDED_METRICS:
                reference = [correct_answer.split()]
                candidate = model_response.split()
                bleu = sentence_bleu(reference, candidate, smoothing_function=bleu_smoother.method1)
                scores['bleu'].append(bleu)
        else:
            # 选择题评测 (完全匹配您脚本的逻辑)
            is_correct = (model_response[0] == correct_answer[0])
            scores['accuracy'].append(1 if is_correct else 0)

    # 计算平均分
    final_metrics = {}
    if 'accuracy' in scores and scores['accuracy']:
        accuracy = (sum(scores['accuracy']) / len(scores['accuracy'])) * 100
        final_metrics[f"{task_name}_Accuracy"] = accuracy
        
    if 'rouge-l' in scores and scores['rouge-l']:
        avg_rouge = sum(scores['rouge-l']) / len(scores['rouge-l'])
        final_metrics[f"{task_name}_ROUGE-L"] = avg_rouge
        
    if 'bleu' in scores and scores['bleu']:
        avg_bleu = sum(scores['bleu']) / len(scores['bleu'])
        final_metrics[f"{task_name}_BLEU"] = avg_bleu
    
    # if "Anti-RJ" in str(json_path):
    #     import pdb; pdb.set_trace()
    
    return final_metrics

def extract_checkpoint_number(dir_name):
    """从目录名称中提取数字用于排序"""
    try:
        # 兼容 checkpoint-5600-KD-base... 等复杂名称
        return int(dir_name.split('-')[1])
    except (IndexError, ValueError):
        return 0

# --- 主程序 ---

def main():
    args = parse_args()
    results_dir = Path(args.results_dir)

    if not results_dir.is_dir():
        logger.error(f"指定的目录不存在: {results_dir}")
        return

    output_file = Path(args.output_file) if args.output_file else results_dir / "benchmark_summary.xlsx"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"开始从 {results_dir} 扫描并汇总所有checkpoint的评测结果...")

    checkpoint_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    if not checkpoint_dirs:
        logger.error(f"在 {results_dir} 中未找到任何checkpoint子目录。")
        return
        
    logger.info(f"找到 {len(checkpoint_dirs)} 个checkpoint目录。")

    all_checkpoints_summary = {}

    for checkpoint_dir in sorted(checkpoint_dirs, key=lambda p: extract_checkpoint_number(p.name)):
        checkpoint_name = checkpoint_dir.name
        logger.info(f"--- 正在处理Checkpoint: {checkpoint_name} ---")

        json_files = list(checkpoint_dir.glob("*.json"))
        if not json_files:
            logger.warning(f"在 {checkpoint_dir} 中未找到任何JSON文件，跳过。")
            continue

        checkpoint_metrics = {}
        for json_file in json_files:
            logger.info(f"  - 评测任务: {json_file.name}")
            metrics = evaluate_json_file(json_file)
            if metrics:
                checkpoint_metrics.update(metrics)
        
        if checkpoint_metrics:
            all_checkpoints_summary[checkpoint_name] = checkpoint_metrics
        else:
            logger.warning(f"Checkpoint {checkpoint_name} 未能生成任何评测结果。")

    if not all_checkpoints_summary:
        logger.error("所有目录均未能生成有效的评测结果。")
        return

    # 转换为DataFrame进行处理
    df = pd.DataFrame.from_dict(all_checkpoints_summary, orient='index')
    df.index.name = "Checkpoint"

    # 计算总平均分（所有指标的平均值）
    # 注意：直接平均不同类型的指标（如Accuracy和ROUGE）仅为参考
    numeric_cols = df.select_dtypes(include=['number']).columns
    if not numeric_cols.empty:
        df['Overall_Avg_Score'] = df[numeric_cols].mean(axis=1)
        # 将总分移动到第一列
        cols = ['Overall_Avg_Score'] + [col for col in df.columns if col != 'Overall_Avg_Score']
        df = df[cols]
    
    # 保存到Excel
    try:
        df.to_excel(output_file, sheet_name="Benchmark Summary", float_format="%.4f")
        logger.info(f"✅ 评测汇总完成！Excel报告已保存至: {output_file}")

        # 打印最终总结
        print("\n" + "="*50)
        print("📊 评测结果汇总")
        print("="*50)
        print(f"总共处理了 {len(df)} 个有效的checkpoint。")
        if "Overall_Avg_Score" in df.columns:
            best_checkpoint = df["Overall_Avg_Score"].idxmax()
            best_score = df["Overall_Avg_Score"].max()
            print(f"🏆 综合表现最佳的Checkpoint: {best_checkpoint} (总平均分: {best_score:.4f})")
        print(f"详细报告已生成: {output_file}")
        print("="*50)

    except Exception as e:
        logger.error(f"写入Excel文件失败: {e}")

if __name__ == "__main__":
    main()