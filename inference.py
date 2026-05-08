# inference_dataset.py
import os
import json
import torch
import yaml
import argparse
from types import SimpleNamespace
from tqdm import tqdm
import datasets
import pathlib

from models.emmllm import EMMLLM
from models.em_encoder.em_encoder import EMEncoder, init_sit
from models.llm import EMLanguageModel
from models.projector import EMProjector
from utils.utils import transform_downstream
from data.collator import EMInferCollator

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from accelerate import Accelerator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/inference.yaml")
    return parser.parse_args()


def build_model(config):
    encoder_args = SimpleNamespace(**config["em_encoder"])
    sit_model = init_sit(encoder_args)
    em_encoder = EMEncoder(sit_model)
    llm = EMLanguageModel(
        model_name_or_path=config["model_name"],
        lora_config=config["lora"]
    )
    projector = EMProjector(
        em_hidden_size=config["em_hidden_size"],
        llm_hidden_size=llm.model.config.hidden_size,
        projector_dims=config.get("projector_dims", 1536),
        activation="gelu"
    )
    model = EMMLLM(
        em_encoder=em_encoder,
        projector=projector,
        language_model=llm,
        pool_to_span="avg",
        freeze_em_encoder=False,
        freeze_llm=False
    )
    return model, llm.tokenizer


def main():
    args = parse_args()
    
    accelerator = Accelerator(mixed_precision="bf16")

    # --- 加载配置 ---
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # --- 构建模型 ---
    model, tokenizer = build_model(config)

    # --- 加载 checkpoint ---
    ckpt_path = os.path.join(config["checkpoint"], "pytorch_model.bin")
    print(f"Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    
    if config["bf16"]:
        model = model.to(torch.bfloat16)
        
    model = accelerator.prepare(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    inference_dataset_dict = datasets.load_from_disk(config["dataset_path"])
    for sub_dir in pathlib.Path(config["dataset_path"]).glob("*"):
        if not sub_dir.is_dir():
            continue
        # --- 加载 dataset ---
        inference_dataset = inference_dataset_dict[sub_dir.name]
        
        with accelerator.main_process_first():
            dataset = inference_dataset.map(
                lambda x: transform_downstream(
                    x,
                    [8, 1],
                    "samples"
                ),
                batched=True,
                batch_size=1,
                num_proc=config["num_proc"]
            )

        # --- collator 准备 ---
        collator = EMInferCollator(
            tokenizer=tokenizer,
            iq_max_seq_len=config["em_encoder"]["max_seq_len"],
            max_seq_len=config["max_seq_len"],
            patch_size=config["em_encoder"]["patch_size"],
        )

        dataloader = DataLoader(
            dataset,
            batch_size=config["batch_size"],  # 可以改成你想要的 batch size
            shuffle=False,
            collate_fn=collator,
            num_workers=config["num_proc"],  # 根据显存和 CPU 核数调整
            pin_memory=True
        )
        
        dataloader = accelerator.prepare(dataloader)
        
        all_outputs = []
        all_ids = []
        all_prompts = []
        all_answers = []
        all_snrs = []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader):
                if batch is None:
                    continue
                # # 移动到 device
                # for k, v in batch.items():
                #     if torch.is_tensor(v):
                #         batch[k] = v.to(device)
                #     elif k == "sample_ids_seq":
                #         batch[k] = [t.to(device) for t in v]

                with accelerator.autocast():
                    outputs = accelerator.unwrap_model(model).generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        samples=batch.get("samples"),
                        patch_positions=batch.get("patch_positions"),
                        sample_ids_seq=batch.get("sample_ids_seq"),
                        patch_batch_idx=batch.get("patch_batch_idx"),
                        max_new_tokens=config["max_new_tokens"],
                        temperature=config["temperature"],
                        top_p=config["top_p"],
                        do_sample=config["do_sample"]
                    )

                all_outputs.extend(accelerator.gather_for_metrics(outputs))
                all_ids.extend(accelerator.gather_for_metrics(batch["id"]))
                all_prompts.extend(accelerator.gather_for_metrics(batch["human_prompt"]))
                all_answers.extend(accelerator.gather_for_metrics(batch["answer"]))
                all_snrs.extend(accelerator.gather_for_metrics(batch["snr"]))
        
        
        
        if accelerator.is_main_process:
            save_file = pathlib.Path(config["output_dir"]).joinpath(sub_dir.stem + ".json")
            if not save_file.parent.exists():
                save_file.parent.mkdir(parents=True)
            with open(save_file, "w", encoding="utf-8") as f:
                results = []
                for id_, prompt, answer, output, snr in zip(all_ids, all_prompts, all_answers, all_outputs, all_snrs):
                    results.append({
                        "id": id_,
                        "human_prompt": prompt,
                        "model_response": output,
                        "answer": answer,
                        "snr": snr
                    })
                json.dump(results, f, ensure_ascii=False)

    accelerator.print(f"✅ Inference finished")


if __name__ == "__main__":
    main()