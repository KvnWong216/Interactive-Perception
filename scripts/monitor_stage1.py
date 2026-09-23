"""Record live preparation/training progress and GPU 7 telemetry once a minute."""

import fcntl
import json
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    runs = ROOT / "runs"
    with (runs / "stage1_monitor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pid = read_json(runs / "stage1_pipeline_process.json")["pid"]
        while True:
            timestamp = datetime.now(timezone(timedelta(hours=8))).isoformat(
                timespec="seconds"
            )
            preparation = read_json(ROOT / "data/prepared/stage1_full/progress.json")
            validation = read_json(
                ROOT / "data/prepared/stage1_full/validation_progress.json"
            )
            latest, training = {}, {}
            metrics = runs / "stage1_s17_gpu7/metrics.jsonl"
            if metrics.exists():
                for line in metrics.read_text().splitlines():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # A live writer may not yet have ended its last line.
                    latest.update(row)
                    if "seconds_per_update" in row:
                        training = row
            try:
                state = (
                    Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
                )
                alive = state not in {"Z", "X"}
            except FileNotFoundError:
                alive = False
            complete = bool(latest.get("training_complete"))
            phase = (
                "训练完成"
                if complete
                else "流程已退出，请查看控制台"
                if not alive
                else "正式训练"
                if training
                else "加载模型或训练前验证"
                if (runs / "stage1_s17_gpu7").exists()
                else "全量数据准备与检查"
            )
            gpu = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    "7",
                    "--query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            telemetry = {
                "timestamp": timestamp,
                "manual_seed": 17,
                "phase": phase,
                "preparation": preparation,
                "final_validation": validation,
                "latest_training": latest,
                "gpu7_memory_mib_utilization_percent_power_w_temperature_c": gpu.stdout.strip(),
            }
            with (runs / "stage1_telemetry.jsonl").open("a") as stream:
                stream.write(json.dumps(telemetry, ensure_ascii=False) + "\n")
            text = (
                f"**阶段一实时进度：{phase}**\n\n更新时间：{timestamp}；manual_seed=17；物理 GPU 7。\n\n"
                f"数据：已处理 {preparation.get('completed', 0)} / 2,000 条候选，"
                f"保留 {preparation.get('accepted', 0)} 条，排除 {preparation.get('excluded', 0)} 条。\n\n"
                f"数据准备并行进程：{preparation.get('workers', '尚未记录')}；"
                f"本次复用 {preparation.get('reused_episodes', '尚未记录')} 条，"
                f"新处理 {preparation.get('new_episodes', '尚未记录')} 条。\n\n"
                f"最终逐轨迹检查：{validation.get('completed', 0)} / {validation.get('total', preparation.get('accepted', 0))}；"
                f"检查并行数：{validation.get('workers', '尚未记录')}。\n\n"
                f"正式优化器更新：{latest.get('update', 0)} / 2,000；有效 batch=32。\n\n"
                f"最近训练 loss：{training.get('loss', '尚未记录')}；"
                f"验证 loss：{latest.get('validation_loss', '尚未记录')}；"
                f"秒/更新：{training.get('seconds_per_update', '尚未记录')}。\n\n"
                f"GPU 7 当前显存 MiB / 利用率 % / 功率 W / 温度 °C：{gpu.stdout.strip()}。\n\n"
                "[控制台日志](stage1_pipeline.console.log) · "
                "[正式训练指标](stage1_s17_gpu7/metrics.jsonl) · "
                "[性能记录](stage1_telemetry.jsonl)\n"
            )
            temporary = runs / "stage1_progress.md.tmp"
            temporary.write_text(text)
            temporary.replace(runs / "stage1_progress.md")
            if complete or not alive:
                with (ROOT / "docs/experiment_log.md").open("a") as stream:
                    stream.write(
                        f"\n{timestamp}：阶段一监控记录：{phase}，"
                        f"已记录 {latest.get('update', 0)} 次更新。详见 runs/stage1_progress.md。\n"
                    )
                break
            time.sleep(60)


if __name__ == "__main__":
    main()
