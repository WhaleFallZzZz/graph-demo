#!/usr/bin/env python3
import os
import time
import random
from pathlib import Path

# 目标输出目录
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_PATH = OUT_DIR / "perf_curve.png"

# 导入构建器与模块工厂
from kg_manager import builder, DOCUMENT_CONFIG
from factories import LlamaModuleFactory

# 确保 LlamaIndex 基础模块可用（不依赖在线 LLM）
if builder.modules is None:
    mods = LlamaModuleFactory.get_modules()
    if not mods:
        print("ERROR: LlamaIndex 模块不可用，无法进行分块与性能测试")
        raise SystemExit(1)
    builder.modules = mods

Doc = builder.modules["Document"]

# 构造模拟医学文本
keywords = ["近视", "远视", "散光", "弱视", "斜视", "视力", "眼轴", "角膜", "阿托品", "OK镜"]
symptoms = ["视物模糊", "眼痛", "畏光", "视力下降"]
treatments = ["低浓度阿托品", "角膜塑形镜(OK镜)", "RGP镜片"]

random.seed(42)


def make_text(min_lines=20, max_lines=30):
    lines = []
    for _ in range(random.randint(min_lines, max_lines)):
        line = f"患者{random.choice(['张三','李四','王五'])}，主诉{random.choice(symptoms)}。"
        line += f"检查显示{random.choice(['眼轴长度','眼压','屈光度'])}为{random.uniform(22.0,26.0):.2f}。"
        line += f"诊断为{random.choice(keywords)}，建议{random.choice(treatments)}。"
        lines.append(line)
    return "\n".join(lines)


def measure_chunking(raw_docs_count: int):
    """测量分块阶段性能（不依赖LLM）"""
    raw_docs = [Doc(text=make_text(), metadata={"file_name": f"mock_{i}.txt"}) for i in range(raw_docs_count)]
    t0 = time.time()
    total_chunks = 0
    for rd in raw_docs:
        chunks = builder._chunk_document(rd)
        total_chunks += len(chunks)
    elapsed_ms = (time.time() - t0) * 1000.0
    return {"raw_docs": raw_docs_count, "chunks": total_chunks, "time_ms": elapsed_ms}


def simulate_processing(chunks_count: int):
    """模拟批处理循环的开销（替代实际 index.insert_nodes），用于观察 batch=5 的拐点"""
    docs = [Doc(text="示例分块文本" * 50, metadata={"file_name": f"chunk_{i}.txt"}) for i in range(chunks_count)]
    batch_size = DOCUMENT_CONFIG.get("batch_size", 5)
    start_pct, end_pct = 30, 90
    t0 = time.time()
    total_docs = len(docs)

    def fake_insert(batch):
        # 小型CPU工作负载以模拟处理开销
        s = 0
        for i in range(1000):
            s += i * i
        return s

    for i in range(0, total_docs, batch_size):
        batch = docs[i : i + batch_size]
        # 进度计算（与生产代码一致）
        _ = start_pct + ((min(i + batch_size, total_docs) / total_docs) * (end_pct - start_pct))
        _ = fake_insert(batch)

    elapsed_ms = (time.time() - t0) * 1000.0
    return {"chunks": chunks_count, "time_ms": elapsed_ms, "batch_size": batch_size}


def main():
    scenarios = [1, 5, 10, 20]
    chunking_results = []
    processing_results = []

    for n in scenarios:
        chunking_results.append(measure_chunking(n))
        processing_results.append(simulate_processing(n))

    print("ChunkingResults:")
    for r in chunking_results:
        avg = (r["time_ms"] / max(1, r["chunks"])) if r["chunks"] else 0.0
        print(f"  raw={r['raw_docs']}, chunks={r['chunks']}, time={r['time_ms']:.2f}ms, avg_per_chunk={avg:.2f}ms")

    print("ProcessingResults:")
    for r in processing_results:
        avg = r["time_ms"] / max(1, r["chunks"])
        print(f"  chunks={r['chunks']}, time={r['time_ms']:.2f}ms, avg_per_chunk={avg:.2f}ms (batch={r['batch_size']})")

    # 生成性能变化曲线图（若可用）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = scenarios
        ys_chunk = [r["time_ms"] for r in chunking_results]
        ys_proc = [r["time_ms"] for r in processing_results]
        plt.figure(figsize=(7, 4.5))
        plt.plot(xs, ys_chunk, marker="o", label="Chunking Time (ms)")
        plt.plot(xs, ys_proc, marker="s", label="Processing Overhead (ms)")
        plt.xlabel("文档块数量")
        plt.ylabel("耗时 (毫秒)")
        plt.title("文档处理性能变化曲线")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(PLOT_PATH.as_posix(), dpi=140, bbox_inches="tight")
        print(f"CurveSaved: {PLOT_PATH.as_posix()}")
    except Exception as e:
        print(f"CurvePlotFailed: {e}")


if __name__ == "__main__":
    main()

