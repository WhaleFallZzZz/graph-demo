import os
import time
try:
    import psutil
except ImportError:
    psutil = None
import threading
from pathlib import Path

# 设置环境变量 (必须在导入 kg_manager 之前)
benchmark_dir = str(Path(__file__).parent / "data" / "benchmark")
os.environ["DOCUMENT_PATH"] = benchmark_dir
os.environ["DOCUMENT_NUM_WORKERS"] = "8" 
os.environ["DOC_CHUNK_SIZE"] = "500"
# 禁用增量处理以确保每次都运行
os.environ["INCREMENTAL_PROCESSING"] = "false" 

# 导入 kg_manager (会自动读取上面的环境变量)
from kg_manager import builder, DOCUMENT_CONFIG

def monitor_resources(stop_event):
    cpu_usage = []
    mem_usage = []
    while not stop_event.is_set():
        try:
            if psutil:
                cpu = psutil.cpu_percent(interval=0.5)
                cpu_usage.append(cpu)
                
                mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                mem_usage.append(mem)
            else:
                 # Fallback using resource module (Memory only)
                 import sys
                 raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                 if sys.platform == 'darwin':
                     # macOS: bytes
                     mem = raw / (1024 * 1024)
                 else:
                     # Linux: kilobytes
                     mem = raw / 1024
                 
                 mem_usage.append(mem)
                 time.sleep(0.5)
        except:
            pass
    
    avg_cpu = sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0
    max_cpu = max(cpu_usage) if cpu_usage else 0
    max_mem = max(mem_usage) if mem_usage else 0
    
    print(f"\nResource Usage Metrics:")
    if psutil:
        print(f"  Avg CPU Usage: {avg_cpu:.1f}%")
        print(f"  Max CPU Usage: {max_cpu:.1f}%")
    else:
        print("  CPU Usage: (psutil not installed)")
    print(f"  Max Memory Usage: {max_mem:.1f} MB")

def run_benchmark():
    print("="*50)
    print("Starting Performance Benchmark")
    print(f"Target Directory: {DOCUMENT_CONFIG['path']}")
    print("="*50)
    
    # 启动资源监控
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_resources, args=(stop_event,))
    monitor_thread.start()
    
    t0 = time.time()
    
    try:
        # 1. 加载文档
        print("Phase 1: Loading Documents...")
        documents = builder.load_documents()
        load_time = time.time() - t0
        print(f"-> Loaded {len(documents)} chunks in {load_time:.2f}s")
        
        # 2. 构建图谱
        if documents:
            print("Phase 2: Building Knowledge Graph (Parallel)...")
            t1 = time.time()
            builder.build_knowledge_graph(documents)
            build_time = time.time() - t1
            print(f"-> Graph built in {build_time:.2f}s")
            
            total_time = time.time() - t0
            print("-" * 30)
            print(f"Total Execution Time: {total_time:.2f}s")
            print(f"Average Time per Chunk: {total_time / len(documents):.2f}s")
            
            # 验证性能指标
            if total_time / len(documents) < 5.0:
                 print("✅ Performance Target Met: < 5s/chunk")
            else:
                 print("⚠️ Performance Target Missed: > 5s/chunk")
                 
        else:
            print("No documents loaded.")
            
    except Exception as e:
        print(f"Benchmark Failed: {e}")
    finally:
        stop_event.set()
        monitor_thread.join()

if __name__ == "__main__":
    # 确保有数据
    if not os.path.exists(benchmark_dir) or not os.listdir(benchmark_dir):
        print("Generating benchmark data first...")
        import generate_benchmark_data
    
    run_benchmark()
