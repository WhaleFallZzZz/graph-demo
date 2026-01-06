import os
import random
import shutil

# 确保目录存在
data_dir = os.path.join(os.path.dirname(__file__), "data", "benchmark")
if os.path.exists(data_dir):
    shutil.rmtree(data_dir)
os.makedirs(data_dir, exist_ok=True)

keywords = ["近视", "远视", "散光", "弱视", "斜视", "视力", "眼轴", "角膜", "晶状体", "视网膜", "阿托品", "OK镜"]
symptoms = ["视物模糊", "眼痛", "畏光", "视力下降", "眼涨"]
treatments = ["低浓度阿托品", "角膜塑形镜(OK镜)", "RGP镜片", "离焦框架镜", "视觉训练"]

templates = [
    "患者{name}，{age}岁，主诉{symptom}持续{duration}月。",
    "眼科检查显示：{eye}眼{check_item}为{value}，{check_item2}为{value2}。",
    "初步诊断为{disease}，属于{type}。",
    "医生建议采取{treatment}进行干预，并定期复查{check_item}。",
    "经治疗{time}个月后，复查发现{check_item}控制稳定，{symptom}有所缓解。",
    "{disease}的发病机制与{factor}有关，需要注意用眼卫生。",
    "对于青少年{disease}防控，{treatment}是目前公认有效的手段之一。",
    "眼轴长度的增长是监测{disease}进展的重要指标，正常值为{normal_value}mm。"
]

def generate_text():
    lines = []
    # 生成 10-30 行文本，模拟不同长度
    for _ in range(random.randint(10, 30)):
        tmpl = random.choice(templates)
        line = tmpl.format(
            name=random.choice(["张三", "李四", "王五", "赵六", "孙七"]),
            age=random.randint(6, 16),
            symptom=random.choice(symptoms),
            duration=random.randint(1, 12),
            eye=random.choice(["左", "右", "双"]),
            check_item=random.choice(["眼轴长度", "眼压", "屈光度"]),
            value=f"{random.uniform(22, 26):.2f}",
            check_item2=random.choice(["角膜曲率", "调节幅度"]),
            value2=f"{random.uniform(40, 45):.2f}",
            disease=random.choice(keywords),
            type=random.choice(["轴性", "屈光性", "先天性"]),
            treatment=random.choice(treatments),
            time=random.randint(3, 24),
            factor=random.choice(["遗传", "环境", "光照"]),
            normal_value=f"{random.uniform(23, 24):.2f}"
        )
        lines.append(line)
    return "\n".join(lines)

print(f"Generating 100 benchmark documents in {data_dir}...")
for i in range(100):
    file_path = os.path.join(data_dir, f"doc_{i:03d}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(generate_text())

print("Done.")
