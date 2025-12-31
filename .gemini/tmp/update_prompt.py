import re
import os

file_path = "llama/config.py"

new_prompt_content = """你是一位专业的知识图谱构建专家，精通中文。你的任务是**从文本中提取尽可能多的、细致入微的实体和关系**，旨在构建一个高度详细和连接紧密的知识图谱。

核心指令：**最大化提取（Maximize Extraction）**

1.  **全面识别所有实体**：
    *   **任何有意义的名词或名词短语**都应被视为潜在实体。这包括但不限于：人物（姓名、昵称、职位）、组织（公司、部门、机构）、地点（城市、建筑、房间、地理区域）、时间（日期、年份、时间段）、事件、物品（文件、设备、资产）、概念、数字、代码、ID、状态、属性值等。
    *   即使是看起来微不足道的细节，如果它能独立存在或与其他实体建立联系，就提取它。
2.  **识别所有可能的联系**：
    *   仔细分析句子中的**动词、介词、形容词和具有连接意义的词汇**，它们是构建关系的关键。
    *   寻找**直接的、间接的、隐含的**或**逻辑上的**关联。例如，因果关系、包含关系、时间顺序、属性描述等。
3.  **丰富关系属性**：
    *   每条关系都应附带所有相关的属性，以提供完整的上下文。
    *   属性的格式严格为 `关系类型 (属性1: 值1, 属性2: 值2, ...)`。
    *   属性值必须具体，例如，金额要包含货币和单位，时间要具体到年、月、日。
4.  **处理别名与指代**：
    *   如果文本中出现同一个实体的不同称呼（别名、简称、代词），务必提取“别名”或“指代”关系。
    *   将代词（如“他”、“这”）解析为具体实体，并建立关系。
5.  **输出格式**：
    *   输出必须是一个标准的JSON数组，每个元素是一个表示三元组的JSON对象。
    *   每个三元组应包含 `head`, `head_type`, `relation`, `tail`, `tail_type` 字段。
    *   确保输出的JSON格式完全正确。

示例（要求提取非常高的密度，包含所有细节）：
输入：“2023年5月10日下午，张峰（上海华夏科技公司CEO）在上海半岛酒店301房间秘密会见了黑客李明。张峰支付了50万USDT购买了‘天眼’系统源代码，该系统由华夏科技研发。”
输出：
```json
[
  {"head": "张峰", "head_type": "人物", "relation": "担任职位", "tail": "CEO", "tail_type": "职位"},
  {"head": "张峰", "head_type": "人物", "relation": "任职于", "tail": "上海华夏科技公司", "tail_type": "组织"},
  {"head": "张峰", "head_type": "人物", "relation": "会见 (时间: 2023年5月10日下午, 方式: 秘密)", "tail": "李明", "tail_type": "人物"},
  {"head": "李明", "head_type": "人物", "relation": "身份", "tail": "黑客", "tail_type": "身份"},
  {"head": "会见事件", "head_type": "事件", "relation": "发生时间", "tail": "2023年5月10日下午", "tail_type": "时间"},
  {"head": "会见事件", "head_type": "事件", "relation": "发生地点", "tail": "半岛酒店301房间", "tail_type": "地点"},
  {"head": "半岛酒店301房间", "head_type": "地点", "relation": "位于", "tail": "上海", "tail_type": "地点"},
  {"head": "张峰", "head_type": "人物", "relation": "支付 (金额: 50万, 货币: USDT)", "tail": "李明", "tail_type": "人物"},
  {"head": "张峰", "head_type": "人物", "relation": "购买", "tail": "天眼系统源代码", "tail_type": "资产"},
  {"head": "天眼系统源代码", "head_type": "资产", "relation": "属于", "tail": "天眼系统", "tail_type": "软件"},
  {"head": "天眼系统", "head_type": "软件", "relation": "由研发", "tail": "华夏科技", "tail_type": "组织"}
]
```

Text: {text}"""

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to find the 'extract_prompt' key and its multi-line string value
    # This regex is more robust as it captures the entire string content between """..."""
    # It accounts for the indentation before "extract_prompt" and after the triple quotes
    # The (?:.|
)*? makes it non-greedy and matches across newlines
    pattern = r'(    "extract_prompt": "))((?:.|
)*?)(""")'
    
    # The replacement captures the prefix and suffix of the old pattern, inserting new_prompt_content in the middle
    modified_content = re.sub(pattern, r'\1' + new_prompt_content + r'\3', content, flags=re.DOTALL)

    if content == modified_content:
        # If no change, try to match the *original* simpler prompt, in case the file was reverted differently
        original_simple_prompt_content = """你是一位不仅精通中文，而且这种极度注重细节的知识图谱专家。你的目标是**穷尽**文本中的所有信息。

核心指令：**地毯式提取（Exhaustive Extraction）**

1. **拒绝摘要**：绝不要只提取主要情节或主要人物。文中的每一个**微小细节**（包括具体的时间点、地点、物品、次要人物、职位、文件名称等）都必须作为独立的节点提取出来。
2. **万物皆实体**：
   - 不要忽略非人类实体。例如：“举报信”、“离岸账户”、“服务器”、“密钥”、“会议记录”都是重要的**资产**或**物品**实体。
   - 不要忽略地点。每一个提到的城市、街道、房间、国家都是**地点**实体。
   - 不要忽略时间。具体的年份、日期、时间段都应作为属性或独立的时间节点（如果它连接多个事件）。
3. **关系稠密化**：
   - 句子中的每一个动词或介词都可能隐含一个关系。
   - 提取属性！关系必须包含丰富的上下文信息。
   - 格式：`关系类型 (属性: 值, 属性: 值)`。例如 `被扣押 (地点: 海关, 时间: 昨天)`。
4. **别名与指代**：
   - 必须提取“别名”关系。如“老王”是“王力”的别名。

示例（注意提取的密度）：
输入：“2023年5月，张峰在上海的半岛酒店秘密会见了代号‘幽灵’的黑客，支付了50万USDT购买了‘天眼’系统的源代码。”
输出：
[
  {"head": "张峰", "head_type": "人物", "relation": "会见 (时间: 2023年5月, 地点: 上海半岛酒店, 方式: 秘密)", "tail": "幽灵", "tail_type": "人物"},
  {"head": "幽灵", "head_type": "人物", "relation": "拥有身份 (代号)", "tail": "黑客", "tail_type": "角色"},
  {"head": "幽灵", "head_type": "人物", "relation": "别名", "tail": "黑客", "tail_type": "角色"},
  {"head": "上海半岛酒店", "head_type": "地点", "relation": "位于", "tail": "上海", "tail_type": "地点"},
  {"head": "张峰", "head_type": "人物", "relation": "支付 (金额: 50万, 货币: USDT)", "tail": "幽灵", "tail_type": "人物"},
  {"head": "张峰", "head_type": "人物", "relation": "购买", "tail": "天眼系统源代码", "tail_type": "资产"},
  {"head": "天眼系统源代码", "head_type": "资产", "relation": "属于", "tail": "天眼系统", "tail_type": "项目"}
]

请模仿上述示例的**密度**，对下面的文本进行提取。即使是细枝末节也不要放过。

Text: {text}"""
        
        pattern_simple = r'(    "extract_prompt": "))((?:.|
)*?)(""")'
        modified_content = re.sub(pattern_simple, r'\1' + new_prompt_content + r'\3', content, flags=re.DOTALL)
        
        if content == modified_content:
            print("Error: extract_prompt not found or content not modified by regex (after trying both complex and simple original prompts). This indicates an unexpected change in the file structure.")
        else:
            print(modified_content)

except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")