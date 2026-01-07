"""
Enhanced Entity Extractor with Standard Term Mapping
"""
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class StandardTermMapper:
    """标准术语映射器"""
    
    # 57个标准实体列表
    STANDARD_ENTITIES = {
        # 疾病(12)
        "近视", "远视", "散光", "弱视", "斜视", "病理性近视", "轴性近视", 
        "屈光不正", "屈光参差", "老视", "并发性白内障", "后巩膜葡萄肿",
        # 症状体征(13)
        "视物模糊", "眼胀", "虹视", "眼痛", "畏光", "流泪", "视力下降", 
        "豹纹状眼底", "视网膜萎缩", "脉络膜萎缩", "黄斑出血", "漆样裂纹", "视盘杯盘比(C/D)扩大",
        # 解剖结构(12)
        "角膜", "晶状体", "视网膜", "视神经", "黄斑区", "中心凹", 
        "睫状肌", "悬韧带", "脉络膜", "巩膜", "前房", "房水",
        # 检查参数(10)
        "眼轴长度", "屈光度", "远视储备", "调节幅度", "调节灵敏度", 
        "眼压", "角膜曲率", "调节滞后", "五分记录法", "LogMAR视力表",
        # 治疗防控(10)
        "户外活动", "角膜塑形镜(OK镜)", "低浓度阿托品", "RGP镜片", "后巩膜加固术", 
        "离焦框架镜", "视觉训练", "准分子激光手术(LASIK)", "全飞秒激光手术(SMILE)", "眼内接触镜植入(ICL)"
    }
    
    # 静态映射表
    # 负向约束列表：禁止提取的非医学名词
    FORBIDDEN_TERMS = {
        "青少年", "家长", "儿童", "学生", "患者", "医生", "专家", 
        "老师", "学校", "医院", "机构", "我们", "你们", "他们",
        "眼睛", "眼部", "眼球", "视力表", "检查", "治疗", "手术", # 过于宽泛的词
        "问题", "原因", "方法", "措施", "情况", "结果", "影响",
        "建议", "提示", "注意", "可能", "可以", "需要","Entity","type"
    }

    # 无效字符集合 - 包含这些字符的实体将被直接过滤
    INVALID_CHARS = {'/', '\\', '{', '}', '[', ']', '<', '>', '|', '`', '~'}
    
    # 无效前缀 - 以这些开头的实体将被过滤
    INVALID_PREFIXES = ('http', 'https', 'ftp', 'www', 'file:', 'mailto:')
    
    # 无效后缀 - 以这些结尾的实体将被过滤 (主要是文件扩展名)
    INVALID_SUFFIXES = ('.pdf', '.txt', '.doc', '.docx', '.jpg', '.png', '.json', '.xml', '.html')

    # 实体最大长度
    MAX_ENTITY_LENGTH = 50
    
    # 机构/场所后缀与关键词，用于剔除非医学实体
    INSTITUTION_SUFFIXES = ("中心", "医院", "门诊", "科室", "机构", "公司", "集团", "学校", "大学", "学院", "研究所", "实验室")
    INSTITUTION_KEYWORDS = {"视光中心", "眼视光中心", "眼科中心", "眼视光机构"}
    ALLOW_MEDICAL_CENTER_TERMS = {"中心凹", "黄斑中心凹"}
    
    # 系统/平台/软件后缀与关键词，用于剔除技术/系统类非医学实体
    SYSTEM_SUFFIXES = ("系统", "平台", "软件", "客户端", "服务端", "APP")
    SYSTEM_KEYWORDS = {
        "电子病历系统", "EMR", "HIS", "LIS", "PACS", "RIS",
        "挂号系统", "收费系统", "就诊平台", "预约平台", "管理系统",
        "医院信息系统", "医疗信息系统"
    }
    ALLOW_MEDICAL_SYSTEM_TERMS = set()

    # 修饰语正则列表
    MODIFIER_PATTERNS = [
        r"^严重的", r"^轻度的", r"^早期的", r"^晚期的", r"^急性的", r"^慢性的",
        r"^原发性", r"^继发性", r"^先天性", r"^后天性", r"^进行性",
        r"^明显的", r"^显著的", r"^可能的", r"^疑似的", r"^高度的", r"^低度的",
        r"^伴有", r"^合并", r"^引起", r"^导致",
        r"的$", # 结尾的'的'
    ]

    SYNONYM_MAP = {
    # 眼轴长度相关 (扩展)
    "眼球长度": "眼轴长度", "AL": "眼轴长度", "轴长": "眼轴长度", "眼轴": "眼轴长度", "AL长度": "眼轴长度",
    "眼轴增长": "眼轴长度", "眼轴增长速率": "眼轴长度", "眼轴增长年率": "眼轴长度",
    "眼球轴长": "眼轴长度", "轴长度": "眼轴长度", "眼球AL": "眼轴长度",
    
    # 近视相关 (扩展)
    "近视眼": "近视", "近视症": "近视", "轴性近视眼": "轴性近视", "病理性近视眼": "病理性近视",
    "近视防控": "近视", "近视加深": "近视", "近视易感性": "近视", "近视发生与发展": "近视",
    "屈光性近视": "近视", "进行性近视": "近视", "高度近视": "近视",
    "近视进展加快": "近视", "假性近视向真性近视发展": "近视",
    
    # 远视相关
    "远视眼": "远视", "远视症": "远视",
    
    # 散光相关 (扩展)
    "散光眼": "散光", "散光轴位": "散光", "规则散光": "散光", "不规则散光": "散光",
    
    # 弱视相关 (扩展)
    "弱视眼": "弱视", "弱视症": "弱视",
    "屈光不正性弱视": "弱视", "屈光参差性弱视": "弱视", "形觉剥夺性弱视": "弱视",
    "弱视类型分类": "弱视", "屈光不正性弱视诊断标准": "弱视", "斜视性弱视": "弱视",
    
    # 斜视相关 (扩展)
    "斜视眼": "斜视", "斜视症": "斜视",
    "调节性内斜视": "斜视", "非调节性内斜视": "斜视", "间歇性外斜视": "斜视",
    
    # 屈光不正相关
    "屈光异常": "屈光不正", "屈光问题": "屈光不正",
    
    # 角膜塑形镜相关 (扩展)
    "OK镜": "角膜塑形镜(OK镜)", "角膜塑形镜": "角膜塑形镜(OK镜)", "塑形镜": "角膜塑形镜(OK镜)",
    "Ortho-K": "角膜塑形镜(OK镜)", "Orthokeratology": "角膜塑形镜(OK镜)",
    
    # 低浓度阿托品相关 (扩展)
    "低浓度阿托品眼药水": "低浓度阿托品", "阿托品": "低浓度阿托品",
    "Atropine": "低浓度阿托品", "阿托品滴眼液": "低浓度阿托品",
    "阿托品眼药水": "低浓度阿托品", "低浓度阿托品滴眼液": "低浓度阿托品",
    
    # 视觉训练相关 (扩展)
    "视功能训练": "视觉训练", "视觉功能训练": "视觉训练", "视训": "视觉训练",
    "立体视觉": "视觉训练", "双眼视觉系统": "视觉训练", "视觉发育关键期": "视觉训练",
    "视觉刺激疗法": "视觉训练", "视觉治疗": "视觉训练",
    
    # 眼压相关 (扩展)
    "眼内压": "眼压", "IOP": "眼压", "眼内压力": "眼压",
    
    # 屈光度相关 (扩展)
    "度数": "屈光度", "D": "屈光度", "球镜度数": "屈光度",
    "等效球镜": "屈光度", "SE": "屈光度", "Diopter": "屈光度",
    
    # 调节幅度相关 (扩展)
    "调节力": "调节幅度", "AMP": "调节幅度",
    "调节不足": "调节幅度", "调节功能": "调节幅度", "调节能力": "调节幅度",
    
    # 调节灵敏度相关 (扩展)
    "调节灵活度": "调节灵敏度", "Flipper": "调节灵敏度",
    "调节灵敏度下降": "调节灵敏度",
    
    # 调节滞后相关
    "调节滞后量": "调节滞后", "Lag": "调节滞后",
    
    # 角膜曲率相关 (扩展)
    "K值": "角膜曲率", "角膜K值": "角膜曲率", "Keratometry": "角膜曲率",
    
    # 视物模糊相关 (扩展)
    "视力模糊": "视物模糊", "看东西模糊": "视物模糊", "模糊": "视物模糊",
    "视物不清": "视物模糊", "视物不清晰": "视物模糊",
    
    # 视力下降相关 (扩展)
    "视力降低": "视力下降", "视力减退": "视力下降",
    "视疲劳": "视力下降", "12岁以后视力难以改善": "视力下降",
    
    # 眼痛相关
    "眼睛痛": "眼痛", "眼部疼痛": "眼痛",
    
    # 畏光相关 (扩展)
    "怕光": "畏光", "光敏感": "畏光", "光敏": "畏光", "对光敏感": "畏光",
    
    # 流泪相关
    "流眼泪": "流泪", "眼泪多": "流泪",
    
    # 眼胀相关 (扩展)
    "眼睛胀": "眼胀", "眼部胀痛": "眼胀", "眼球胀痛": "眼胀",
    "眼球胀痛,甚至恶心呕吐及神经官能症": "眼胀",
    
    # 解剖结构 - 视网膜相关 (扩展)
    "眼底": "视网膜", "视网膜层": "视网膜", "双眼视网膜像差异": "视网膜",
    
    # 解剖结构 - 视神经相关 (扩展)
    "视盘": "视神经", "视乳头": "视神经", "Optic Nerve": "视神经",
    
    # 解剖结构 - 黄斑区相关 (扩展)
    "黄斑": "黄斑区", "黄斑部": "黄斑区",
    "黄斑水肿": "黄斑区", "黄斑功能退化": "黄斑区",
    
    # 解剖结构 - 中心凹相关 (扩展)
    "中央凹": "中心凹", "黄斑中心凹": "中心凹",
    "黄斑中心凹功能正常": "中心凹",
    
    # 解剖结构 - 睫状肌相关
    "睫状体": "睫状肌",
    
    # 解剖结构 - 晶状体相关 (扩展)
    "晶状体变凸": "晶状体", "晶状体屈光力": "晶状体",
    
    # 解剖结构 - 脉络膜相关
    "脉络膜层": "脉络膜",
    
    # 解剖结构 - 巩膜相关
    "巩膜层": "巩膜", "白眼球": "巩膜",
    
    # 解剖结构 - 前房相关
    "前房角": "前房",
    
    # 解剖结构 - 房水相关
    "眼房水": "房水", "前房水": "房水",
    
    # 治疗防控 - 户外活动相关
    "户外运动": "户外活动", "室外活动": "户外活动",
    
    # 治疗防控 - RGP镜片相关
    "RGP": "RGP镜片", "硬性隐形眼镜": "RGP镜片", "硬性接触镜": "RGP镜片",
    "硬性角膜接触镜": "RGP镜片", "硬性透氧性角膜接触镜": "RGP镜片",
    
    # 治疗防控 - 后巩膜加固术相关
    "后巩膜加固": "后巩膜加固术", "后巩膜手术": "后巩膜加固术",
    
    # 治疗防控 - 离焦框架镜相关 (扩展)
    "离焦眼镜": "离焦框架镜", "离焦镜片": "离焦框架镜", "离焦镜": "离焦框架镜",
    "离焦型眼镜": "离焦框架镜",
    
    # 治疗防控 - 准分子激光手术相关 (扩展)
    "LASIK手术": "准分子激光手术(LASIK)", "LASIK": "准分子激光手术(LASIK)",
    "准分子手术": "准分子激光手术(LASIK)",
    
    # 治疗防控 - 全飞秒激光手术相关 (扩展)
    "SMILE手术": "全飞秒激光手术(SMILE)", "SMILE": "全飞秒激光手术(SMILE)",
    "全飞秒手术": "全飞秒激光手术(SMILE)",
    
    # 治疗防控 - 眼内接触镜植入相关 (扩展)
    "ICL手术": "眼内接触镜植入(ICL)", "ICL": "眼内接触镜植入(ICL)",
    "眼内镜": "眼内接触镜植入(ICL)", "眼内接触镜": "眼内接触镜植入(ICL)",
    
    # 检查参数 - 远视储备相关 (扩展)
    "远视储备值": "远视储备", "远储": "远视储备", "远视储备消耗": "远视储备",
    
    # 检查参数 - 五分记录法相关
    "5分记录法": "五分记录法", "五分制": "五分记录法",
    
    # 检查参数 - LogMAR视力表相关 (扩展)
    "LogMAR": "LogMAR视力表", "LogMAR表": "LogMAR视力表",
    "国际通用视力表": "LogMAR视力表", "国际通用视力评估标准": "LogMAR视力表",
}
    
    @classmethod
    def remove_modifiers(cls, entity_name: str) -> str:
        """移除实体修饰语"""
        if not entity_name:
            return entity_name
            
        import re
        clean_name = entity_name.strip()
        
        # 迭代应用正则规则
        for pattern in cls.MODIFIER_PATTERNS:
            prev_name = clean_name
            clean_name = re.sub(pattern, "", clean_name).strip()
            if prev_name != clean_name:
                logger.debug(f"修饰语消除: '{prev_name}' -> '{clean_name}'")
                
        return clean_name

    @classmethod
    def standardize(cls, entity_name: str) -> str:
        """
        标准化实体名称
        """
        if not hasattr(cls, "_lru_cache"):
            from collections import OrderedDict
            cls._lru_cache = OrderedDict()
            cls._lru_max = 2048
        if not entity_name:
            return entity_name
            
        key = entity_name.strip()
        cached = cls._lru_cache.get(key)
        if cached is not None:
            cls._lru_cache.move_to_end(key)
            return cached
        clean_name = key
        clean_name = cls.remove_modifiers(clean_name)
        if clean_name in cls.SYNONYM_MAP:
            standard_name = cls.SYNONYM_MAP[clean_name]
            result = standard_name
        else:
            result = clean_name
        cls._lru_cache[key] = result
        if len(cls._lru_cache) > cls._lru_max:
            cls._lru_cache.popitem(last=False)
        return result

    @classmethod
    def validate_batch(cls, triplets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量验证三元组有效性
        """
        valid_triplets = []
        # 批量检查逻辑
        for triplet in triplets:
            head = triplet.get("head")
            tail = triplet.get("tail")
            relation = triplet.get("relation")
            
            # 1. 基础完整性检查
            if not (head and tail and relation):
                continue
                
            # 2. 避免自环
            if head == tail:
                continue
                
            # 3. 实体类型检查 (如果开启了严格模式)
            # 这里我们执行"软验证"，只标记不过滤，或者根据置信度过滤
            # 为了满足用户"验证"的需求，我们确保实体不是纯数字或特殊符号
            if not cls._is_valid_entity(head) or not cls._is_valid_entity(tail):
                continue
                
            valid_triplets.append(triplet)
        return valid_triplets

    @classmethod
    def _is_valid_entity(cls, text: str) -> bool:
        if not text or len(text) < 1:
            return False
        
        # 0. 长度检查
        if len(text) > cls.MAX_ENTITY_LENGTH:
            logger.warning(f"实体被长度限制拦截 (长度 {len(text)} > {cls.MAX_ENTITY_LENGTH}): '{text[:20]}...'")
            return False

        # 1. 负向约束检查
        if text in cls.FORBIDDEN_TERMS:
            logger.debug(f"实体被负向约束拦截: '{text}'")
            return False
        
        # 2. 无效字符检查 (包含路径分隔符等)
        for char in cls.INVALID_CHARS:
            if char in text:
                logger.warning(f"实体被无效字符拦截 ('{char}'): '{text}'")
                return False

        # 3. 无效前缀检查 (URL, file协议等)
        if text.lower().startswith(cls.INVALID_PREFIXES):
            logger.warning(f"实体被无效前缀拦截: '{text}'")
            return False

        # 4. 无效后缀检查 (文件扩展名等)
        if text.lower().endswith(cls.INVALID_SUFFIXES):
            logger.warning(f"实体被无效后缀拦截: '{text}'")
            return False

        # 5. 过滤纯数字
        if text.replace(".", "").isdigit():
            return False
            
        # 6. 过滤纯标点
        if all(char in ",./<>?;':\"[]\\{}|`~!@#$%^&*()-_=+" for char in text):
            return False
        
        # 7. 机构/场所剔除：以机构后缀结尾或包含机构关键词，且不在医学允许术语例外列表
        try:
            clean = text.strip()
            if clean not in cls.ALLOW_MEDICAL_CENTER_TERMS:
                if any(clean.endswith(suf) for suf in cls.INSTITUTION_SUFFIXES):
                    return False
                if any(kw in clean for kw in cls.INSTITUTION_KEYWORDS):
                    return False
            # 8. 系统/平台/软件剔除：以系统类后缀结尾或包含典型系统关键词
            if clean not in cls.ALLOW_MEDICAL_SYSTEM_TERMS:
                if any(clean.endswith(suf) for suf in cls.SYSTEM_SUFFIXES):
                    return False
                if any(kw in clean for kw in cls.SYSTEM_KEYWORDS):
                    return False
        except Exception:
            pass
            
        return True

    @classmethod
    def process_triplets(cls, triplets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量处理三元组，标准化其中的实体并进行批量验证
        """
        processed_triplets = []
        for triplet in triplets:
            head = triplet.get("head")
            tail = triplet.get("tail")
            
            # 标准化
            std_head = cls.standardize(head)
            std_tail = cls.standardize(tail)
            
            # 记录变化
            if std_head != head:
                triplet["original_head"] = head
                triplet["head"] = std_head
            
            if std_tail != tail:
                triplet["original_tail"] = tail
                triplet["tail"] = std_tail
            
            processed_triplets.append(triplet)
        
        # 执行批量验证
        return cls.validate_batch(processed_triplets)
