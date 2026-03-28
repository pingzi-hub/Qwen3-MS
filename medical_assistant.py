#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医疗助手集成脚本
基于 Qwen3-0.6B 医疗微调模型，提供多种医疗场景的智能助手功能
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import json
import time
from datetime import datetime
import os
import re

# 医疗专业提示词模板
MEDICAL_PROMPTS = {
    "diagnosis": "你是临床医生，根据症状给出初步诊断与检查建议。",
    "treatment": "你是医生，提供治疗方案、用药指导。",
    "prevention": "你是预防医学专家，提供疾病预防与健康建议。",
    "education": "你是医学教育专家，通俗解释医学知识。",
    "emergency": "你是急诊科医生，评估紧急程度。",
    "nutrition": "你是营养师，提供饮食指导。",
    "mental_health": "你是心理医生，提供心理支持。",
    "pediatric": "你是儿科医生，提供儿童医疗建议。",
    "geriatric": "你是老年医学专家，服务老年人。",
    "women_health": "你是妇科医生，服务女性健康。",
    "first_aid": "急救指导。",
    "palliative_care": "安宁疗护。",
    "medical_decision": "治疗方案决策支持。",
    "pain_management": "疼痛管理。",
    "digital_health": "数字健康指导。",
    "hospital_recommend": "你是导诊专家，根据症状推荐科室。",
    "lab_analysis": "你是检验专家，分析体检指标。",
}

# 常见医疗场景
MEDICAL_SCENARIOS = {
    "1": "症状诊断",
    "2": "治疗方案",
    "3": "疾病预防",
    "4": "医学教育",
    "5": "紧急评估",
    "6": "营养指导",
    "7": "心理健康",
    "8": "儿科咨询",
    "9": "老年健康",
    "10": "女性健康",
    "11": "🏥 智能导诊（症状→科室→医院）",
    "12": "📊 体检指标分析（血脂/血糖等）",
}

#新增：医院科室匹配库
HOSPITAL_DEPARTMENTS = {
    # 内科系统
    "头痛头晕": "神经内科",
    "偏头痛": "神经内科",
    "头晕目眩": "神经内科/耳鼻喉科",
    "失眠多梦": "神经内科/心理科/睡眠科",
    "记忆力下降": "神经内科",
    "手脚麻木": "神经内科/骨科",
    "面瘫": "神经内科",

    "胸痛胸闷": "心血管内科",
    "心慌心悸": "心血管内科",
    "心律不齐": "心血管内科",
    "高血压": "心血管内科",
    "冠心病": "心血管内科",

    "腹痛腹胀": "消化内科",
    "胃痛胃胀": "消化内科",
    "反酸烧心": "消化内科",
    "恶心呕吐": "消化内科",
    "腹泻拉肚子": "消化内科",
    "便秘": "消化内科/肛肠科",
    "便血": "消化内科/肛肠科",
    "黄疸": "消化内科/肝胆外科",

    "咳嗽咳痰": "呼吸内科",
    "呼吸困难": "呼吸内科/急诊科",
    "哮喘": "呼吸内科",
    "肺炎": "呼吸内科",
    "支气管炎": "呼吸内科",

    "关节痛": "风湿免疫科/骨科",
    "关节肿痛": "风湿免疫科",
    "腰背痛": "骨科/康复科",
    "颈椎痛": "骨科/康复科",
    "骨质疏松": "骨科/内分泌科",

    "血糖高": "内分泌科",
    "糖尿病": "内分泌科",
    "甲状腺问题": "内分泌科",
    "肥胖减肥": "内分泌科/营养科",
    "生长发育": "内分泌科/儿科",

    "尿频尿急": "泌尿外科/肾内科",
    "血尿": "泌尿外科/肾内科",
    "蛋白尿": "肾内科",
    "水肿": "肾内科/心内科",
    "肾结石": "泌尿外科",

    # 外科系统
    "外伤出血": "急诊科/骨科",
    "骨折": "骨科",
    "扭伤": "骨科/康复科",
    "烧伤烫伤": "烧伤科/急诊科",
    "肿瘤": "肿瘤科",
    "癌症": "肿瘤科",
    "乳腺肿块": "乳腺外科/甲乳外科",
    "甲状腺结节": "甲乳外科/内分泌科",
    "胆囊结石": "肝胆外科",
    "疝气": "普外科",
    "静脉曲张": "血管外科",
    "动脉硬化": "血管外科/心血管内科",

    # 妇产科
    "月经不调": "妇科",
    "痛经": "妇科",
    "白带异常": "妇科",
    "妇科炎症": "妇科",
    "备孕不孕": "生殖医学科/妇科",
    "怀孕产检": "产科",
    "产后恢复": "产科/康复科",
    "更年期": "妇科/内分泌科",

    # 儿科
    "儿童发热": "儿科",
    "儿童咳嗽": "儿科",
    "儿童腹泻": "儿科",
    "儿童皮疹": "儿科/皮肤科",
    "儿童发育": "儿科/儿保科",
    "新生儿": "新生儿科/儿科",
    "预防接种": "儿保科",

    # 皮肤科
    "皮肤痒": "皮肤科",
    "皮疹": "皮肤科",
    "湿疹": "皮肤科",
    "痤疮痘痘": "皮肤科",
    "脱发": "皮肤科/毛发科",
    "白癜风": "皮肤科",
    "银屑病": "皮肤科",

    # 五官科
    "眼睛红痛": "眼科",
    "视力下降": "眼科",
    "近视手术": "眼科",
    "白内障": "眼科",
    "青光眼": "眼科",

    "耳朵痛": "耳鼻喉科",
    "耳鸣": "耳鼻喉科/神经内科",
    "听力下降": "耳鼻喉科",
    "中耳炎": "耳鼻喉科",

    "鼻塞流涕": "耳鼻喉科",
    "鼻炎": "耳鼻喉科",
    "鼻出血": "耳鼻喉科",
    "鼻窦炎": "耳鼻喉科",

    "咽喉痛": "耳鼻喉科",
    "声音嘶哑": "耳鼻喉科",
    "扁桃体炎": "耳鼻喉科",
    "打鼾": "耳鼻喉科/睡眠科",

    "牙痛": "口腔科",
    "牙龈出血": "口腔科",
    "蛀牙": "口腔科",
    "牙齿矫正": "口腔正畸科",
    "洗牙": "口腔科",

    # 精神心理
    "焦虑抑郁": "心理科/精神科",
    "强迫症": "心理科/精神科",
    "心理咨询": "心理科",
    "睡眠障碍": "睡眠科/心理科",

    # 其他专科
    "体检": "体检中心/健康管理中心",
    "职业病": "职业病科",
    "康复理疗": "康复科",
    "中医调理": "中医科",
    "针灸推拿": "针灸科/推拿科",
    "营养咨询": "营养科",
    "疼痛管理": "疼痛科",
    "过敏": "变态反应科/皮肤科",
    "艾滋病": "感染科/皮肤科",
    "肝炎": "感染科/肝病科",
    "结核病": "感染科/呼吸科",
    "狂犬疫苗": "急诊科/预防保健科",

    # 急重症
    "胸痛呼吸困难": "急诊科/心血管内科",
    "剧烈腹痛": "急诊科/普外科",
    "高热不退": "急诊科/感染科",
    "意识不清": "急诊科/神经内科",
    "食物中毒": "急诊科/消化内科",
    "药物中毒": "急诊科",
    "动物咬伤": "急诊科",
    "溺水": "急诊科",
    "电击": "急诊科",
    "窒息": "急诊科",
}

#体检指标标准库（血脂/血糖/血压/尿酸）
LAB_REFERENCE = {
    "血压(高压)": {"normal": (90, 140), "high": ">140 高血压", "low": "<90 低血压"},
    "血压(低压)": {"normal": (60, 90), "high": ">90 高血压", "low": "<60 低血压"},
    "空腹血糖": {"normal": (3.9, 6.1), "high": ">6.1 血糖偏高", "low": "<3.9 低血糖"},
    "餐后2h血糖": {"normal": (3.9, 7.8), "high": ">7.8 糖耐量异常"},
    "糖化血红蛋白": {"normal": (4, 6), "high": ">6.5 糖尿病"},

    "甘油三酯": {"normal": (0.56, 1.7), "high": ">1.7 高甘油三酯血症"},
    "总胆固醇": {"normal": (0, 5.2), "high": ">5.2 高胆固醇血症"},
    "高密度脂蛋白": {"normal": (1.04, 1.55), "low": "<1.04 心血管风险"},
    "低密度脂蛋白": {"normal": (0, 3.4), "high": ">3.4 高LDL风险"},

    "尿酸(男)": {"normal": (208, 428), "high": ">428 高尿酸血症"},
    "尿酸(女)": {"normal": (155, 357), "high": ">357 高尿酸血症"},

    "谷丙转氨酶": {"normal": (7, 40), "high": ">40 肝功能异常"},
    "谷草转氨酶": {"normal": (13, 35), "high": ">35 肝功能异常"},
    "总胆红素": {"normal": (3.4, 20.5), "high": ">20.5 黄疸可能"},

    "肌酐(男)": {"normal": (53, 106), "high": ">106 肾功能异常"},
    "肌酐(女)": {"normal": (44, 97), "high": ">97 肾功能异常"},
    "尿素氮": {"normal": (2.9, 7.1), "high": ">7.1 肾功能异常"},

    "白细胞": {"normal": (4, 10), "high": ">10 感染/炎症", "low": "<4 免疫力低"},
    "红细胞(男)": {"normal": (4.3, 5.8), "low": "<4.3 贫血"},
    "红细胞(女)": {"normal": (3.8, 5.1), "low": "<3.8 贫血"},
    "血红蛋白(男)": {"normal": (130, 175), "low": "<130 贫血"},
    "血红蛋白(女)": {"normal": (115, 150), "low": "<115 贫血"},
    "血小板": {"normal": (100, 300), "high": ">300 血栓风险", "low": "<100 出血风险"},
    "促甲状腺激素": {"normal": (0.27, 4.2), "high": ">4.2 甲减可能", "low": "<0.27 甲亢可能"},
    "癌胚抗原": {"normal": (0, 5), "high": ">5 肿瘤筛查建议"},
    "甲胎蛋白": {"normal": (0, 7), "high": ">7 肝病筛查建议"},
    "前列腺特异抗原": {"normal": (0, 4), "high": ">4 前列腺检查建议"},
}
#医院推荐数据库
# 全国34省 医院推荐数据库（完整版）
HOSPITAL_DATABASE = {
    "北京": {
        "综合医院": ["北京协和医院", "中国人民解放军总医院(301医院)", "北京大学第一医院", "北京大学第三医院",
                     "北京医院"],
        "神经内科": ["北京协和医院神经内科", "宣武医院神经内科", "天坛医院神经内科", "301医院神经内科"],
        "心血管内科": ["中国医学科学院阜外医院", "北京安贞医院", "北京大学人民医院心内科"],
        "消化内科": ["北京协和医院消化内科", "解放军总医院消化内科", "北京友谊医院消化内科"],
        "呼吸内科": ["北京协和医院呼吸内科", "中日友好医院呼吸中心", "北京朝阳医院呼吸科"],
        "内分泌科": ["北京协和医院内分泌科", "北京大学第一医院内分泌科"],
        "骨科": ["北京积水潭医院", "北京大学第三医院骨科", "解放军总医院骨科"],
        "肿瘤科": ["中国医学科学院肿瘤医院", "北京大学肿瘤医院"],
        "儿科": ["北京儿童医院", "首都儿科研究所", "北京大学第一医院儿科"],
        "妇科": ["北京协和医院妇科", "北京大学人民医院妇科", "北京妇产医院"],
        "产科": ["北京协和医院产科", "北京大学第三医院产科", "北京妇产医院"],
        "眼科": ["北京同仁医院眼科", "北京协和医院眼科", "北京大学人民医院眼科"],
        "口腔科": ["北京大学口腔医院", "北京口腔医院"],
        "皮肤科": ["北京大学第一医院皮肤科", "北京协和医院皮肤科"],
        "中医科": ["北京中医药大学东直门医院", "中国中医科学院广安门医院"],
        "急诊科": ["北京协和医院急诊科", "解放军总医院急诊科"],
    },

    "上海": {
        "综合医院": ["复旦大学附属华山医院", "上海交通大学医学院附属瑞金医院", "复旦大学附属中山医院"],
        "神经内科": ["华山医院神经内科", "瑞金医院神经内科", "仁济医院神经内科"],
        "心血管内科": ["中山医院心内科", "瑞金医院心内科", "上海市胸科医院"],
        "消化内科": ["中山医院消化内科", "仁济医院消化内科"],
        "肿瘤科": ["复旦大学附属肿瘤医院", "上海市胸科医院"],
        "儿科": ["上海儿童医学中心", "复旦大学附属儿科医院"],
        "精神心理科": ["上海市精神卫生中心"],
    },

    "广州": {
        "综合医院": ["中山大学附属第一医院", "广东省人民医院", "南方医科大学南方医院"],
        "神经内科": ["中山一院神经内科", "南方医院神经内科"],
        "心血管内科": ["广东省人民医院心内科", "中山一院心内科"],
        "肿瘤科": ["中山大学肿瘤防治中心"],
        "眼科": ["中山大学中山眼科中心"],
        "口腔科": ["中山大学光华口腔医院"],
    },

    "深圳": {
        "综合医院": ["深圳市人民医院", "北京大学深圳医院", "深圳市第二人民医院"],
        "神经内科": ["深圳市第二人民医院神经内科", "北大深圳医院神经内科"],
        "心血管内科": ["深圳市人民医院心内科"],
        "儿科": ["深圳市儿童医院"],
    },

    "成都": {
        "综合医院": ["四川大学华西医院", "四川省人民医院", "成都军区总医院"],
        "神经内科": ["华西医院神经内科", "四川省人民医院神经内科"],
        "心血管内科": ["华西医院心内科"],
        "肿瘤科": ["四川省肿瘤医院"],
    },

    "重庆": {
        "综合医院": ["重庆医科大学附属第一医院", "第三军医大学西南医院", "新桥医院"],
        "神经内科": ["重医一院神经内科", "西南医院神经内科"],
    },

    "杭州": {
        "综合医院": ["浙江大学医学院附属第一医院", "浙江大学医学院附属第二医院", "浙江省人民医院"],
        "神经内科": ["浙医一院神经内科", "浙医二院神经内科"],
    },

    "南京": {
        "综合医院": ["南京鼓楼医院", "江苏省人民医院", "南京军区总医院"],
        "神经内科": ["鼓楼医院神经内科", "江苏省人民医院神经内科"],
    },

    "武汉": {
        "综合医院": ["华中科技大学同济医学院附属同济医院", "华中科技大学同济医学院附属协和医院"],
        "神经内科": ["同济医院神经内科", "协和医院神经内科"],
    },

    "西安": {
        "综合医院": ["西安交通大学第一附属医院", "第四军医大学西京医院", "陕西省人民医院"],
        "神经内科": ["交大一附院神经内科", "西京医院神经内科"],
    },

    "天津": {
        "综合医院": ["天津医科大学总医院", "天津市第一中心医院", "泰达国际心血管病医院"],
        "神经内科": ["天津医科大学总医院神经内科"],
    },

    "沈阳": {
        "综合医院": ["中国医科大学附属第一医院", "中国医科大学附属盛京医院", "沈阳军区总医院"],
        "神经内科": ["医大一院神经内科", "盛京医院神经内科"],
    },

    "哈尔滨": {
        "综合医院": ["哈尔滨医科大学附属第一医院", "哈尔滨医科大学附属第二医院"],
        "神经内科": ["哈医大一院神经内科"],
    },

    "长春": {
        "综合医院": ["吉林大学第一医院", "吉林大学第二医院"],
        "神经内科": ["吉大一院神经内科"],
    },

    "石家庄": {
        "综合医院": ["河北医科大学第二医院", "河北省人民医院"],
        "神经内科": ["河北医大二院神经内科"],
    },

    "郑州": {
        "综合医院": ["郑州大学第一附属医院", "河南省人民医院"],
        "神经内科": ["郑大一附院神经内科", "河南省人民医院神经内科"],
    },

    "济南": {
        "综合医院": ["山东大学齐鲁医院", "山东省立医院"],
        "神经内科": ["齐鲁医院神经内科", "省立医院神经内科"],
    },

    "合肥": {
        "综合医院": ["安徽医科大学第一附属医院", "安徽省立医院"],
        "神经内科": ["安医大一附院神经内科"],
    },

    "福州": {
        "综合医院": ["福建医科大学附属协和医院", "福建省立医院"],
        "神经内科": ["福建协和医院神经内科"],
    },

    "长沙": {
        "综合医院": ["中南大学湘雅医院", "湘雅二医院", "湖南省人民医院"],
        "神经内科": ["湘雅医院神经内科", "湘雅二医院神经内科"],
    },

    "南昌": {
        "综合医院": ["南昌大学第一附属医院", "江西省人民医院"],
        "神经内科": ["南大一附院神经内科"],
    },

    "南宁": {
        "综合医院": ["广西医科大学第一附属医院", "广西壮族自治区人民医院"],
        "神经内科": ["广西医大一附院神经内科"],
    },

    "昆明": {
        "综合医院": ["昆明医科大学第一附属医院", "云南省第一人民医院"],
        "神经内科": ["昆医大一附院神经内科"],
    },

    "贵阳": {
        "综合医院": ["贵州医科大学附属医院", "贵州省人民医院"],
        "神经内科": ["贵医附院神经内科"],
    },

    "兰州": {
        "综合医院": ["兰州大学第一医院", "甘肃省人民医院"],
        "神经内科": ["兰大一院神经内科"],
    },

    "西宁": {
        "综合医院": ["青海大学附属医院", "青海省人民医院"],
        "神经内科": ["青大附院神经内科"],
    },

    "银川": {
        "综合医院": ["宁夏医科大学总医院", "宁夏回族自治区人民医院"],
        "神经内科": ["宁医大总院神经内科"],
    },

    "乌鲁木齐": {
        "综合医院": ["新疆医科大学第一附属医院", "新疆维吾尔自治区人民医院"],
        "神经内科": ["新疆医大一附院神经内科"],
    },

    "拉萨": {
        "综合医院": ["西藏自治区人民医院", "西藏军区总医院"],
        "神经内科": ["西藏自治区人民医院神经内科"],
    },

    "海口": {
        "综合医院": ["海南省人民医院", "海南医学院第一附属医院"],
        "神经内科": ["海南省人民医院神经内科"],
    },

    # 经济特区/计划单列市
    "厦门": {
        "综合医院": ["厦门大学附属第一医院", "厦门中山医院"],
        "神经内科": ["厦大附一院神经内科"],
    },

    "青岛": {
        "综合医院": ["青岛大学附属医院", "青岛市立医院"],
        "神经内科": ["青大附院神经内科"],
    },

    "大连": {
        "综合医院": ["大连医科大学附属第一医院", "大连市中心医院"],
        "神经内科": ["大医一院神经内科"],
    },

    "宁波": {
        "综合医院": ["宁波市第一医院", "宁波大学医学院附属医院"],
        "神经内科": ["宁波一院神经内科"],
    },

    # 港澳台
    "香港": {
        "综合医院": ["香港玛丽医院", "香港威尔斯亲王医院", "香港伊丽莎白医院"],
        "神经内科": ["香港玛丽医院神经科"],
    },

    "澳门": {
        "综合医院": ["仁伯爵综合医院", "镜湖医院"],
        "神经内科": ["仁伯爵综合医院神经科"],
    },

    "台北": {
        "综合医院": ["台大医院", "台北荣民总医院", "长庚医院"],
        "神经内科": ["台大医院神经部"],
    },

    "高雄": {
        "综合医院": ["高雄医学大学附设医院", "高雄长庚医院"],
        "神经内科": ["高医神经内科"],
    }
}


# 预设问题示例
SAMPLE_QUESTIONS = {
    "diagnosis": [
        "我最近经常头痛，伴有恶心，这是什么原因？",
        "胸痛持续了3天，呼吸时加重，可能是什么问题？",
        "持续发热一周，体温在38-39度之间，需要做什么检查？"
    ],
    "treatment": [
        "高血压患者应该如何控制血压？",
        "糖尿病患者除了控制血糖，还需要注意什么？",
        "感冒期间应该怎么用药？"
    ],
    "prevention": [
        "如何预防心血管疾病？",
        "冬季如何预防感冒？",
        "如何预防骨质疏松？"
    ],
    "education": [
        "什么是高血压？",
        "糖尿病的发病机制是什么？",
        "心肌梗死是如何发生的？"
    ]
}



#医院科室匹配库
HOSPITAL_DEPARTMENTS = {
    # 内科系统
    "头痛|头晕|眩晕|偏头痛|失眠|记忆力下降|手脚麻木|面瘫": "神经内科",
    "胸痛|胸闷|心慌|心悸|心律不齐|高血压|冠心病|心力衰竭": "心血管内科",
    "腹痛|腹胀|胃痛|胃胀|反酸|烧心|恶心|呕吐|腹泻|便秘|便血|黄疸|肝区痛": "消化内科",
    "咳嗽|咳痰|呼吸困难|哮喘|肺炎|支气管炎|胸痛伴咳嗽": "呼吸内科",
    "关节痛|关节肿痛|腰痛|颈椎痛|背痛|骨质疏松|痛风": "风湿免疫科/骨科",
    "血糖高|糖尿病|甲状腺|肥胖|减肥|生长发育异常|内分泌失调": "内分泌科",
    "尿频|尿急|尿痛|血尿|蛋白尿|水肿|肾区痛|肾结石": "泌尿外科/肾内科",

    # 外科系统
    "外伤|出血|骨折|扭伤|韧带损伤|关节脱位": "骨科/急诊科",
    "烧伤|烫伤|冻伤|电击伤": "烧伤科/急诊科",
    "肿瘤|癌症|肿块|结节|占位|恶性肿瘤": "肿瘤科",
    "乳腺肿块|乳腺结节|乳腺增生|乳腺癌": "乳腺外科/甲乳外科",
    "甲状腺结节|甲状腺肿大|甲状腺癌": "甲乳外科/内分泌科",
    "胆囊结石|胆道疾病|胆囊炎": "肝胆外科",
    "疝气|腹股沟包块": "普外科",
    "静脉曲张|血管瘤|动脉硬化|深静脉血栓": "血管外科",

    # 妇产科
    "月经不调|痛经|白带异常|妇科炎症|阴道炎|宫颈炎|盆腔炎": "妇科",
    "备孕|不孕|不孕不育|试管|人工授精": "生殖医学科/妇科",
    "怀孕|产检|孕期|胎动异常|羊水异常": "产科",
    "产后恢复|产后抑郁|产后检查": "产科/康复科",
    "更年期|围绝经期|潮热|盗汗": "妇科/内分泌科",

    # 儿科
    "儿童发热|小儿发热|宝宝发烧|婴儿发热": "儿科",
    "儿童咳嗽|小儿咳嗽|宝宝咳嗽": "儿科",
    "儿童腹泻|小儿腹泻|宝宝拉肚子": "儿科",
    "儿童皮疹|小儿湿疹|宝宝皮疹": "儿科/皮肤科",
    "儿童发育|生长发育迟缓|智力发育|语言发育": "儿科/儿保科",
    "新生儿|早产儿|婴儿黄疸|新生儿肺炎": "新生儿科/儿科",
    "预防接种|疫苗|预防针|免疫接种": "儿保科",

    # 皮肤科
    "皮肤痒|皮疹|湿疹|痤疮|痘痘|青春痘|粉刺": "皮肤科",
    "脱发|斑秃|白发|头皮屑|头皮痒": "皮肤科/毛发科",
    "白癜风|银屑病|牛皮癣|荨麻疹|皮炎|皮肤过敏": "皮肤科",

    # 五官科
    "眼睛红|眼痛|视力下降|近视|远视|散光|白内障|青光眼|结膜炎": "眼科",
    "耳朵痛|耳鸣|听力下降|中耳炎|外耳炎|耳聋": "耳鼻喉科",
    "鼻塞|流涕|鼻炎|鼻窦炎|鼻出血|鼻息肉|过敏性鼻炎": "耳鼻喉科",
    "咽喉痛|声音嘶哑|扁桃体炎|咽炎|喉炎|打鼾|睡眠呼吸暂停": "耳鼻喉科",
    "牙痛|牙龈出血|蛀牙|牙周炎|口腔溃疡|牙齿矫正|洗牙|种植牙": "口腔科",

    # 精神心理
    "焦虑|抑郁|强迫症|心理咨询|心理治疗|心理问题|情绪低落": "心理科/精神科",
    "睡眠障碍|失眠症|嗜睡|睡眠呼吸暂停|梦游": "睡眠科/心理科",

    # 其他专科
    "体检|健康体检|入职体检|年度体检": "体检中心/健康管理中心",
    "职业病|尘肺|职业中毒|噪声聋": "职业病科",
    "康复|理疗|针灸|推拿|按摩|中医调理": "康复科/中医科",
    "营养咨询|饮食指导|减肥营养|疾病营养": "营养科",
    "疼痛管理|慢性疼痛|癌痛|术后疼痛": "疼痛科",
    "过敏|食物过敏|药物过敏|过敏性休克": "变态反应科/皮肤科",
    "艾滋病|HIV|梅毒|淋病|性病": "感染科/皮肤科",
    "肝炎|乙肝|丙肝|肝硬化|肝癌": "感染科/肝病科",
    "结核病|肺结核|骨结核|肠结核": "感染科/呼吸科",
    "狂犬疫苗|动物咬伤|蛇咬伤|蜂蜇伤": "急诊科/预防保健科",

    # 急重症
    "胸痛呼吸困难|剧烈胸痛|心梗症状|心绞痛": "急诊科/心血管内科",
    "剧烈腹痛|急腹症|阑尾炎|肠梗阻|胰腺炎": "急诊科/普外科",
    "高热不退|持续高热|败血症|感染性休克": "急诊科/感染科",
    "意识不清|昏迷|晕厥|抽搐|癫痫": "急诊科/神经内科",
    "食物中毒|药物中毒|化学中毒|农药中毒": "急诊科",
    "溺水|电击|窒息|中暑|冻伤": "急诊科",
}


class MedicalAssistant:
    def __init__(self,checkpoint_path="./final_model"):
        """初始化医疗助手"""
        self.checkpoint_path = "./final_model"
        self.device, self.dtype = self._select_device_and_dtype()
        self.model = None
        self.tokenizer = None
        self.conversation_history = []
        self.loaded_hospitals = {} #动态加载医疗数据
        
    def _select_device_and_dtype(self):
        """选择设备和数据类型"""
        if torch.cuda.is_available():
            try:
                major, _ = torch.cuda.get_device_capability()
                if major >= 12:
                    raise RuntimeError("Unsupported CUDA capability for current PyTorch")
                _ = torch.zeros(1, device="cuda")
                return "cuda", torch.float16
            except Exception:
                pass
        return "cpu", torch.float32
    
    def load_model(self):
        """加载模型和分词器"""
        print("正在加载医疗助手模型...")
        
        # 检查路径是否存在
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"模型路径不存在: {self.checkpoint_path}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint_path, 
            use_fast=False, #慢性能分词
            trust_remote_code=True,#远程仓库
            local_files_only=True  # 只使用本地文件
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path, 
            torch_dtype=self.dtype,
            local_files_only=True  # 只使用本地文件
        )
        self.model.to(self.device)
        self.model.eval()
        
        print(f"模型加载完成！使用设备: {self.device}")
    
    def predict(self, messages, max_new_tokens=512):
        """执行预测"""
        model_device = next(self.model.parameters()).device#next从可迭代对象中获取下一个元素
        text = self.tokenizer.apply_chat_template(#自动将对话列表转换为模型需要的输入格式。
            messages,
            tokenize=False,
            add_generation_prompt=True,#控制是否在对话末尾添加"让模型开始生成回复"的提示
        )
        inputs = self.tokenizer([text], return_tensors="pt")
        input_ids = inputs.input_ids.to(model_device)
        # 意思是：如果 B 为真，attention_mask = A，否则 attention_mask = C
        attention_mask = inputs.attention_mask.to(model_device) if hasattr(inputs, "attention_mask") else None

        generated = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )

        # 只解码新生成部分
        new_tokens = generated[:, input_ids.shape[1]:]#第一个：取所有，input_ids.shape[1]:表示从这个后边的维度开始取
        response = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]


        return response

    # ✅ 智能导诊 + 自动收录新词 + 永久保存
    def smart_guidance_enhanced(self,symptom,city="北京"):
        print(f"\n症状分析:【{symptom}】")
        print("="*50)

        #查找科室（增强匹配逻辑）
        matched_departments=[]
        for pattern,dept in HOSPITAL_DEPARTMENTS.items():
            keywords = pattern.split("|")
            for keyword in keywords:
                if keyword in symptom or symptom in keyword:
                    if dept not in matched_departments:
                        matched_departments.append(dept)

        #如果没有找到。尝试模糊匹配
        if not matched_departments:
            for pattern,dept in HOSPITAL_DEPARTMENTS.items():
                keywords = pattern.split("|")
                for keyword in keywords:
                    if len(keyword) >= 2 and (keyword in symptom or symptom in keyword):
                        if dept not in matched_departments:
                            matched_departments.append(dept)
        #确定最终科室
        if matched_departments:
            department = matched_departments[0] #取第一个匹配的
            if len(matched_departments) > 1:
                print(f"⚠️  找到多个匹配科室: {', '.join(matched_departments)}")
                print(f"✅ 推荐首选: 【{department}】")
                print("🔍 其他可能: " + " | ".join(matched_departments[1:3]))

        else:
            department = "全科/普通内科"
            print("⚠️  未精确匹配，建议挂【全科/普通内科】进行初步筛查")


        # 自动收录新词
        add_new = None
        if not matched_departments:
            add_new = input(f"\n⚠️ 未识别症状【{symptom}】，是否收录？(y/n)：").strip().lower()
            if add_new == "y":
                new_dept = input("请输入对应科室：").strip()
                if new_dept:
                    try:
                        with open("new_symptoms.json", "r", encoding="utf-8") as f:
                            existing_data = json.load(f)
                    except:
                        existing_data = {}

                    existing_data[symptom] = new_dept
                    with open("new_symptoms.json", "w", encoding="utf-8") as f:
                        json.dump(existing_data, f, ensure_ascii=False, indent=2)
                    print(f"✅ 已永久收录: {symptom} → {new_dept}")
                    department = new_dept

        # 医院推荐
        if city is None or city =="":
            city = input("请输入所在城市: ").strip() or "北京"

        result = {
            "symptom": symptom,
            "recommended_department": department,
            "matched_departments": matched_departments,
            "city": city,
            "recommended_hospitals": [],
            "advice": ""
        }

        #加载医院数据
        all_hospitals = HOSPITAL_DATABASE.copy()
        try:
            with open ("new_hospitals.json", "r", encoding="utf-8") as f:
                custom_hospitals = json.load(f)
                all_hospitals.update(custom_hospitals)
        except:
            pass

        if city not in all_hospitals:
            print(f"\n❌ 暂无 {city} 的医院数据")
            if add_city == "y":
                self.add_city_hospitals(city)
                all_hospitals[city] =self.loaded_hospitals.get(city, {})

        if city in all_hospitals:
            #查找具体科室
            dept_found = False
            for dept_pattern in all_hospitals[city].keys():
                if department in dept_pattern or dept_pattern in department:
                    result["recommended_hospitals"]= all_hospitals[city][dept_pattern]
                    dept_found = True
                    break
            #如果没有找到具体科室，找综合医院
            if not dept_found and "综合医院" in all_hospitals[city]:
                result["recommended_hospitals"] = all_hospitals[city]["综合医院"]
                result["advice"] += f"({city}暂无该专科医院，推荐综合医院"

            #如果连综合医院都没有
            if not result["recommended_hospitals"] and "神经内科" in all_hospitals[city]:
                result["recommended_hospitals"] = all_hospitals[city]["神经内科"] #默认神经内科
                result["advice"] +=f"(推荐神经内科相关医院）"

            #生成建议文本
            advice = f"""
            📋 智能导诊报告
========================================
👤 症状描述: {symptom}
🏥 推荐科室: 【{department}】
📍 所在城市: {city}
🏆 推荐医院:
"""
            if result["recommended_hospitals"]:
               for i,hospital in enumerate(result["recommended_hospitals"][:5],1):#最多5家
                   advice +=f"{i}. {hospital}\n"
            else:
                advice += "暂无该城市/科室的医院数据\n"
            advice += f"""
💡 就医建议:
1. 可先挂【{department}】进行初步诊断
2. 携带以往病历和检查报告
3. 建议提前通过医院官网/APP预约挂号
4. 急诊情况请直接前往医院急诊科
{result.get('advice', '')}
========================================
"""
            return advice

    def add_city_hospitals(self, city):
        """添加城市医院数据"""
        print(f"\n📝 正在添加 {city} 的医院数据")
        self.loaded_hospitals[city] = {}

        while True:
            dept = input("输入科室名称（输入'完成'结束): ").strip()
            if dept == "完成":
                break
            hospitals_input = input(f"输入{dept}的医院（逗号分隔）: ").strip()
            hospitals = [h.strip() for h in hospitals_input.split(",") if h.strip()]

            if hospitals:
                self.loaded_hospitals[city][dept] = hospitals
                print(f"✅ 已添加 {dept}: {', '.join(hospitals)}")
        #保存到文件
        try:
            with open ("new_hospitals.json", "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except:
            existing_data = {}
        existing_data[dept] = self.loaded_hospitals[city]
        with open("new_hospitals.json", "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

        print(f"✅ {city} 医院数据已保存")


    # 体检分析
    def analyze_lab_data_enhanced(self, data):
        """增强版体检分析"""
        result = '\n📊 体检指标分析报告\n' + '=' * 50 + '\n'
        abnormalities = []

        for key,value in data.items():
            if key not in LAB_REFERENCE or not value or value =="0":
                continue
            ref = LAB_REFERENCE[key]
            try:
                val = float(value)
                if "normal" in ref:
                    lo,hi = ref["normal"]
                    if val< lo:
                        status = "❌ 偏低"
                        abnormalities.append(f"{key}偏低 ({value})")
                    elif val> hi:
                        status = "❌ 偏高"
                        abnormalities.append(f"{key}偏高 ({value})")
                    else:
                        status = "✅ 正常"

                    result += f"{key:15}: {value:>8} → {status}\n"
            except ValueError:
                result +=f"{key:15}: {value:>8} → ⚠️  无效数值\n"
        result += '=' * 50 + '\n'

        #添加综合建议
        if abnormalities:
            result += "\n⚠️ 异常指标提醒:\n"
            for ab in abnormalities:
                result += f" * {ab}\n"

            result += "\n💡 就医建议:\n"
            if "血压" in str(abnormalities):
                result += "  • 血压异常 → 建议心血管内科就诊\n"
            if "血糖" in str(abnormalities):
                result += "  • 血糖异常 → 建议内分泌科就诊\n"
            if "血脂" in str(abnormalities) or "胆固醇" in str(abnormalities):
                result += "  • 血脂异常 → 建议心内科/内分泌科就诊\n"
            if "尿酸" in str(abnormalities):
                result += "  • 尿酸偏高 → 建议风湿免疫科/肾内科就诊\n"
            if "肝功能" in str(abnormalities) or "转氨酶" in str(abnormalities):
                result += "  • 肝功能异常 → 建议消化内科/肝病科就诊\n"
            if "肾功能" in str(abnormalities) or "肌酐" in str(abnormalities):
                result += "  • 肾功能异常 → 建议肾内科就诊\n"

            result += "\n🔍 建议:\n"
            result += "  1. 1-2周后复查异常指标\n"
            result += "  2. 如有症状，及时就医\n"
            result += "  3. 改善生活方式：均衡饮食、适度运动\n"
        else:
            result += "\n✅ 所有指标正常，继续保持健康生活方式！\n"

        return result


    def ask_question(self, question, scenario_type="diagnosis", max_tokens=512):
        if scenario_type not in MEDICAL_PROMPTS: scenario_type = "diagnosis"
        messages = [{"role": "system", "content": MEDICAL_PROMPTS[scenario_type]},
                    {"role": "user", "content": question}]
        response = self.predict(messages, max_new_tokens=max_tokens)
        self.conversation_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scenario": scenario_type, "question": question, "response": response
        })
        return response

    def show_scenarios(self):
        print("\n🏥 医疗助手场景：")
        print("=" * 50)
        for k, v in MEDICAL_SCENARIOS.items(): print(f"{k:2}. {v}")
        print("=" * 50)

    def interactive_mode(self):
        print("\n🤖 医疗助手已启动！输入 quit 退出")
        while True:
            try:
                self.show_scenarios()
                choice = input("\n选择场景 (1-12)：").strip()

                if choice.lower() in ['quit', 'exit', 'q']:
                    print("👋 感谢使用医疗助手！")
                    break

                if choice not in MEDICAL_SCENARIOS:
                    print("❌ 无效选择,请重新输入")
                    continue

                if choice == "11":
                    symptom= input("\n🤒 请输入症状描述: ").strip()
                    if not symptom:
                        print("❌ 症状不能为空")
                        continue

                    city = input("📍 请输入所在城市(留空则默认北京): ").strip()
                    city = city or "北京"

                    print("\n🔍 正在分析症状并推荐医院...")
                    advice = self.smart_guidance_enhanced(symptom,city)
                    print(advice)
                    continue

                if choice == "12":
                    print("\n📊 体检指标分析")
                    print("=" * 40)
                    data = {}
                    items = {
                        ("血压(高压)", "高压(mmHg): "),
                        ("血压(低压)", "低压(mmHg): "),
                        ("空腹血糖", "空腹血糖(mmol/L): "),
                        ("甘油三酯", "甘油三酯(mmol/L): "),
                        ("总胆固醇", "总胆固醇(mmol/L): "),
                        ("尿酸", "尿酸(μmol/L): "),
                    }

                    for key,prompt in items:
                        val = input(prompt).strip()
                        if val:
                            data[key] = val
                    if data:
                        print("\n📈 正在分析...")
                        result = self.analyze_lab_data_enhanced(data)
                        print(result)

                    else:
                        print("❌ 未输入任何指标")
                    continue

                #其他医疗咨询
                scenario_keys = list(MEDICAL_SCENARIOS.keys())
                if int(choice) <= len(scenario_keys):
                    scenario = scenario_keys[int(choice)-1]
                    question = input(f"\n❓请输入{MEDICAL_SCENARIOS[choice]}问题: ").strip()

                    if not question:
                        print("❌ 问题不能为空")
                        continue

                    print(f"\n🔄 正在分析【{MEDICAL_SCENARIOS[choice]}】问题...")
                    start_time = time.time()

                    response = self.ask_question(question, scenario, max_tokens=512)

                    elapsed_time = time.time() - start_time
                    print(f"\n💡 医疗助手回答 (耗时: {elapsed_time:.2f}秒):")
                    print("=" * 60)
                    print(response)
                    print("=" * 60)

                else:
                    print("❌ 无效的场景编号")


                #询问是否继续
                continue_choice = input("\n是否继续咨询？（y/n): ").strip().lower()
                if continue_choice in ["n","no","否"]:
                    print("👋 感谢使用医疗助手！")
                    break

            except KeyboardInterrupt:
                print("\n\n👋 感谢使用医疗助手！")
                break
            except Exception as e:
                print(f"❌ 发生错误: {str(e)}")
                continue

def main():
    parser = argparse.ArgumentParser(description="医疗助手 - 智能医疗咨询系统")
    parser.add_argument("--checkpoint", "-c", default="./final_model",
                        help="模型检查点路径")
    parser.add_argument("--max-tokens", "-m", type=int, default=512,  # ✅ 默认 512
                        help="最大生成token数（默认: 512）")
    parser.add_argument("--symptom", "-s", type=str,
                        help="直接输入症状进行导诊")
    parser.add_argument("--city", type=str, default="北京",
                        help="所在城市（默认: 北京）")


    args = parser.parse_args()

    assistant = MedicalAssistant(args.checkpoint)
    assistant.load_model()

    # 命令行直接导诊
    if args.symptom:
        advice = assistant.smart_guidance_enhanced(args.symptom, args.city)
        print(advice)
    else:
        # 交互模式
        assistant.interactive_mode()

if __name__ == "__main__":
    main()
