# -*- coding: utf-8 -*-
"""
表情 -> 语义动作映射（SVD 视频生成用）。

区别于 motion_map.py 的「整体位移近似」：这里给每个表情一段**具体的动作描述**，
由 SVD 真正渲染出来（举手、张嘴、跺脚…），而不是把原图整体挪一挪。
描述尽量贴近中文原意且是**可见的肢体/口型动作**，这样成片才有"意义"。
"""

# 关键字片段 -> 动作描述（英文，SVD 理解最好）
RULES = [
    # 打call / 欢呼 / 加油  —— 挥手打call
    (("打call", "加油", "冲鸭", "奥利给", "燃", "干巴", "点赞", "牛逼", "yyds", "绝了"),
     "both arms raised high waving energetically, jumping up and down, bright smiling face"),
    # 肌肉 /  Dollar / 帅
    (("肌肉", "dollar", " dollar", "帅", "酷", "举铁", "健身"),
     "one arm flexed showing bicep, head tilted, confident smirk"),
    # 吃饭 / 零食 / 喝
    (("吃饭", "吃", "零食", "奶茶", "喝", "汉堡", "面", "寿司", "肉"),
     "raising food towards mouth, mouth chewing, swallowing"),
    # 睡觉 / 休息 / 躺
    (("睡", "困", "午休", "打盹", "躺", "懒"),
     "eyes closed sleeping calmly, gentle breathing, head resting low"),
    # 生气 / 怒
    (("生气", "怒", "气死", "愤", "爆炸", "滚"),
     "angry furled eyebrows, mouth frowning, stomping one foot."),
    # 哭 / 泪
    (("哭", "泪", "委屈", "难受", "伤心"),
     "tears flowing from eyes, mouth open sobbing, head tilted down"),
    # 惊讶 / WOW / 震惊
    (("wow", "Wow", "WOW", "震惊", "惊讶", "纳尼", "啊", "不好", "救命"),
     "eyes wide open in surprise, mouth agape, both hands near face, leaning back"),
    # 疑问 / 不懂
    (("？", "?", "疑问", "不懂", "什么鬼", "啊这", "迷惑"),
     "one eyebrow raised, head tilting questioningly, mouth slightly open"),
    # 无语 / 死线 / 已读乱回
    (("无语", "已读", "乱回", "栓q", "6", "服了", "离谱", "抽象"),
     "rolling eyes, dry flat expression, one hand waving dismissively, slight head shake"),
    # 捂脸 / 害羞 / 不好意思
    (("捂脸", "害羞", "不好意思", "羞", "脸红", "社死", "尴尬"),
     "both hands covering flushed cheeks, looking down shyly, blushing"),
    # 思考 / 纠结
    (("思考", "纠结", "想", "难道", "其实"),
     "one hand scratching chin, looking to the side while thinking, thoughtful"),
    # 看戏 / 吃瓜 / 围观
    (("看戏", "吃瓜", "围观", "监控", "spy", "偷看"),
     "leaning back with arms crossed, watching curiously, one eyebrow raised"),
    # 充电 / 能量 / 复活
    (("充电", "能量", "复活", "满血", "启动", "开机", "加载"),
     "eyes glowing with sparkling energy, arms slightly out, glowing aura pulsing"),
    # 不慌 / 淡定 / 稳
    (("不慌", "淡定", "稳", "从容", "慢慢", "放心"),
     "calm relaxed expression, gentle slow breathing, faint confident smile"),
    # 答应 / 好 / 可以 / 收到
    (("好的", "收到", "一定", "行", "好耶", "没问题", "ok", "OK"),
     "nodding head once with a determined smile, one hand raised giving a thumbs up"),
    # 转圈 / 展示 / 转
    (("转", "展示", "翻", "表演"),
     "spinning around gracefully with arms swinging, happy expression"),
    # 招手 / 再见 / hi
    (("你好", "再见", "拜拜", "hi", "HI", "嗨", "招手"),
     "one hand waving hello goodbye with a friendly smile"),
    # 吐舌 / 调皮 / 鬼脸
    (("吐舌", "调皮", "鬼脸", "坏笑", "逗"),
     "sticking out tongue playfully, winking one eye, mischievous grin"),
    # 看手机 / 刷
    (("手机", "刷", "短视频", "看", "平板", "电脑"),
     "looking down at phone screen, thumb swiping, slight smile"),
    # olleh / 比心 / 爱心
    (("比心", "爱心", "心", "喜欢", "爱"),
     "both hands forming a heart shape near chest, blushing happy smile"),
    # 睡觉打呼 / 流口水
    (("喝水", "渴"), "raising hand to mouth taking a sip, neck moving"),
    # 原地踏步 / 跑
    (("跑", "赶", "迟到", "快"), "running in place, legs moving, panting lightly"),
    # 坐 / 瘫
    (("瘫", "坐", "躺平", "放弃"), "slumping down relaxed, eyes half closed, exhaling"),
]

DEFAULT = "gentle head swaying slowly side to side, slight smile, soft breathing"

# 镜头调度（SVD 生成时叠加，让动作更有戏）
CAM = {
    "打call": "Dynamic handheld shot, energetic, low angle.",
}


def action_of(keyword: str) -> str:
    """关键字 -> 具体动作英文描述。"""
    k = keyword or ""
    for keys, act in RULES:
        for t in keys:
            if t and t in k:
                return act
    return DEFAULT


if __name__ == "__main__":
    import sys
    for kw in (sys.argv[1:] or ["打call", "吃饭", "睡觉", "生气", "wow", "无语", "思考",
                                 "看戏", "充电", "不慌", "比心", "吐舌"]):
        print(f"{kw}  ->  {action_of(kw)}")
