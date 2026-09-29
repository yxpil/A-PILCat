# -*- coding: utf-8 -*-
"""每个贴纸的专属重绘提示词 —— 按贴纸语义逐条单独写，不是通用套话。

风格前缀/后缀统一（贴纸画风），中间主体按该贴纸的动作/表情/道具单独描述。
配合 ControlNet lineart_anime（锁线稿结构）+ img2img（重画细节）使用。
"""

STYLE_PRE = ("masterpiece, best quality, very aesthetic, chibi anime sticker, "
             "clean bold lineart, flat cel shading, vibrant colors, ")
STYLE_SUF = ", simple background, full body, centered, high detail"

NEG = ("lowres, worst quality, low quality, blurry, jpeg artifacts, watermark, "
       "signature, username, text, logo, realistic, 3d render, photorealistic, "
       "bad anatomy, bad hands, missing fingers, extra digits, extra limbs, "
       "disfigured, deformed, messy lineart, rough sketch, dark, gloomy, "
       "motion blur, oversaturated, grayscale")

# ---------------------------------------------------------------- 逐条提示词
BODY = {
    # ---- A~F ----
    "打call": "holding two glowing light sticks up high, cheering excitedly, sparkling eyes, open mouth smile, energetic jumping pose",
    "DNA动了": "extremely shocked excited face, both hands on cheeks, hearts and sparkles floating around, thrilled reaction",
    "WOW": "amazed wide open eyes, mouth open in astonishment, both hands raised, star sparkles around head, surprised pose",
    "emo": "sad gloomy face, sitting with knees hugged to chest, dark cloud over head, teardrop, drooping eyes",
    "yyds": "proud confident pose, thumbs up with both hands, shining aura, worshipping sparkle background, smug smile",
    "不想动": "lying face down lazily, limp arms spread on ground, sleepy half closed eyes, melting relaxed posture",
    "不慌": "calm composed expression, waving hand dismissively, slight confident smile, relaxed steady posture",
    "不服": "pouting angry face, crossed arms, turning head away, small angry vein mark, defiant expression",
    "不知道": "tilting head with blank confused face, both hands spread out, question mark above head, innocent shrug",
    "举牌": "holding a blank white sign board up with both hands, cheerful smile, standing pose",
    "交给我": "confident determined face, thumb pointing at own chest, sparkling eyes, reliable dependable pose",
    "佛系": "calm peaceful serene face, sitting cross legged with hands together, closed eyes, lotus flower and halo around",
    "你真厉害": "clapping hands with amazed admiring face, sparkling eyes, small stars around head, impressed pose",
    "你说啥": "shocked confused face, hand cupped behind ear, leaning forward to listen, question marks around head",
    "你过来": "beckoning with index finger curled, teasing confident smirk, leaning forward, playful challenging look",
    "傲娇": "blushing turned away face with pouting lips, arms crossed, tsundere embarrassed expression, small hearts",
    "充电中": "sitting plugged into a charging cable, sleepy content face, battery icon glowing above head, resting pose",
    "冒泡": "peeking up from bubbles, only head and hands visible, cute shy smile, round bubble shapes floating",
    "减肥": "crying while stepping on a weighing scale, dramatic tears, shocked at the number, diet despair pose",
    "别卷了": "tired pleading face, waving hand to stop, dark circles under eyes, exhausted drooping posture",
    "别吵": "covering both ears with hands, frowning annoyed face, angry vein mark, irritated expression",
    "别烦我": "turning back with annoyed face, hand pushing away, angry puffing cheeks, leave me alone gesture",
    "别离开": "reaching out with both arms, teary pleading eyes, sad worried expression, clingy desperate pose",
    "别闹": "half smiling tolerant face, hand gently pushing away, slightly annoyed but amused expression",
    # ---- G~J ----
    "加油": "clenching both fists in front of chest, determined fired up eyes, sparkling fiery aura, cheering encouraging pose",
    "加班": "typing on a laptop with tired dead eyes, coffee cup and stack of documents on desk, overtime exhausted pose",
    "加鸡腿": "happy face holding a big fried chicken drumstick, drooling slightly, sparkling eyes, celebrating reward pose",
    "勿扰": "holding up a do not disturb sign, closed eyes, calm indifferent face, busy working gesture",
    "包在我身": "patting own chest confidently, proud grin, sparkling determined eyes, reliable dependable pose",
    "发呆中": "blank staring face, drooling slightly, unfocused eyes, floating thought bubbles, spacing out pose",
    "变身": "dramatic transformation pose with sparkling light, one arm raised, magical aura and stars swirling around",
    "只准想我": "pointing at viewer with demanding jealous face, puffed cheeks, small heart above head, possessive cute pose",
    "吃东西": "stuffing a big rice ball into mouth, puffed cheeks, happy squinted eyes, chopsticks in hand",
    "吃瓜": "holding a slice of watermelon while watching curiously, gossipy interested eyes, snacking pose",
    "听我说": "one hand raised palm out, serious talking face, open mouth speaking, attention demanding gesture",
    "哈哈哈": "laughing out loud with head tilted back, eyes squeezed shut, tears of joy, big open smile",
    "哦": "flat indifferent face, deadpan eyes, slight frown, unimpressed bored expression",
    "哭泣": "crying with big teardrops streaming from eyes, hands rubbing eyes, sad wailing face",
    "啊": "shocked open mouth, blank stunned eyes, realization dawning expression, hands slightly raised",
    "喜欢": "blushing shy face with hands on cheeks, sparkling heart eyes, small hearts floating around",
    "嗨": "waving hand cheerfully, big happy smile, energetic greeting pose, sparkle around",
    "嗯": "nodding with closed eyes and gentle smile, hands together, calm agreeable expression",
    "嘿嘿": "mischievous grin with narrowed sly eyes, hand covering mouth, scheming playful expression",
    "在吗": "peeking shyly around a corner, holding a phone, hopeful curious eyes, timid hello pose",
    "在干嘛": "curiously looking at a smartphone screen, tilted head, interested sparkling eyes, spying pose",
    "坐等": "sitting patiently with hands on knees, calm relaxed smile, waiting pose, small clock beside",
    "大无语": "deadpan shocked face, sweat drop on head, mouth in flat line, speechless overwhelmed expression",
    "大笑": "big joyful laughter, eyes closed in crescents, hands on belly, wide open happy smile",
    "头秃": "clutching head with both hands, falling hair strands, shocked crying face, stress and despair pose",
    "好不好": "pleading cute face with big shiny eyes, hands clasped together, begging hopeful pose",
    "好家伙": "shocked sarcastic face, eyebrow raised, hand on forehead, exaggerated surprise reaction",
    "好巧": "surprised happy face, pointing at viewer, sparkling eyes, coincidental meeting pose",
    "好气哦": "pouting angry face with puffed cheeks, crossed arms, steam puffing from head, cute irritation",
    "委屈": "teary eyes holding back tears, quivering lips, sad pouty face, wronged expression",
    "委屈巴巴": "extremely teary pleading eyes, trembling pout, sad small posture, wronged pitiful face",
    "嫌弃": "disgusted face leaning away, squinting eyes, one eyebrow raised, judgmental displeased expression",
    "安排": "confident smirk with finger guns, cool sunglasses, sparkle, everything under control pose",
    "害羞": "blushing deeply with hands covering face, shy eyes peeking through fingers, steam from head",
    "对不起": "bowing deeply with hands together in apology, sad guilty face, sweat drop, sorry gesture",
    "对的对的": "nodding eagerly with both thumbs up, happy agreeing face, sparkling eyes, yes yes pose",
    "尊嘟假嘟": "cute confused tilted head, sparkling big eyes, pouting lips, really question mark above head",
    "小事一桩": "waving hand carelessly with smug confident face, closed eyes smile, easy peasy relaxed pose",
    "尴尬": "awkward forced smile with sweat drop, averting eyes, stiff posture, embarrassed nervous look",
    "已读乱回": "deadpan face typing nonsense on phone, blank eyes, tired unbothered expression, chat bubble above",
    "干杯": "raising a glass of drink to toast, happy grin, sparkling eyes, clinking celebration pose",
    "庆祝": "throwing confetti with both hands, big joyful smile, party popper, festive celebration pose",
    "开心": "big happy smile with sparkling eyes, both hands raised in joy, energetic cheerful pose",
    "得意": "smug proud face with hands on hips, chin up, confident smirk, sparkle around head",
    "心累": "exhausted sigh with slumped shoulders, tired half closed eyes, sweat drop, drained pose",
    "怎么这样": "shocked betrayed face, mouth open in disbelief, hands raised, confused complaint pose",
    "思考": "hand on chin in deep thought, narrowed serious eyes, thinking pose, light bulb and question marks above",
    "恰饭": "eating happily with chopsticks and rice bowl, puffed cheeks, satisfied content expression",
    "惊了": "extremely shocked wide eyes, jaw dropped, hands raised in surprise, exclamation mark above head",
    "惊讶": "surprised face with wide sparkling eyes, small open mouth, hands slightly raised, shock marks",
    "想哭": "holding back tears with watery eyes, trembling pout, sad face about to cry",
    "想开了": "relieved peaceful smile with closed eyes, hands clasped, light shining from behind, enlightenment pose",
    "想抱抱": "reaching out with both arms, teary hopeful eyes, asking for a hug, lonely cute pose",
    "愤怒": "furious angry face with flaming aura, gritted teeth, clenched fists, angry vein marks",
    "我不信": "doubtful skeptical face, squinting eyes, hand raised palm out, shaking head in disbelief",
    "我可以": "determined fired up face with clenched fist, sparkling confident eyes, ready to go pose",
    "我酸了": "jealous sour face, pouting, looking sideways at lemons, envious expression with sweat drop",
    "我错了": "repentant kneeling pose with hands together, teary apologetic eyes, sad guilty face",
    "打卡": "holding a checklist and stamping it, cheerful smile, daily habit completion pose",
    "打滚": "rolling on the floor playfully, legs kicking up, happy laughing face, silly cute pose",
    "抱抱我": "sitting with arms open wide, teary pleading eyes, wanting a hug, lonely expression",
    "拒绝": "crossing both arms in an X shape, serious firm face, refusing gesture, stern expression",
    "搞快点": "impatiently tapping foot and leaning forward, urging face, hurry up gesture, clock beside",
    "摸头": "being patted on the head, happy closed eyes smile, blushing cheeks, receiving headpat pose",
    "摸鱼": "secretly slacking with a fishing rod at a desk, sneaky grin, pretending to work pose",
    "撒花": "throwing flower petals up in celebration, joyful smile, sparkling eyes, festive pose",
    "收到": "saluting with a cheerful serious face, thumbs up, confirming pose, sparkle",
    "救命": "screaming for help with hands raised, panicked expression, sweat drops, distressed pose",
    "无语": "speechless deadpan face, flat mouth line, blank eyes, sweat drop, unimpressed pose",
    "无语住了": "completely speechless with blank stare, mouth slightly open, tired eyes, dumbfounded pose",
    "暗中观察": "peeking from behind a wall, only eyes visible, curious suspicious look, watching pose",
    "有被笑到": "covering mouth while laughing, amused sparkling eyes, trying not to laugh, entertained pose",
    "柠檬精": "sour jealous face squeezing a lemon, pouting, green envy aura, envious expression",
    "栓Q": "forced smile with tears in eyes, thanking reluctantly, speechless gratitude pose, sweat drop",
    "比心": "making a finger heart with both hands, happy sparkling eyes, loving cute pose",
    "求带": "clasping hands together begging, teary hopeful eyes, please take me along pose",
    "沉默": "silent blank face with a small x mark over mouth, empty eyes, quiet awkward pause",
    "没毛病": "confident nodding with thumbs up, satisfied smile, agreement pose, sparkle",
    "没问题": "making an ok hand sign, cheerful confident smile, reliable pose, sparkle",
    "淡定": "sipping tea calmly with closed eyes, unbothered composed face, serene relaxing pose",
    "温暖": "gentle warm smile with closed eyes, holding a hot drink, cozy scarf, soft glowing aura",
    "溜了": "running away quickly with motion lines, sweating nervously, escaping pose, back turned",
    "点赞": "giving a big thumbs up with a happy wink, smiling face, approval pose, sparkle",
    "然后呢": "tilting head waiting expectantly, raised eyebrow, arms crossed, curious impatient expression",
    "爱了爱了": "heart eyes with hands on cheeks, deeply in love face, hearts floating everywhere",
    "牛逼": "impressed shocked face with thumbs up, sparkling eyes, amazed admiring expression",
    "略略略": "sticking tongue out playfully, pulling down lower eyelid, teasing silly face",
    "看不懂": "squinting at a document with confused face, question marks around, puzzled expression",
    "看戏": "eating popcorn while watching curiously, amused gossipy eyes, entertained spectator pose",
    "看报纸": "holding a newspaper wide open, serious calm face, reading pose, focused eyes",
    "真的": "surprised hopeful face, leaning forward, wide sparkling eyes, really question expression",
    "真的假的": "shocked disbelief face, jaw dropped, hands raised, doubt and surprise marks around",
    "真香": "blushing satisfied face eating delicious food, sparkling eyes, aroma swirls, delighted expression",
    "睡着": "sleeping peacefully with zzz symbols, drooling slightly, relaxed lying pose, closed eyes",
    "知道了": "nodding calmly with a slight smile, one hand raised, understanding gesture, sparkle",
    "码字": "typing intensively on a laptop, focused eyes, sweat drop, writer concentration pose",
    "破防": "emotionally overwhelmed crying face, hand on chest, shattered heart pieces, teary eyes",
    "磕到了": "blushing hand over heart, deeply moved happy face, heart sparkles, shipping excitement pose",
    "社恐": "hiding behind a wall nervously, sweating, timid scared eyes, avoiding people pose",
    "社牛": "confidently chatting with a big grin, sparkling aura, outgoing energetic pose, microphone",
    "祈祷": "hands clasped in prayer, closed eyes, hopeful serene face, glowing light above",
    "稳住": "holding steady with both hands out, focused determined face, balancing pose, sweat drop",
    "笑死": "dying of laughter, rolling with closed eyes and tears, big open smile, hands on belly",
    "等等我": "running while reaching out one hand, panicked hopeful face, motion lines, chasing pose",
    "算你狠": "grudging annoyed face, pointing at viewer, half smiling half frowning, admitting defeat pose",
    "美滋滋": "blissful happy face with closed eyes, small flowers around, content satisfied pose",
    "耶": "jumping with both arms raised in a V, big joyful smile, sparkling eyes, victory pose",
    "胡说": "denying with both hands waving, frowning face, confused angry expression, no no gesture",
    "膜拜": "kneeling in worship with hands raised, sparkling admiring eyes, respect pose, glow",
    "记仇": "writing in a small notebook with a dark grudge face, side glancing, holding a grudge pose",
    "请收下": "bowing while offering a gift box with both hands, polite sincere smile, respect pose",
    "谁懂啊": "spreading arms in despair, teary exasperated face, hopeless expression, asking for sympathy",
    "谢谢": "bowing politely with hands together, grateful warm smile, thanking pose, sparkle",
    "赢了": "raising a trophy proudly, victorious confident grin, sparkling eyes, winning celebration pose",
    "起哄": "cheering and whistling with hands cupped at mouth, excited teasing grin, riling up pose",
    "跪了": "kneeling on the floor defeated, hands on ground, shocked despairing face, admitting defeat pose",
    "躺平": "lying flat on the floor relaxed, closed content eyes, hands behind head, giving up pose",
    "这合理吗": "shocked doubtful face holding a document, displeased expression, questioning pose, sweat drop",
    "迷路了": "looking around confused with a map, teary worried eyes, lost expression, question marks around",
    "退退退": "pushing both palms forward to ward off, frowning serious face, back off gesture, motion lines",
    "送礼": "holding out a wrapped gift box with both hands, happy warm smile, offering pose, sparkle",
    "闪人": "running away fast with motion blur lines, determined grin, escaping quickly pose",
    "隐身": "fading into transparency with only outline visible, nervous smile, hiding pose, dashed lines",
    "难过": "sad downcast face with teary eyes, drooping head, small rain cloud above, gloomy expression",
    "震惊": "extremely shocked wide eyes and open mouth, hands raised to cheeks, lightning marks, thunderstruck pose",
    "骗你的": "mischievous tongue out grin, winking, one hand raised, teasing trickster pose",
    "魔法少女": "magical girl transformation pose with wand raised, sparkling star aura, cute frilly outfit, glowing magic circle",
}


def prompt_of(kw):
    """返回 (prompt, negative)。找不到的关键词退回通用表情包描述。"""
    body = BODY.get(kw)
    if not body:
        # 兜底：关键词本身作为语义线索（中文词模型难懂，用通用可爱表情描述）
        body = "cute chibi character with a playful expressive face, lively pose"
    return STYLE_PRE + body + STYLE_SUF, NEG


if __name__ == "__main__":
    import sys
    kws = [l.strip() for l in open("_kw_list.txt", encoding="utf-8") if l.strip()]
    miss = [k for k in kws if k not in BODY]
    print(f"贴纸 {len(kws)} 个，映射 {len(BODY)} 条，缺失 {len(miss)}: {miss}")
    for k in kws[:3]:
        print(f"\n[{k}]\n{prompt_of(k)[0]}")
