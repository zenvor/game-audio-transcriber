"""
classifier.py — 用 CLAP 对纯音效文件进行零样本分类标注
基于 LAION CLAP，支持自定义标签，无需训练数据。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

_model = None
_text_embeddings = None

# ── 王者荣耀音效分类标签 ──────────────────────────────────
# 每项: (CLAP 匹配用英文描述, 中文显示名)
LABELS = [
    # ═══════════════════════════════════════════
    # 近战武器 / Melee Weapons
    # ═══════════════════════════════════════════
    ("sword slashing through air", "剑挥砍"),
    ("sword stabbing and piercing", "剑刺击"),
    ("two swords clashing metal on metal", "刀剑碰撞"),
    ("heavy axe chopping swing", "斧头劈砍"),
    ("spear thrusting and poking", "长枪刺击"),
    ("hammer smashing with heavy impact", "锤子重击"),
    ("dagger quick stabbing", "匕首快刺"),
    ("chain whip swinging", "锁链/鞭击"),
    ("fist punching impact", "拳击"),
    ("kick impact on body", "踢击"),
    ("staff or pole weapon swing", "棍棒挥击"),
    ("blade spinning in the air", "旋转刀刃"),

    # ═══════════════════════════════════════════
    # 远程武器 / Ranged Weapons
    # ═══════════════════════════════════════════
    ("bow drawing and arrow release", "弓箭射击"),
    ("crossbow bolt firing", "弩箭发射"),
    ("throwing knife or shuriken", "飞刀/暗器"),
    ("energy projectile launching", "能量弹发射"),
    ("bullet or gun shooting", "枪械射击"),
    ("boomerang throwing and spinning", "回旋镖"),

    # ═══════════════════════════════════════════
    # 命中与受击 / Hit & Impact
    # ═══════════════════════════════════════════
    ("weapon hitting flesh and body", "武器命中肉体"),
    ("weapon hitting metal armor", "武器命中金属"),
    ("critical hit with powerful impact and crunch", "暴击"),
    ("blunt impact thud on body", "钝击"),
    ("sharp slicing cut through flesh", "利刃切割"),
    ("projectile arrow hitting target", "远程命中"),
    ("multiple rapid hits combo attack", "连击"),
    ("body falling to the ground after death", "倒地"),
    ("bone cracking and breaking", "骨骼碎裂"),

    # ═══════════════════════════════════════════
    # 格挡与防御 / Block & Defense
    # ═══════════════════════════════════════════
    ("shield blocking a weapon attack with clang", "盾牌格挡"),
    ("parry deflecting sword attack", "武器招架"),
    ("magical barrier absorbing damage", "魔法护盾"),
    ("shield breaking and shattering", "护盾破碎"),
    ("armor absorbing heavy hit", "护甲吸收"),
    ("dodge and evasion quick movement", "闪避"),

    # ═══════════════════════════════════════════
    # 火系技能 / Fire Skills
    # ═══════════════════════════════════════════
    ("fireball launching with whoosh", "火球发射"),
    ("fire explosion and blast", "火焰爆炸"),
    ("continuous fire burning and crackling", "持续燃烧"),
    ("fire ember and spark particles", "火花飞溅"),
    ("flame pillar erupting from ground", "火柱喷发"),
    ("lava and molten bubbling", "岩浆翻涌"),

    # ═══════════════════════════════════════════
    # 冰系技能 / Ice Skills
    # ═══════════════════════════════════════════
    ("ice freezing and crystallizing", "冰冻结晶"),
    ("ice shattering and breaking apart", "冰碎裂"),
    ("blizzard snowstorm wind and ice", "暴风雪"),
    ("frost spreading on surface", "寒霜蔓延"),
    ("icicle falling and stabbing", "冰锥坠落"),

    # ═══════════════════════════════════════════
    # 雷电技能 / Lightning Skills
    # ═══════════════════════════════════════════
    ("lightning bolt striking with crack", "闪电劈击"),
    ("chain lightning arcing between targets", "链式闪电"),
    ("electric spark and static buzzing", "电火花"),
    ("thunder rumbling after lightning", "雷鸣"),
    ("electric field humming and crackling", "电场持续"),

    # ═══════════════════════════════════════════
    # 风系技能 / Wind Skills
    # ═══════════════════════════════════════════
    ("wind gust and air swoosh", "风刃"),
    ("tornado spinning wind vortex", "龙卷风"),
    ("air pressure shockwave blast", "气浪冲击"),
    ("gentle breeze flowing", "微风"),

    # ═══════════════════════════════════════════
    # 暗影与光明 / Dark & Light
    # ═══════════════════════════════════════════
    ("dark shadow energy pulse", "暗影脉冲"),
    ("void and darkness consuming", "虚空吞噬"),
    ("shadow teleport vanishing", "暗影传送"),
    ("holy light beam shining bright", "圣光照射"),
    ("divine purification and cleansing", "净化驱散"),
    ("light and dark energy colliding", "光暗交织"),

    # ═══════════════════════════════════════════
    # 土/自然系技能 / Earth & Nature
    # ═══════════════════════════════════════════
    ("earthquake ground shaking and rumble", "地震"),
    ("rock smashing and boulder impact", "岩石碎裂"),
    ("ground cracking and splitting open", "地面裂开"),
    ("vine and plant growing rapidly", "藤蔓生长"),
    ("poison acid dripping and sizzling", "毒液腐蚀"),
    ("sand and dust swirling", "沙尘旋转"),

    # ═══════════════════════════════════════════
    # 通用技能 / General Abilities
    # ═══════════════════════════════════════════
    ("energy charging up and powering", "能量蓄力"),
    ("energy burst release and explosion", "能量释放"),
    ("magical spell casting with mystical tone", "法术吟唱"),
    ("aura activation glowing around body", "光环激活"),
    ("buff power up sparkle gained", "增益buff"),
    ("debuff negative effect applied", "减益debuff"),
    ("healing magical chime and restoration", "治疗回复"),
    ("resurrection and revival with glow", "复活特效"),
    ("teleport blink instant dash", "闪现/位移"),
    ("summoning creature from portal", "召唤生物"),
    ("trap activation and trigger", "陷阱触发"),
    ("marking target with sigil", "标记目标"),
    ("stun electric shock crowd control", "眩晕控制"),
    ("silence and mute debuff applied", "沉默效果"),
    ("slow movement debuff freezing", "减速效果"),
    ("knockback pushing force impact", "击退效果"),
    ("pull and grab magnetic attraction", "牵引/抓取"),
    ("area of effect large explosion blast", "范围爆炸"),
    ("small explosion and pop", "小型爆炸"),

    # ═══════════════════════════════════════════
    # 地图与野怪 / Map & Monsters
    # ═══════════════════════════════════════════
    ("large boss monster roaring", "暴君/主宰怒吼"),
    ("dragon roar and epic creature cry", "龙吼"),
    ("jungle monster growling and attacking", "野怪攻击"),
    ("jungle monster dying and collapsing", "野怪死亡"),
    ("minion small creature marching", "小兵行军"),
    ("minion small creature dying", "小兵死亡"),
    ("tower shooting energy beam laser", "防御塔射击"),
    ("tower being destroyed and collapsing", "防御塔摧毁"),
    ("crystal base structure resonating", "水晶/基地"),
    ("healing plant or fruit pickup", "回复果实/植物"),
    ("speed boost zone activation", "加速区域"),
    ("river water splashing when crossing", "河道涉水"),

    # ═══════════════════════════════════════════
    # UI 与系统 / UI & System
    # ═══════════════════════════════════════════
    ("button click in menu interface", "界面按钮"),
    ("menu opening or panel sliding", "菜单打开"),
    ("item purchase with coin spending sound", "装备购买"),
    ("skill level up upgrade sound", "技能升级"),
    ("character level up with fanfare", "角色升级"),
    ("gold coins dropping and clinking", "金币获取"),
    ("countdown timer ticking clock", "倒计时"),
    ("notification alert ping bell", "信号提示"),
    ("error or invalid action buzzer", "操作失败"),
    ("match found queue pop alert", "匹配成功"),
    ("achievement or quest completion chime", "成就/任务完成"),
    ("scroll or page turning", "翻页"),
    ("inventory bag opening", "背包打开"),
    ("star rating or score tally", "评分/结算"),

    # ═══════════════════════════════════════════
    # 环境音效 / Ambient & Environment
    # ═══════════════════════════════════════════
    ("outdoor nature birds chirping and wind", "户外自然环境"),
    ("night time crickets and insects ambient", "夜间虫鸣"),
    ("water stream river flowing gently", "溪流水声"),
    ("ocean waves crashing on shore", "海浪拍岸"),
    ("waterfall rushing water", "瀑布"),
    ("rain falling on ground and leaves", "雨声"),
    ("thunderstorm with rain and thunder", "雷雨"),
    ("wind howling strong gust", "强风呼啸"),
    ("fire torch crackling ambient", "火把燃烧"),
    ("cave echo dripping water", "洞穴回声"),
    ("crowd audience cheering in arena", "观众欢呼"),
    ("battle ambience distant fighting", "远处战斗声"),
    ("mystical magical ambient humming", "魔法环境音"),
    ("mechanical gears and machinery running", "机械运转"),

    # ═══════════════════════════════════════════
    # 角色动作 / Character Actions
    # ═══════════════════════════════════════════
    ("footsteps walking on stone floor", "石地脚步"),
    ("footsteps walking on grass and dirt", "草地脚步"),
    ("footsteps splashing in shallow water", "水中脚步"),
    ("fast running footsteps", "奔跑"),
    ("jumping up into the air", "跳跃起跳"),
    ("landing on ground after jump", "落地"),
    ("rolling or tumbling on ground", "翻滚"),
    ("character recall teleporting back to base", "回城传送"),
    ("riding mount galloping horse", "骑乘/坐骑"),
    ("wings flapping and flying", "飞行/翅膀"),
    ("swimming in water splashing", "游泳"),
    ("climbing and grappling wall", "攀爬"),

    # ═══════════════════════════════════════════
    # 材质与物理 / Materials & Physics
    # ═══════════════════════════════════════════
    ("glass breaking and shattering", "玻璃碎裂"),
    ("wood breaking and splintering", "木材断裂"),
    ("metal clanging and ringing", "金属碰撞"),
    ("stone crumbling and debris falling", "石块崩碎"),
    ("cloth and fabric tearing", "布料撕裂"),
    ("rope or chain rattling", "绳索/锁链"),
    ("crystal resonating and chiming", "水晶共鸣"),
    ("liquid splashing and pouring", "液体泼溅"),
    ("smoke and steam hissing", "烟雾/蒸汽"),
    ("dust and debris scattering", "尘土飞扬"),

    # ═══════════════════════════════════════════
    # 特殊音效 / Special Effects
    # ═══════════════════════════════════════════
    ("portal opening with swirling energy", "传送门"),
    ("time slow motion effect", "时间减缓"),
    ("ghostly ethereal whooshing", "鬼魅/灵体"),
    ("musical instrument playing melody", "乐器演奏"),
    ("bell tolling or ringing", "钟声"),
    ("horn trumpet blowing signal", "号角"),
    ("drum beating rhythm", "鼓声"),
    ("whoosh fast object passing by", "快速掠过"),
    ("rising pitch tension building", "紧张感渐强"),
    ("impact boom cinematic hit", "电影级重击"),

    # ═══════════════════════════════════════════
    # 水系技能 / Water Skills
    # ═══════════════════════════════════════════
    ("water splash and wave impact", "水花冲击"),
    ("water bubble rising and popping", "水泡破裂"),
    ("underwater deep rumble and pressure", "深水压迫"),
    ("tidal wave surging forward", "潮汐涌动"),
    ("water whirlpool spinning", "水漩涡"),
    ("steam jet hissing from hot water", "蒸汽喷射"),

    # ═══════════════════════════════════════════
    # 更多近战细分 / More Melee Details
    # ═══════════════════════════════════════════
    ("scythe wide sweeping slash", "镰刀横扫"),
    ("claw scratching and ripping", "爪击撕裂"),
    ("fan weapon swift cutting wind", "扇击"),
    ("gauntlet powered fist slam", "铁拳重砸"),
    ("shield bash forward charge", "盾击冲撞"),
    ("weapon drawn unsheathing from scabbard", "拔刀出鞘"),
    ("weapon sheathing putting away", "收刀入鞘"),
    ("dual wielding rapid alternating strikes", "双持连斩"),

    # ═══════════════════════════════════════════
    # 更多远程细分 / More Ranged Details
    # ═══════════════════════════════════════════
    ("magic missile homing projectile", "追踪弹"),
    ("cannon heavy artillery firing", "炮击"),
    ("multiple arrows volley barrage", "箭雨齐射"),
    ("beam laser continuous shooting", "激光射线"),
    ("explosive grenade thrown and bouncing", "投掷爆弹"),
    ("projectile ricocheting and bouncing", "弹射反弹"),

    # ═══════════════════════════════════════════
    # 更多命中反馈 / More Hit Feedback
    # ═══════════════════════════════════════════
    ("last hit killing blow final strike", "致命一击"),
    ("glancing hit weak deflected blow", "偏斜轻击"),
    ("backstab attack from behind", "背刺"),
    ("counter attack reflected damage", "反击"),
    ("splash damage hitting multiple targets", "溅射伤害"),
    ("damage over time tick burning poison", "持续伤害"),

    # ═══════════════════════════════════════════
    # 状态变化 / Status Changes
    # ═══════════════════════════════════════════
    ("stealth invisibility activating", "隐身"),
    ("stealth breaking and revealing", "隐身破除"),
    ("invincibility golden shield activation", "无敌状态"),
    ("berserk rage power surge", "狂暴状态"),
    ("fear and terror dark debuff", "恐惧效果"),
    ("charm and confusion pink sparkle", "魅惑效果"),
    ("petrification turning to stone", "石化效果"),
    ("transformation shapeshifting morph", "变身"),
    ("shrinking becoming smaller", "缩小效果"),
    ("growing becoming larger giant", "巨大化"),

    # ═══════════════════════════════════════════
    # 更多地图交互 / More Map Interaction
    # ═══════════════════════════════════════════
    ("bush grass rustling entering hiding", "草丛进入"),
    ("wall terrain collision bumping", "撞墙"),
    ("door gate opening heavy creaking", "城门打开"),
    ("door gate closing and locking", "城门关闭"),
    ("bridge collapsing and falling", "桥梁坍塌"),
    ("flag capturing and planting", "旗帜插下"),
    ("treasure chest opening with sparkle", "宝箱开启"),
    ("ward or vision item placed", "视野放置"),
    ("shrine altar activation magical", "祭坛激活"),
    ("barrier wall raising from ground", "屏障升起"),
    ("barrier wall crumbling down", "屏障消失"),
    ("conveyor or speed pad boost", "传送带/加速板"),

    # ═══════════════════════════════════════════
    # 更多野怪与生物 / More Monsters & Creatures
    # ═══════════════════════════════════════════
    ("wolf howling at night", "狼嚎"),
    ("bird eagle screeching flying", "鹰啸"),
    ("snake hissing slithering", "蛇嘶"),
    ("insect buzzing flying around", "虫鸣飞行"),
    ("bear heavy stomping and growling", "熊咆哮"),
    ("spider skittering crawling", "蜘蛛爬行"),
    ("bat wings fluttering swarm", "蝙蝠振翅"),
    ("golem stone creature heavy steps", "石像怪步伐"),
    ("ghost phantom floating wail", "幽灵飘荡"),
    ("elemental creature crackling energy", "元素生物"),

    # ═══════════════════════════════════════════
    # 更多UI细分 / More UI Details
    # ═══════════════════════════════════════════
    ("slider dragging adjustment", "滑块调节"),
    ("toggle switch on off click", "开关切换"),
    ("tab switching panel change", "标签切换"),
    ("typing keyboard input tapping", "输入打字"),
    ("popup window appearing", "弹窗出现"),
    ("popup window dismissing closing", "弹窗关闭"),
    ("loading spinner processing waiting", "加载等待"),
    ("reward chest loot box opening", "奖励开箱"),
    ("card flipping and revealing", "卡牌翻转"),
    ("roulette spinning wheel of fortune", "轮盘转动"),
    ("skin or item preview showcase", "皮肤预览"),
    ("social friend request notification", "社交通知"),

    # ═══════════════════════════════════════════
    # 更多环境细分 / More Ambient Details
    # ═══════════════════════════════════════════
    ("volcano eruption and lava rumble", "火山喷发"),
    ("desert wind sand blowing", "沙漠风沙"),
    ("swamp bubbling and squelching", "沼泽冒泡"),
    ("frozen tundra ice cracking ambient", "冻原冰裂"),
    ("forest leaves rustling in wind", "森林树叶沙沙"),
    ("city town marketplace distant noise", "城镇喧嚣"),
    ("temple sacred ambient choir pad", "神殿氛围"),
    ("dungeon dark dripping eerie ambient", "地牢阴森"),
    ("space cosmic deep ambient drone", "宇宙深空"),
    ("underwater ambient bubbles and pressure", "水下环境"),

    # ═══════════════════════════════════════════
    # 更多角色动作 / More Character Actions
    # ═══════════════════════════════════════════
    ("sliding on ground or ice", "滑行"),
    ("dashing forward fast burst movement", "冲刺"),
    ("hovering floating in mid air", "悬浮"),
    ("wall jumping bouncing off wall", "蹬墙跳"),
    ("heavy landing ground pound slam", "重击落地"),
    ("spinning attack twirling body", "旋转攻击"),
    ("crouching or sneaking quietly", "蹲伏潜行"),
    ("grappling hook launching and pulling", "钩锁发射"),

    # ═══════════════════════════════════════════
    # 更多材质 / More Materials
    # ═══════════════════════════════════════════
    ("paper scroll unrolling and rustling", "卷轴展开"),
    ("leather stretching and creaking", "皮革拉伸"),
    ("ceramic pottery breaking", "陶瓷碎裂"),
    ("bamboo cracking and snapping", "竹子折断"),
    ("silk fabric whooshing softly", "丝绸拂动"),
    ("ice surface cracking under weight", "冰面踩裂"),
    ("sand gravel crunching underfoot", "沙砾踩踏"),

    # ═══════════════════════════════════════════
    # 更多特殊音效 / More Special FX
    # ═══════════════════════════════════════════
    ("echo reverberation in large hall", "大厅回响"),
    ("reverse sound playing backwards", "倒放音效"),
    ("glitch digital distortion artifact", "数字故障"),
    ("heartbeat pulsing tension rhythm", "心跳"),
    ("breathing heavy exhausted panting", "喘息"),
    ("wind chime tinkling gentle", "风铃"),
    ("fireworks launching and bursting", "烟花绽放"),
    ("confetti celebration particles falling", "庆祝彩带"),
    ("magical sparkle twinkling shimmer", "魔法闪烁"),
    ("energy humming constant low drone", "能量低鸣"),
    ("alarm warning siren blaring", "警报鸣响"),
    ("clock ticking mechanical rhythm", "钟表滴答"),
    ("whip cracking sharp snap", "鞭子抽响"),
    ("flute playing short melody", "笛声短旋律"),
    ("string instrument pluck pizzicato", "弦乐拨弦"),
    ("gong deep resonating strike", "铜锣"),
    ("chime ascending positive feedback", "正向反馈音"),
    ("buzz descending negative feedback", "负向反馈音"),

    # ═══════════════════════════════════════════
    # 召唤师技能 / Summoner Spells
    # ═══════════════════════════════════════════
    ("flash blink teleport instant short distance", "闪现"),
    ("ignite fire burning damage over time on enemy", "点燃/惩戒"),
    ("heal burst green healing aura on ally", "治疗术"),
    ("barrier temporary protective shield bubble", "屏障"),
    ("exhaust slow and weaken enemy gray mist", "虚弱"),
    ("cleanse purify removing crowd control", "净化"),
    ("ghost speed boost sprint running fast aura", "疾跑"),
    ("smite strike jungle monster with magic damage", "惩戒打击"),
    ("teleport channeling to tower or minion", "传送术"),
    ("revive respawn at fountain instantly", "复活术"),

    # ═══════════════════════════════════════════
    # 装备主动/被动触发 / Item Active & Passive
    # ═══════════════════════════════════════════
    ("hourglass stasis golden invulnerable freeze", "金身/中亚沙漏"),
    ("stopwatch single use stasis freeze", "秒表静止"),
    ("shield item activating protective barrier", "护盾装备触发"),
    ("locket team shield dome protection", "团队护盾"),
    ("redemption healing circle falling from sky", "救赎之光"),
    ("item passive proc on-hit magic sparkle", "装备被动触发"),
    ("lifesteal health drain vampiric healing on hit", "吸血回复"),
    ("armor penetration breaking through defense", "破甲穿透"),
    ("movement speed boost item activation", "装备加速"),
    ("ability power surge magical empowerment", "法强增幅"),
    ("attack speed increase rapid striking", "攻速提升"),
    ("critical strike item enhanced damage", "暴击装备触发"),
    ("spell vamp magic drain ability healing", "法术吸血"),
    ("mana restoration blue energy recovery", "蓝量回复"),
    ("health potion drinking and recovery", "血瓶回复"),
    ("ward item placing vision on ground", "眼石放置"),
    ("control ward revealing invisible enemy", "控制守卫"),
    ("trinket ward short range vision", "饰品放置"),

    # ═══════════════════════════════════════════
    # 英雄技能类型 / Ability Types
    # ═══════════════════════════════════════════
    ("passive ability ambient glow always active", "被动技能光效"),
    ("basic ability quick cast short cooldown", "基础技能施放"),
    ("ultimate ability powerful cast long cooldown", "大招施放"),
    ("ultimate ability ready notification power up", "大招就绪"),
    ("ability on cooldown ticking waiting", "技能冷却中"),
    ("ability missed whiffed no target hit", "技能空放"),
    ("ability enhanced empowered stronger cast", "强化技能"),
    ("auto attack basic melee swing", "普攻近战"),
    ("auto attack basic ranged projectile", "普攻远程"),
    ("ability combo chain multiple skills rapid", "技能连招"),

    # ═══════════════════════════════════════════
    # 控制效果细分 / Crowd Control Details
    # ═══════════════════════════════════════════
    ("root snare binding feet stuck to ground", "定身/缠绕"),
    ("airborne knock up launched into air", "击飞"),
    ("suppress pinned down unable to act", "压制"),
    ("taunt forced to attack taunter", "嘲讽"),
    ("polymorph transformed into harmless creature", "变形"),
    ("blind missing attacks unable to see", "致盲"),
    ("ground preventing dashes and movement abilities", "禁锢"),
    ("displacement hook pulling enemy toward you", "钩子拉扯"),
    ("displacement kick pushing enemy away far", "踢飞"),
    ("sleep falling asleep then waking on damage", "催眠"),
    ("slow gradually reducing movement speed ice", "减速"),
    ("stasis frozen in time unable to act invulnerable", "静止/石化"),

    # ═══════════════════════════════════════════
    # 野区与资源 / Jungle & Objectives
    # ═══════════════════════════════════════════
    ("red buff monster fiery aura burning", "红buff怪"),
    ("blue buff monster glowing mana aura", "蓝buff怪"),
    ("buff gained red burning aura on champion", "获得红buff"),
    ("buff gained blue glowing aura on champion", "获得蓝buff"),
    ("dragon elemental roar fire ice earth wind", "元素龙"),
    ("elder dragon epic powerful ancient roar", "远古龙/大龙"),
    ("baron nashor epic boss creature roar deep", "主宰/男爵"),
    ("rift herald eye monster charging forward", "先锋/峡谷先锋"),
    ("scuttle crab small river creature squeaking", "河蟹"),
    ("raptor bird jungle camp flapping wings", "锋鸟/禽鸟"),
    ("wolf jungle camp howling growling pack", "狼群"),
    ("gromp toad jungle camp croaking", "蛤蟆/河豚"),
    ("krugs rock golem jungle camp stone smashing", "石甲虫/岩石怪"),
    ("jungle plant blast cone explosion launch", "爆炸果实"),
    ("jungle plant honeyfruit healing drops", "恢复果实"),
    ("jungle plant scryers bloom vision reveal", "视野果实"),
    ("epic monster slain team buff gained fanfare", "史诗野怪击杀"),

    # ═══════════════════════════════════════════
    # 防御塔与建筑 / Towers & Structures
    # ═══════════════════════════════════════════
    ("tower outer turret laser beam attacking", "外塔射击"),
    ("tower inner turret stronger beam", "内塔射击"),
    ("tower inhibitor turret powerful beam", "高地塔射击"),
    ("tower nexus turret final defense beam", "基地塔射击"),
    ("tower plating breaking gold reward", "塔皮脱落"),
    ("inhibitor structure breaking open", "水晶被摧毁"),
    ("nexus crystal exploding game ending", "主水晶爆炸"),
    ("tower fortification armor bonus active", "塔防强化"),
    ("minion wave super empowered marching", "超级兵行军"),
    ("siege minion cannon firing at tower", "炮车射击"),

    # ═══════════════════════════════════════════
    # 经济与成长 / Economy & Progression
    # ═══════════════════════════════════════════
    ("gold earned coin clink from last hit", "补刀金币"),
    ("gold earned assist reward", "助攻奖励"),
    ("gold earned kill bounty reward", "击杀赏金"),
    ("experience gained level progress", "经验获取"),
    ("level up character with stat increase fanfare", "英雄升级"),
    ("max level reached full power", "满级达成"),
    ("item completed full build chime", "装备合成完成"),
    ("item component purchased partial build", "散件购买"),
    ("sell item refund gold received", "出售装备"),
    ("passive gold income ticking ambient", "被动金币收入"),
    ("bounty shutdown big gold reward", "赏金终结"),

    # ═══════════════════════════════════════════
    # 战场事件 / Battlefield Events
    # ═══════════════════════════════════════════
    ("minion wave spawning at nexus", "兵线刷新"),
    ("cannon minion spawning heavier wave", "炮车刷新"),
    ("jungle camp respawning monsters returning", "野怪刷新"),
    ("dragon pit objective spawning", "龙坑刷新"),
    ("baron pit objective spawning", "主宰坑刷新"),
    ("team fight chaotic multiple abilities clashing", "团战混战"),
    ("ambush surprise attack from bush", "伏击突袭"),
    ("chase pursuit running after enemy", "追击"),
    ("escape fleeing running away from danger", "逃跑"),
    ("split push minion tower pressure distant", "分推"),

    # ═══════════════════════════════════════════
    # 视野与地图机制 / Vision & Map
    # ═══════════════════════════════════════════
    ("fog of war darkness edge of vision", "战争迷雾"),
    ("ward placed gaining vision light reveal", "插眼"),
    ("ward destroyed vision denied cracking", "排眼"),
    ("sweeper lens scanning for invisible objects", "扫描"),
    ("invisible enemy revealed caught exposed", "隐身暴露"),
    ("minimap ping alert attention signal", "小地图标记"),
    ("danger ping warning exclamation alert", "危险信号"),
    ("on my way ping heading to location", "正在路上信号"),
    ("enemy missing ping question mark alert", "敌人消失信号"),
    ("assist me ping requesting help", "请求支援信号"),
    ("retreat ping falling back signal", "撤退信号"),

    # ═══════════════════════════════════════════
    # 回城与泉水 / Recall & Fountain
    # ═══════════════════════════════════════════
    ("recall channeling swirling energy going home", "回城吟唱"),
    ("recall completed teleport back to base flash", "回城完成"),
    ("recall interrupted cancelled by damage", "回城被打断"),
    ("fountain healing rapid health regeneration", "泉水回复"),
    ("spawn platform at game start waiting", "出生平台"),
    ("leaving base walking out of fountain", "离开泉水"),

    # ═══════════════════════════════════════════
    # 皮肤与特效 / Skins & Cosmetics
    # ═══════════════════════════════════════════
    ("legendary skin special ability enhanced effect", "传说皮肤特效"),
    ("recall animation special skin celebration", "皮肤回城动画"),
    ("emote taunt dance celebration animation", "表情/舞蹈动作"),
    ("death animation special skin dramatic", "皮肤死亡特效"),
    ("spawn animation special entrance dramatic", "出场特效"),
    ("trail particle movement effect behind champion", "移动拖尾特效"),
    ("attack particle special auto effect skin", "皮肤攻击特效"),

    # ═══════════════════════════════════════════
    # 对局阶段 / Game Phases
    # ═══════════════════════════════════════════
    ("game loading screen ambient waiting", "加载界面"),
    ("game start horn battle begins", "对局开始号角"),
    ("minion first wave arriving laning phase", "首波兵线到达"),
    ("late game intense music tension rising", "后期紧张氛围"),
    ("nexus exploding victory ending fanfare", "胜利爆破"),
    ("nexus exploding defeat ending somber", "失败结束"),
    ("surrender vote popup notification", "投降投票"),

    # ═══════════════════════════════════════════
    # 移动与地形 / Movement & Terrain
    # ═══════════════════════════════════════════
    ("walking through river water splashing steps", "河道行走"),
    ("walking through jungle forest path", "野区行走"),
    ("walking on lane stone paved road", "线上行走"),
    ("entering bush grass concealment rustling", "进入草丛"),
    ("exiting bush grass revealed stepping out", "离开草丛"),
    ("terrain wall flash through blink over wall", "穿墙闪现"),
    ("movement ability dash through terrain", "技能穿墙"),
    ("blast cone knocked over wall launched", "爆炸果实弹射"),

    # ═══════════════════════════════════════════
    # 更多打击感细分 / More Combat Feel
    # ═══════════════════════════════════════════
    ("tower shot hitting champion high damage", "塔射命中英雄"),
    ("execute finishing blow low health target", "斩杀/处刑"),
    ("overkill excessive damage on target", "过量伤害"),
    ("true damage pure health reduction effect", "真实伤害"),
    ("magic damage ability power spell hitting", "法术伤害命中"),
    ("physical damage weapon attack hitting", "物理伤害命中"),
    ("percent health damage chunk based on max hp", "百分比伤害"),
    ("shield absorbed damage blocked by barrier", "护盾吸收伤害"),
    ("healing received health restored green plus", "治疗生效"),
    ("grievous wounds healing reduction debuff", "重伤效果"),
    ("tenacity crowd control duration reduced", "韧性减控"),
    ("adaptive force switching damage type", "自适应切换"),

    # ═══════════════════════════════════════════
    # 技能起手 / Skill Cast Start
    # ═══════════════════════════════════════════
    ("instant cast quick snap ability activation", "瞬发技能起手"),
    ("short cast quick tap ability under one second", "短促技能点按"),
    ("charge up hold building power before release", "蓄力起手"),
    ("channel start casting sustained holding still", "持续施法开始"),
    ("wind up slow heavy swing preparation", "重击预备摆动"),
    ("quick draw weapon unsheathe for strike", "快速拔刀起手"),
    ("finger snap magical instant trigger", "弹指瞬发"),
    ("hand clap energy burst from palms", "击掌释放"),
    ("stomp foot ground pound initiation", "跺脚起手"),
    ("spin start beginning rotational attack", "旋转起手"),

    # ═══════════════════════════════════════════
    # 技能飞行 / Projectile Travel
    # ═══════════════════════════════════════════
    ("projectile flying through air whistling", "弹道飞行呼啸"),
    ("slow projectile heavy traveling thud", "慢速弹道飞行"),
    ("fast projectile zipping quick whistle", "高速弹道飞行"),
    ("homing projectile tracking curving toward target", "追踪弹道"),
    ("projectile bouncing between multiple targets", "弹射飞行"),
    ("arcing projectile lobbed high trajectory", "抛物线弹道"),
    ("boomerang projectile going out and returning", "去回弹道"),
    ("piercing projectile passing through targets", "穿透弹道"),
    ("expanding projectile growing larger as it travels", "扩散弹道"),
    ("splitting projectile breaking into multiple", "分裂弹道"),

    # ═══════════════════════════════════════════
    # 命中反馈-轻击 / Hit Feedback - Light
    # ═══════════════════════════════════════════
    ("light hit tap tiny impact soft feedback", "轻击反馈"),
    ("light slash quick small cut sound", "轻斩反馈"),
    ("light poke small jab minimal impact", "轻戳反馈"),
    ("graze barely hitting glancing light touch", "擦伤反馈"),
    ("tick small damage proc tiny hit sound", "微量伤害跳字"),
    ("dot damage tick periodic small hit burn", "持续伤害跳动"),

    # ═══════════════════════════════════════════
    # 命中反馈-中击 / Hit Feedback - Medium
    # ═══════════════════════════════════════════
    ("medium hit solid impact satisfying thud", "中击反馈"),
    ("medium slash clean cut through target", "中斩反馈"),
    ("medium punch solid body blow", "中拳反馈"),
    ("ability hitting target magical impact medium", "技能命中反馈"),
    ("projectile hitting target medium impact thud", "弹道命中反馈"),
    ("melee skill connecting with target medium", "近战技能命中"),

    # ═══════════════════════════════════════════
    # 命中反馈-重击 / Hit Feedback - Heavy
    # ═══════════════════════════════════════════
    ("heavy hit massive impact crunching powerful blow", "重击反馈"),
    ("heavy slam ground shaking powerful smash", "重砸反馈"),
    ("heavy cleave splitting through with force", "重劈反馈"),
    ("ultimate ability hitting target devastating impact", "大招命中反馈"),
    ("max stack full charge releasing all power hit", "满层释放命中"),
    ("finisher killing blow execution impact", "终结技命中"),

    # ═══════════════════════════════════════════
    # 命中反馈-多段 / Hit Feedback - Multi-hit
    # ═══════════════════════════════════════════
    ("rapid multi hit flurry many small impacts fast", "快速多段命中"),
    ("two hit double strike quick successive", "二连击"),
    ("three hit triple strike combo rhythm", "三连击"),
    ("machine gun rapid continuous hitting stream", "机关枪式连击"),
    ("scattered multi target hitting several at once", "群体命中"),
    ("rhythmic hitting steady beat repeated impact", "节奏性连击"),

    # ═══════════════════════════════════════════
    # 技能余韵 / Skill Aftermath
    # ═══════════════════════════════════════════
    ("skill end lingering echo fading away", "技能余韵消散"),
    ("explosion aftermath debris settling dust", "爆炸残留"),
    ("fire lingering burning ground after ability", "火焰地面残留"),
    ("ice lingering frozen area after ability", "冰冻地面残留"),
    ("poison lingering toxic cloud after ability", "毒雾残留"),
    ("electric lingering sparking after lightning", "电弧残留"),
    ("magic dissipating particles fading sparkle", "魔法粒子消散"),
    ("crater impact mark ground depression", "地面撞击坑"),

    # ═══════════════════════════════════════════
    # 战斗状态反馈 / Combat State Feedback
    # ═══════════════════════════════════════════
    ("low health warning heartbeat critical danger", "低血量心跳警告"),
    ("near death alarm urgent pulsing danger", "濒死紧急警报"),
    ("health dropping rapidly taking burst damage", "血量骤降"),
    ("out of mana empty blue resource depleted", "蓝量耗尽"),
    ("out of energy resource empty unable to cast", "能量耗尽"),
    ("ability ready off cooldown available ping", "技能就绪提示"),
    ("passive stack gained counting up", "被动层数增加"),
    ("passive fully stacked ready to consume", "被动满层就绪"),
    ("combo counter incrementing hit count", "连击计数增加"),
    ("on-hit effect proc item or rune triggered", "命中特效触发"),

    # ═══════════════════════════════════════════
    # 补刀与小兵交互 / Last Hit & Minion
    # ═══════════════════════════════════════════
    ("last hit minion killing blow gold earned pop", "补刀命中"),
    ("last hit cannon minion bigger gold reward thud", "炮车补刀"),
    ("missed last hit minion died no gold", "漏刀"),
    ("minion auto attacking another minion small hit", "小兵互殴"),
    ("siege minion cannon ball hitting tower thud", "炮车打塔"),
    ("super minion heavy footstep powerful march", "超级兵重步"),

    # ═══════════════════════════════════════════
    # 瞬态音效 / Transient Micro Sounds
    # ═══════════════════════════════════════════
    ("swoosh fast movement air displacement short", "快速位移气流"),
    ("whoosh medium speed swing through air", "中速挥动气流"),
    ("zip very fast tiny projectile passing", "极速掠过"),
    ("pop small bubble or projectile bursting", "气泡破裂"),
    ("snap quick sharp short breaking sound", "短促脆响"),
    ("ding small bell notification trigger chime", "叮咚触发提示"),
    ("zap quick electric short shock hit", "电击短促"),
    ("thwack sharp physical hit contact", "啪击接触"),
    ("sizzle hot burning contact brief hiss", "灼烧嘶声"),
    ("crunch crushing compressing short impact", "碾压脆响"),
    ("clink small metal contact light tap", "金属轻碰"),
    ("thump dull heavy short impact on ground", "沉闷短促落地"),
    ("swish light blade cutting air brief", "轻刃划空"),
    ("pew small energy shot fired tiny blast", "小型能量弹"),
    ("boing spring bounce elastic recoil", "弹性弹跳"),
    ("squelch wet fleshy impact squishing", "湿润肉击"),

    # ═══════════════════════════════════════════
    # 动画绑定音效 / Animation-tied Sounds
    # ═══════════════════════════════════════════
    ("weapon trail light streak following swing", "武器挥动拖尾"),
    ("body motion twist turning torso attack", "身体扭转攻击"),
    ("cape cloth fluttering during quick movement", "披风飘动"),
    ("hair accessories jingling during action", "饰品晃动"),
    ("armor plates shifting during movement", "甲片位移"),
    ("ground crack from heavy champion landing", "重型落地裂地"),
    ("dust cloud puff from sudden dash", "冲刺扬尘"),
    ("splash water spray from aquatic ability", "水花四溅"),
    ("petal flower scatter from nature ability", "花瓣飘散"),
    ("feather scatter from wing or bird ability", "羽毛飘散"),
    ("spark shower from metal grinding contact", "金属摩擦火星"),
    ("energy trail glowing path behind movement", "能量移动轨迹"),

    # ═══════════════════════════════════════════
    # 普攻细分 / Auto Attack Details
    # ═══════════════════════════════════════════
    ("auto attack first in sequence initial swing", "普攻第一段"),
    ("auto attack second in sequence follow up", "普攻第二段"),
    ("auto attack third in sequence combo finisher", "普攻第三段"),
    ("auto attack reset quick cancel into new swing", "普攻重置"),
    ("auto attack wind up backswing preparation", "普攻前摇"),
    ("auto attack recovery follow through after hit", "普攻后摇"),
    ("empowered auto attack enhanced next basic hit", "强化普攻"),
    ("ranged auto attack projectile launch click", "远程普攻弹射"),
    ("on-hit auto attack extra magic damage proc", "普攻附魔触发"),
    ("auto attack critically striking bonus damage", "普攻暴击"),
    ("auto attack hitting tower structure slow thud", "普攻打塔"),
    ("auto attack hitting ward small object break", "普攻排眼"),
]

# 拆分为两个列表
LABEL_PROMPTS = [f"This is a sound of {desc}" for desc, _ in LABELS]
LABEL_NAMES = [name for _, name in LABELS]


def _load_model():
    global _model, _text_embeddings
    if _model is not None:
        return _model, _text_embeddings

    import laion_clap

    print("加载 CLAP 模型...", end=" ", flush=True)
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()  # 自动下载预训练权重
    print("完成")

    # 预计算文本嵌入（只算一次）
    print("计算标签嵌入...", end=" ", flush=True)
    text_embeddings = model.get_text_embedding(LABEL_PROMPTS, use_tensor=True)
    # 归一化
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    print("完成")

    _model = model
    _text_embeddings = text_embeddings
    return _model, _text_embeddings


def classify(audio_path: str, top_k: int = 3) -> list[dict]:
    """
    对音频文件进行分类，返回 top_k 个最可能的类别。
    返回格式：[{"label": "近战武器攻击", "score": 0.87}, ...]
    """
    try:
        model, text_emb = _load_model()

        # 获取音频嵌入
        audio_emb = model.get_audio_embedding_from_filelist(
            x=[audio_path], use_tensor=True
        )
        audio_emb = audio_emb / audio_emb.norm(dim=-1, keepdim=True)

        # 计算相似度 → softmax 得到概率
        with torch.no_grad():
            similarity = (audio_emb @ text_emb.t()).squeeze(0)
            probs = torch.softmax(similarity, dim=-1).cpu().numpy()

        top_indices = np.argsort(probs)[::-1][:top_k]

        return [
            {"label": LABEL_NAMES[i], "score": round(float(probs[i]), 3)}
            for i in top_indices
        ]

    except Exception as e:
        return [{"label": "unknown", "score": 0.0, "error": str(e)}]


def batch_classify(file_list: list[str], top_k: int = 3) -> dict:
    """
    批量分类，返回 {文件名: [分类结果]} 映射
    """
    results = {}
    total = len(file_list)
    print(f"\n音效分类中，共 {total} 个文件...")

    for i, path in enumerate(file_list):
        filename = Path(path).name
        labels = classify(path, top_k)
        text = " / ".join(l["label"] for l in labels if l["label"] != "unknown")
        results[filename] = {
            "text": text,
            "type": "sfx",
            "labels": labels,
            "path": path
        }
        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{total}")

    print(f"音效分类完成，共 {total} 个")
    return results
