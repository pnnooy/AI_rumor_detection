# event_context.py
# 事件背景信息 - 根据训练数据推文内容补充完善

EVENT_CONTEXT = {
    0: (
        "关于 Gurlitt 艺术藏品的归属争议，涉及纳粹掠夺艺术品归还问题。"
        "2013年在慕尼黑发现 Cornelius Gurlitt 藏有大量纳粹时期掠夺的艺术品，"
        "瑞士伯尔尼艺术博物馆最终决定接受这批藏品，并承诺归还给原主后人。"
    ),
    
    1: (
        "关于 Ferguson 事件中警察执法和 Michael Brown 枪击案的讨论。"
        "2014年8月9日，美国密苏里州 Ferguson 市，18岁黑人青年 Michael Brown "
        "被白人警官 Darren Wilson 开枪打死，引发大规模抗议和关于种族歧视的讨论。"
    ),
    
    2: (
        "关于 AC米兰中场球员 Michael Essien 感染埃博拉病毒的谣言。"
        "2014年埃博拉疫情期间，多家尼日利亚媒体错误报道 Essien 确诊感染埃博拉病毒，"
        "AC米兰俱乐部随后官方辟谣，证实该消息为虚假信息。"
    ),
    
    3: (
        "关于歌手 Prince 在加拿大多伦多 Massey Hall 举行秘密演唱会的谣言。"
        "2015年初，社交媒体传闻 Prince 将在多伦多举办 surprise 演出，"
        "导致大量粉丝聚集 Massey Hall 排队等候，但 Live Nation 随后确认演出并不存在。"
    ),
    
    4: (
        "关于德国之翼航空 9525号航班在法国南部坠毁的讨论。"
        "2015年3月24日，Germanwings A320 客机在法国阿尔卑斯山区坠毁，"
        "机上150人全部遇难。调查显示副驾驶 Andreas Lubitz 蓄意撞山，"
        "引发对航空安全和飞行员心理健康问题的广泛讨论。"
    ),
    
    5: (
        "关于澳大利亚悉尼 Martin Place 咖啡馆人质劫持事件的讨论。"
        "2014年12月15日，悉尼市中心 Martin Place 的 Lindt 咖啡馆发生人质劫持事件，"
        "枪手挟持多名人质长达16小时，最终警方突袭结束对峙，"
        "造成枪手和2名人质死亡，多人受伤。"
    ),
    
    6: (
        "关于加拿大渥太华国会山枪击案的讨论。"
        "2014年10月22日，一名枪手在加拿大渥太华国家战争纪念碑处枪杀一名卫兵，"
        "随后闯入国会山大楼与安全人员交火，枪手被击毙。"
        "事件引发对加拿大反恐政策的讨论，多伦多蓝鸟队赛后也临时取消了烟花表演以示哀悼。"
    ),
}


def get_event_context(event_id: int, default: str = "") -> str:
    """
    获取事件背景信息
    
    Args:
        event_id: 事件 ID (0-6)
        default: 如果事件不存在时返回的默认值
        
    Returns:
        str: 事件背景描述
    """
    return EVENT_CONTEXT.get(event_id, default)


def print_all_contexts() -> None:
    """打印所有事件背景信息"""
    for event_id, context in EVENT_CONTEXT.items():
        print(f"Event {event_id}: {context}\n")


if __name__ == '__main__':
    print("=== 事件背景信息 ===")
    print_all_contexts()