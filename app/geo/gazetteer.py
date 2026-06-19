from __future__ import annotations

from pydantic import BaseModel


class GazetteerEntry(BaseModel):
    name: str
    lat: float
    lon: float
    aliases: list[str]
    confidence: float
    description: str


SEED_AREAS: list[GazetteerEntry] = [
    GazetteerEntry(name="东方枢纽/祝桥", lat=31.1677, lon=121.7756, aliases=["东方枢纽", "祝桥", "浦东机场", "机场联络线"], confidence=0.72, description="浦东祝桥及东方枢纽周边"),
    GazetteerEntry(name="临港新片区/滴水湖", lat=30.8956, lon=121.9296, aliases=["临港", "滴水湖", "南汇新城", "临港新片区"], confidence=0.78, description="临港新片区与滴水湖核心区"),
    GazetteerEntry(name="张江", lat=31.2077, lon=121.5999, aliases=["张江", "张江科学城", "张江高科"], confidence=0.8, description="浦东张江科学城"),
    GazetteerEntry(name="金桥", lat=31.2573, lon=121.5865, aliases=["金桥", "金桥开发区", "金桥经济技术开发区"], confidence=0.78, description="浦东金桥开发区"),
    GazetteerEntry(name="大零号湾", lat=31.0242, lon=121.4384, aliases=["大零号湾", "零号湾", "紫竹", "东川路"], confidence=0.72, description="闵行大零号湾科创策源区"),
    GazetteerEntry(name="虹桥国际中央商务区/西虹桥", lat=31.1941, lon=121.3069, aliases=["虹桥", "西虹桥", "虹桥商务区", "虹桥国际中央商务区"], confidence=0.76, description="虹桥国际中央商务区与西虹桥"),
    GazetteerEntry(name="青浦工业园区", lat=31.1702, lon=121.1244, aliases=["青浦工业园区", "青浦工业园", "青浦新城"], confidence=0.72, description="青浦工业园区与青浦新城周边"),
    GazetteerEntry(name="嘉定新城/安亭", lat=31.2912, lon=121.1663, aliases=["嘉定新城", "安亭", "汽车城", "国际汽车城"], confidence=0.74, description="嘉定新城、安亭和汽车城板块"),
    GazetteerEntry(name="吴淞创新城/南大智慧城", lat=31.3305, lon=121.4217, aliases=["吴淞创新城", "南大智慧城", "南大", "吴淞"], confidence=0.7, description="宝山吴淞创新城与南大智慧城"),
    GazetteerEntry(name="金山工业区/上海化工区", lat=30.8048, lon=121.3527, aliases=["金山工业区", "上海化工区", "化工区", "金山"], confidence=0.7, description="金山工业区与上海化工区"),
    GazetteerEntry(name="奉贤新城/东方美谷", lat=30.9179, lon=121.4732, aliases=["奉贤新城", "东方美谷", "奉贤"], confidence=0.72, description="奉贤新城与东方美谷"),
    GazetteerEntry(name="松江新城/G60科创走廊", lat=31.0322, lon=121.2277, aliases=["松江新城", "G60", "G60科创走廊", "松江"], confidence=0.74, description="松江新城与 G60 科创走廊"),
]


DISTRICT_FALLBACKS: list[GazetteerEntry] = [
    GazetteerEntry(name="浦东新区", lat=31.2215, lon=121.5441, aliases=["浦东新区", "浦东"], confidence=0.45, description="浦东新区泛区域"),
    GazetteerEntry(name="闵行区", lat=31.1128, lon=121.3817, aliases=["闵行区", "闵行"], confidence=0.42, description="闵行区泛区域"),
    GazetteerEntry(name="宝山区", lat=31.4055, lon=121.4896, aliases=["宝山区", "宝山"], confidence=0.42, description="宝山区泛区域"),
    GazetteerEntry(name="嘉定区", lat=31.3747, lon=121.2653, aliases=["嘉定区", "嘉定"], confidence=0.42, description="嘉定区泛区域"),
    GazetteerEntry(name="金山区", lat=30.7428, lon=121.3424, aliases=["金山区", "金山"], confidence=0.42, description="金山区泛区域"),
    GazetteerEntry(name="松江区", lat=31.0325, lon=121.2277, aliases=["松江区", "松江"], confidence=0.42, description="松江区泛区域"),
    GazetteerEntry(name="青浦区", lat=31.1512, lon=121.1242, aliases=["青浦区", "青浦"], confidence=0.42, description="青浦区泛区域"),
    GazetteerEntry(name="奉贤区", lat=30.9184, lon=121.4741, aliases=["奉贤区", "奉贤"], confidence=0.42, description="奉贤区泛区域"),
]


def gazetteer_lookup(text: str | None) -> GazetteerEntry | None:
    if not text:
        return None
    haystack = text.lower()
    for entry in SEED_AREAS + DISTRICT_FALLBACKS:
        if entry.name.lower() in haystack or any(alias.lower() in haystack for alias in entry.aliases):
            return entry
    return None

