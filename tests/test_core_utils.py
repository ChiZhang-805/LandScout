from app.core.utils import leading_label


def test_leading_label_handles_chinese_and_ascii_colons():
    assert leading_label("重点观察：每周监测控规") == "重点观察"
    assert leading_label("Low priority: keep monitoring") == "Low priority"
    assert leading_label("") == "未分配"
    assert leading_label(None) == "未分配"
