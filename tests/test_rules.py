from reelrank.safety.rules import rule_hit, rule_score


def test_blocklist_hits():
    assert rule_hit("加微信abc123领优惠")
    assert rule_hit("click the 小黄车 now")
    assert not rule_hit("今日穿搭分享，秋天的风")
    assert rule_score("加微信") > 0
