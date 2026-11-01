import pytest
from reelrank.serve.validation import validate_recommendations


def test_loadtest_rejects_empty_duplicate_and_invalid_scores():
    item = dict(video_id=1, score=.5, p_click=.4, p_long_view=.5, p_like=.1)
    for items in [[], [item, item], [{**item, 'score': float('nan')}]]:
        with pytest.raises(ValueError):
            validate_recommendations({'items': items, 'timings_ms': {'total_ms': 1.}})
    validate_recommendations({'items': [item], 'timings_ms': {'total_ms': 1.}})
