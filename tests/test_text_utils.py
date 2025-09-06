from utils.text_utils import cluster_personas, score_copy_package

def test_cluster_personas_basic():
    persons = ["A founders","B founders","C marketers","D marketers","E ops"]
    segs = cluster_personas(persons, 2)
    assert 1 <= len(segs) <= 2
    assert all(isinstance(s, str) for s in segs)

def test_scoring_bounds():
    copy = {"headline":"Test headline", "body":"This is a short test body.", "cta":"Learn more"}
    score = score_copy_package(copy, "Brand builds rituals.", "Awareness")
    assert 0 <= score["total"] <= 100
