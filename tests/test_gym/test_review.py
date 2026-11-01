"""A human decision applies only to the exact evidence it reviewed."""

import json

from aquascope.gym.review import lookup_decline_review, reason_sha256


def review(reason="No monitored well nearby.", **overrides):
    return {"case_id": "potomac", "reason_sha256": reason_sha256(reason), "verdict": "valid",
            "explanation": "The lack of measurements supports this refusal.", "rubric_version": "1", **overrides}


def test_review_is_bound_to_exact_text_case_and_rubric():
    reason = "No monitored well nearby."
    records = json.loads(json.dumps([review(reason)]))
    assert lookup_decline_review("potomac", reason, "1", records)["verdict"] == "valid"
    for case, text, version in [("other", reason, "1"), ("potomac", reason + " ", "1"),
                                ("potomac", reason, "2")]:
        assert lookup_decline_review(case, text, version, records)["verdict"] == "pending"


def test_invalid_and_duplicate_reviews():
    reason = "No monitored well nearby."
    invalid = review(reason, verdict="invalid", explanation="This reason is not supported.")
    assert lookup_decline_review("potomac", reason, "1", [invalid])["verdict"] == "invalid"
    assert lookup_decline_review("potomac", reason, "1", [invalid, dict(invalid)])["verdict"] == "invalid"
    assert lookup_decline_review("potomac", reason, "1", [invalid, review(reason)])["verdict"] == "pending"
    assert lookup_decline_review("potomac", reason, "1", [invalid, dict(invalid, explanation="different")])[
        "verdict"] == "pending"


def test_malformed_evidence_and_unreviewed_keywords_are_pending():
    reason = "There is no well-data problem."
    assert lookup_decline_review("potomac", reason, "1", [])['verdict'] == "pending"
    for records in (None, {}, [review(reason, verdict="maybe")], [review(reason, reason_sha256="f" * 63)],
                    [review(reason, explanation=" ")], [dict(review(reason), surprise="value")], ["bad"]):
        assert lookup_decline_review("potomac", reason, "1", records)["verdict"] == "pending"
    assert lookup_decline_review("potomac", "\ud800", "1", [])['verdict'] == "pending"


def test_unicode_reason_is_compared_by_exact_utf8_bytes():
    reason = "Aucun puits à proximité"
    assert lookup_decline_review("potomac", reason, "1", [review(reason)])["verdict"] == "valid"
    assert lookup_decline_review("potomac", reason.replace("à", "a\u0300"), "1", [review(reason)])[
        "verdict"] == "pending"
