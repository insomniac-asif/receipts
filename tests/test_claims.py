"""Tests for the heuristic claim extractor."""

from receipts.claims import extract_claims


def _actions(text):
    return [(c.action, c.text) for c in extract_claims(text)]


def test_simple_past_tense_claim():
    claims = extract_claims("i banned the user")
    assert len(claims) == 1
    assert claims[0].action == "ban"
    assert "user" in claims[0].object_tokens


def test_multiple_actions_one_sentence():
    claims = extract_claims("i created the role and banned the user")
    actions = {c.action for c in claims}
    assert actions == {"create", "ban"}


def test_present_perfect_is_a_claim():
    claims = extract_claims("i've deleted the channel")
    assert len(claims) == 1
    assert claims[0].action == "delete"


def test_future_tense_is_not_a_claim():
    assert extract_claims("i will ban the user") == []
    assert extract_claims("i'm going to update the config") == []


def test_negation_is_not_a_claim():
    assert extract_claims("i did not delete any messages") == []
    assert extract_claims("i never banned anyone") == []


def test_intention_verbs_blocked():
    assert extract_claims("i tried to merge the branch") == []
    assert extract_claims("let me create the role") == []


def test_all_set_idiom_not_a_claim_but_real_verb_survives():
    # "all set" must not fire, but a genuine verb later must still extract.
    claims = extract_claims("all set - i updated the config")
    assert len(claims) == 1
    assert claims[0].action == "update"


def test_bare_verb_without_object_dropped():
    # Too vague to reconcile against anything.
    assert extract_claims("done, i updated.") == []


def test_object_tokens_exclude_stopwords():
    claims = extract_claims("i sent them the rules")
    assert claims[0].action == "send"
    assert "the" not in claims[0].object_tokens
    assert "rules" in claims[0].object_tokens
