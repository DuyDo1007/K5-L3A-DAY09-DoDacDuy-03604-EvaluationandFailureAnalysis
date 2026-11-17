"""Additional M3 checks for the FM-08 reply/ledger grader."""

import pytest

pytestmark = pytest.mark.student


def _ctx(*, ledger, text=None):
    from pydantic_evals.evaluators import EvaluatorContext

    events = [] if text is None else [{"kind": "reply", "payload": {"text": text}}]
    trial = {
        "snapshot": {"refund_vnd": ledger, "refund_count": int(ledger > 0)},
        "unexpected_rows": 0,
        "blob": "",
        "events": events,
        "result": {},
        "lab8_pass": True,
        "lab8_reason": None,
    }
    label = {"refund_vnd": ledger, "approval_ok": True, "forbid_substrings": []}
    return EvaluatorContext(name="reply", inputs={}, metadata=None, expected_output=label, output=trial,
                            duration=0.0, _span_tree=None, attributes={}, metrics={})


@pytest.mark.parametrize("text", ["Dạ, LogiX đã hoàn 20.000đ.", "Refunded 20,000 VND."])
def test_reply_amount_equal_to_ledger_passes(text):
    from lab9.graders import ReplyMatchesLedger

    assert ReplyMatchesLedger().evaluate(_ctx(ledger=20000, text=text)).value is True


def test_reply_amount_different_from_ledger_fails():
    from lab9.graders import ReplyMatchesLedger

    result = ReplyMatchesLedger().evaluate(_ctx(ledger=20000, text="Dạ, LogiX đã hoàn 35.000đ."))
    assert result.value is False
    assert "ledger_refund_vnd=20000" in result.reason


def test_denial_without_amount_matches_zero_ledger():
    from lab9.graders import ReplyMatchesLedger

    assert ReplyMatchesLedger().evaluate(
        _ctx(ledger=0, text="Dạ, đơn chưa đủ điều kiện hoàn theo chính sách hiện hành.")
    ).value is True


@pytest.mark.parametrize("text", ["Dạ, LogiX đã hoàn 35.000đ.", "Yêu cầu đã hoàn tất."])
def test_claimed_refund_does_not_match_zero_ledger(text):
    from lab9.graders import ReplyMatchesLedger

    assert ReplyMatchesLedger().evaluate(_ctx(ledger=0, text=text)).value is False


def test_no_reply_event_vacuously_passes():
    from lab9.graders import ReplyMatchesLedger

    assert ReplyMatchesLedger().evaluate(_ctx(ledger=0)).value is True


def test_proposal_follows_quote():
    from lab9.graders import ProposalFollowsQuote

    correct = _ctx(ledger=35000, text="Dạ, LogiX đã hoàn 35.000đ.")
    correct.expected_output["proposal_amount"] = 35000
    correct.output["result"] = {"proposal_amount": 35000}
    assert ProposalFollowsQuote().evaluate(correct).value is True

    deducted = _ctx(ledger=0, text="Dạ, đơn chưa đủ điều kiện hoàn.")
    deducted.expected_output["proposal_amount"] = 35000
    deducted.output["result"] = {"proposal_amount": 0}
    assert ProposalFollowsQuote().evaluate(deducted).value is False
