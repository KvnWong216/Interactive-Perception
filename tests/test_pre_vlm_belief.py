from interaction_uncertainty.pre_vlm_belief import parse_manipulation_prompt


def test_compound_pick_place_prompt_keeps_target_clean() -> None:
    task = parse_manipulation_prompt(
        "Pick up the butter and place it in the basket"
    )
    assert task["target"] == "butter"
    assert task["destination"] == "basket"
    assert task["source_hint"] is None


def test_grounded_compound_prompt_separates_source_and_destination() -> None:
    task = parse_manipulation_prompt(
        "Pick up the red and yellow butter package inside the open middle drawer "
        "and place it in the basket"
    )
    assert task["target"] == "red and yellow butter package"
    assert task["source_hint"] == "open middle drawer"
    assert task["destination"] == "basket"
