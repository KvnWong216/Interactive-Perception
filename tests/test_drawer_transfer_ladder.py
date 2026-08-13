from scripts.run_drawer_transfer_ladder import make_variants


def test_ladder_changes_one_factor_at_a_time() -> None:
    stock = """(:objects\n    plate_1 - plate\n)\n(:init\n    (On wine_bottle_1 main_table_wine_bottle_region)\n    (On wooden_cabinet_1 main_table_cabinet_region)\n)\n(:goal (And (Open wooden_cabinet_1_middle_region)))"""
    variants = make_variants(stock)
    assert "(Close wooden_cabinet_1_middle_region)" not in variants["B_stock_seeded_reset"]
    assert "(Close wooden_cabinet_1_middle_region)" in variants["C_explicit_close"]
    assert "butter_1 - butter" not in variants["C_explicit_close"]
    assert "(In butter_1 wooden_cabinet_1_middle_region)" in variants["D_hidden_butter"]
    assert "basket_1 - basket" in variants["E_add_basket_open_goal"]
    assert "(Open wooden_cabinet_1_middle_region)" in variants["E_add_basket_open_goal"]
    assert "(In butter_1 basket_1_contain_region)" in variants["F_retrieval_goal"]
