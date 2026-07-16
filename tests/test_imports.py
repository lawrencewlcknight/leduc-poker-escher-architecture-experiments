"""Lightweight import checks for modules that do not require running experiments."""


def test_constants_importable():
    from escher_poker.constants import (
        LEDUC_AVERAGE_POLICY_VALUE_TARGET,
        LEDUC_GAME_VALUE_PLAYER_0,
    )

    assert LEDUC_GAME_VALUE_PLAYER_0 == -0.085606424078
    assert LEDUC_AVERAGE_POLICY_VALUE_TARGET == LEDUC_GAME_VALUE_PLAYER_0
