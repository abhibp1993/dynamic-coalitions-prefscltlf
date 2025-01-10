def _strategy_given_rank(rank, product_game, conc_game, values, n_players):
    # Get P1 states with given ranks
    states = set()  # TODO

    # Fix point computation
    set_u = None  # TODO (Use concurrent game)
    costs = dict()
    while set_u:
        # Iterate over all states in set_u
        for u in set_u:
            for players, act in get_coalition_actions(product_game, u):
                if len(players) > 1:
                    # Decouple players
                    _, player_i = players

                    # Get next states given coalition action
                    next_states_under_a = partial_transition(conc_game, u, players, act, values)

                    # If coalition is NOT rational for player i, eliminate coalition action
                    if values[u][player_i] < max(values[v][player_i] for v in next_states_under_a):
                        pass        # TODO

                    else:
                        pass        # TODO. Update max player-i cost that can be guaranteed

                # Compute costs for all non-coalitional player
                for player_j in set(range(n_players)) - players:
                    pass        # TODO. Update cost for player j

                # Update costs dictionary: {state: {coalition-action: {non-coalitional-action: cost}}
                # TODO

        # Eliminate states with no enabled actions (use costs dictionary)
        # For surviving states, update max costs for all players.

        # Update Vk
        # Break condition

        # Update set_u
        set_u = None  # Pre(Vk) - Vk








    return None, None


def synthesis(product_game, conc_game, ranks, values, n_players):
    # Compute max rank
    max_rank = None  # TODO

    # Iterate over all rank until initial state is winning
    for rank in range(max_rank):
        win_states, strategy = _strategy_given_rank(rank, product_game, conc_game, values, n_players)


if __name__ == '__main__':
    # Load game config
    # Load product game
    # Load concurrent game
    # Load rank function
    # Load values
    # Run synthesis
    pass
