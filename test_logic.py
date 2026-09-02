from logic_engine import KnowledgeBase


def test_forward_chaining():
    kb = KnowledgeBase()
    kb.tell_rule(['TargetVisible', 'HasDust'], 'SafeToEngage')
    kb.tell_rule(['SafeToEngage', 'BloodseekerMissing'], 'Retreat')

    kb.clear_facts()
    kb.tell_fact('TargetVisible')
    kb.tell_fact('HasDust')
    kb.forward_chain()
    assert 'SafeToEngage' in kb.facts
    assert 'Retreat' not in kb.facts

    kb.clear_facts()
    kb.tell_fact('TargetVisible')
    kb.tell_fact('HasDust')
    kb.tell_fact('BloodseekerMissing')
    kb.forward_chain()
    assert 'Retreat' in kb.facts


def test_astar_skips_infeasible_tile():
    from agent import SearchAgent

    agent = SearchAgent(active_algo='AStar')
    tile_percepts = {(1, 0): {'TargetVisible', 'HasDust', 'BloodseekerMissing'}}
    path = agent.astar_search((0, 0), (2, 0), [], (3, 2), tile_percepts=tile_percepts)

    assert path == ['Up', 'Right', 'Right', 'Down']


if __name__ == '__main__':
    test_forward_chaining()
    test_astar_skips_infeasible_tile()
    print('All logic engine tests passed.')