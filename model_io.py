def make_agent(saved, resource_max, network_budget):
    from algo.h2ppo import HPPO

    agent = HPPO(
        saved["num_states"],
        int(saved["num_shards"]),
        bmin=saved["bmin"],
        bmax=saved["bmax"],
        architecture=saved["architecture"],
        control_mode=saved["control_mode"],
        control_architecture=saved.get("control_architecture", "dense"),
        placement_ratio=saved["placement_ratio"],
        resource_max=resource_max,
        network_resource_budget=network_budget,
        placement_guard_cv=None,
        placement_cross_target=None,
    )
    agent.load_model(saved["actor"], saved["critic"])
    return agent
