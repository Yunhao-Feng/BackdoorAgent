__all__ = ["attack_bench"]

def attack_bench(args):
    attack_type = getattr(args, "attack", "normal")
    if attack_type == "agentpoison":
        pass
        # from attack.agentpoison import AgentPoison
        # attacker = AgentPoison(args)
        # args = attacker.run()
    elif attack_type == "poisonedrag":
        from attack.poisonedrag import PoisonedRAG
        attacker = PoisonedRAG(args)
        args = attacker.run()
    elif attack_type == "badagent":
        from attack.badagent import BadAgent
        attacker = BadAgent(args)
        args = attacker.run()
    elif attack_type == "trojanrag":
        from attack.trojanrag import TrojanRAG
        attacker = TrojanRAG(args)
        args = attacker.run()
    elif attack_type == "badchain":
        from attack.badchain import BadChain
        attacker = BadChain(args)
        args = attacker.run()
    elif attack_type == "advagent":
        from attack.advagent import AdvAgent
        attacker = AdvAgent(args)
        args = attacker.run()
    elif attack_type == "demonagent":
        from attack.demonagent import DemonAgent
        attacker = DemonAgent(args)
        args = attacker.run()
    elif attack_type == "normal":
        pass
    else:
        raise NotImplementedError
    return args