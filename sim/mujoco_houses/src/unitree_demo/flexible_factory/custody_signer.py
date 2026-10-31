"""Opt-in: each robot named in a custody event signs it with network/protocol.py.

    factory.bus.subscribers.append(CustodySigner({resource_id: Ed25519PrivateKey}))

Check records with protocol.Verifier(signer.receiver, registry). The scheduler never
imports this module, so it still runs without network/.
"""
from ..trust_session import load_trust

CUSTODY_EVENTS={'MEASURED_PICKUP','MEASURED_DELIVERY','TEAM_FORMED','TEAM_DISSOLVED'}

class CustodySigner:
    def __init__(self,keys,receiver='ledger@factory'):
        # ponytail: one process holds every robot's key because every robot is simulated here; on hardware each robot signs its own ack
        self.protocol,_=load_trust(); self.keys=keys; self.receiver=receiver; self.records=[]

    def __call__(self,row):
        if row['event'] not in CUSTODY_EVENTS: return
        for robot in row['resources']:
            self.records.append(self.protocol.make_message(self.keys[robot],robot,self.receiver,row['event'],**row))
