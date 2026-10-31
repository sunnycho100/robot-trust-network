"""Factory custody/dispatch regression tests. ManualBackend is NOT a physics sim."""
import base64,zlib
import pytest
from cryptography.hazmat.primitives.serialization import Encoding,PublicFormat
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from unitree_demo.trust_session import load_trust
from unitree_demo.flexible_factory.config import load_config
from unitree_demo.flexible_factory.buffer import Buffer
from unitree_demo.flexible_factory.job import Job,Priority
from unitree_demo.flexible_factory.resource_selector import Resource
from unitree_demo.flexible_factory.engine import Factory
from unitree_demo.flexible_factory.custody_signer import CustodySigner

class ManualBackend:
    def __init__(self): self.signals=[]; self.started={}
    def can_execute(self,task,resources): return True
    def start(self,task,resources): self.started[task.id]=resources; return True
    def poll(self): result,self.signals=self.signals,[]; return result
    def phase(self,task_id): return 'RUNNING'

def factory(resources=()):
    backend=ManualBackend(); return Factory(load_config(),resources,backend),backend

def test_buffer_atomic_capacity_and_duplicate_identity():
    b=Buffer('B',1)
    assert b.reserve_in('PART-A','T1')
    assert not b.reserve_in('PART-B','T2')
    with pytest.raises(ValueError): b.put('PART-B','T1')
    b.put('PART-A','T1')
    with pytest.raises(ValueError): b.put('PART-A')
    assert b.reserve_out('PART-A','T1') and not b.reserve_out('PART-A','T2')
    with pytest.raises(ValueError): b.take('PART-A','WRONG')

def test_no_custody_change_without_physical_ack():
    r=Resource('AMR',frozenset(['tote_transport']),max_payload_kg=.1)
    f,b=factory([r]); f.add_job(Job('P','B',('CNC','QC'),'RECEIVING'))
    f.tick(0); task=next(iter(f.tasks.values()))
    assert task.status=='ASSIGNED' and f.jobs['P'].location=='RECEIVING'
    with pytest.raises(ValueError): f.acknowledge_delivery(task.id)
    b.signals=[(task.id,'PICKED')]; f.tick(1)
    assert f.jobs['P'].location=='ON:AMR' and not f.buffers['RECEIVING'].items
    b.signals=[(task.id,'DELIVERED')]; f.tick(2)
    assert f.jobs['P'].location=='CNC_INPUT' and r.state=='AVAILABLE'

def test_priority_over_distance_and_incompatible_robot_stays_free():
    r=Resource('FLEX',frozenset(['tote_transport','rework_transfer','exception_handling']),max_payload_kg=2)
    wrong=Resource('ARM',frozenset(['machine_tending']),max_payload_kg=5,fixed=True)
    f,b=factory([r,wrong])
    f.add_job(Job('NORMAL','B',('CNC',),'RECEIVING'))
    f.add_job(Job('REJECT','B',('QC',),'QC_OUTPUT'))
    a=f.submit('NORMAL','NORMAL','RECEIVING','CNC_INPUT',Priority.WIP,'tote_transport')
    urgent=f.submit('REWORK','REJECT','QC_OUTPUT','REWORK',Priority.REWORK,'rework_transfer')
    f.tick(0)
    assert urgent.status=='ASSIGNED' and a.status=='QUEUED' and wrong.state=='AVAILABLE'

def test_team_reservation_is_atomic_and_dissolves_after_delivery():
    resources=[Resource(n,frozenset(['team_transport']),max_payload_kg=1,profiles=frozenset(['three_frame'])) for n in ('A','B','C')]
    f,b=factory(resources)
    f.add_job(Job('MODULE','D',(), 'ASSEMBLY_LARGE_OUTPUT',mass_kg=3,dimensions=(1.12,1.02,.16),required_grasp_points=3))
    f.tick(0); task=next(iter(f.tasks.values()))
    assert len(task.resources)==3 and all(r.state=='RESERVED' for r in resources)
    b.signals=[(task.id,'PICKED'),(task.id,'DELIVERED')]; f.tick(1)
    assert all(r.state=='AVAILABLE' for r in resources)
    assert any(e['event']=='TEAM_DISSOLVED' for e in f.bus.events)

def test_station_output_full_blocks_and_creates_recovery_priority():
    f,b=factory()
    for i in range(3): f.add_job(Job(f'OUT{i}','B',('QC',),'CNC_OUTPUT'))
    f.add_job(Job('RAW','B',('CNC','QC'),'CNC_INPUT'))
    f.tick(0)
    assert f.stations['CNC'].state=='BLOCKED'
    recovery=[t for t in f.tasks.values() if t.kind=='CNC_OUTPUT_RECOVERY']
    assert recovery and all(t.priority==Priority.BLOCKED for t in recovery)
    assert not any(t.kind=='PROCESS' and t.job_id=='RAW' for t in f.tasks.values())

def test_qc_failure_routes_to_rework_and_does_not_skip_repair():
    name=next(f'PART-{i:03d}' for i in range(100) if zlib.crc32(f'PART-{i:03d}'.encode())%10==7)
    qc=Resource('QC',frozenset(['inspection_support']),max_payload_kg=.1)
    flex=Resource('FLEX',frozenset(['rework_transfer']),max_payload_kg=.1)
    repair=Resource('PANDA',frozenset(['rework']),max_payload_kg=.1)
    f,b=factory([qc,flex,repair]); f.add_job(Job(name,'B',('QC','PACKAGING'),'QC_INPUT'))
    f.tick(0); task=next(t for t in f.tasks.values() if t.kind=='PROCESS')
    b.signals=[(task.id,'PICKED'),(task.id,'DELIVERED')]; f.tick(1)
    assert f.rejects==1 and f.jobs[name].stage==0
    transfer=next(t for t in f.tasks.values() if t.kind=='REWORK_TRANSFER')
    b.signals=[(transfer.id,'PICKED'),(transfer.id,'DELIVERED')]; f.tick(2)
    rework=next(t for t in f.tasks.values() if t.metadata.get('station')=='REWORK')
    assert rework.status=='ASSIGNED'
    b.signals=[(rework.id,'PICKED'),(rework.id,'DELIVERED')]; f.tick(3)
    assert f.jobs[name].reworks==1
    assert any(t.kind=='QC_RETURN' for t in f.tasks.values())

def test_assembly_requires_a_real_component_pickup():
    arm=Resource('PANDA',frozenset(['assembly']),max_payload_kg=.1)
    f,b=factory([arm]); f.add_job(Job('KIT','C',('ASSEMBLY','QC'),'ASSEMBLY_INPUT'))
    f.tick(0); assert f.stations['ASSEMBLY'].state=='STARVED'
    f.add_job(Job('SCREW','COMPONENT_B',(),'COMPONENTS'))
    f.tick(1); task=next(t for t in f.tasks.values() if t.kind=='PROCESS')
    f.acknowledge_pickup(task.id)
    with pytest.raises(ValueError): f.acknowledge_delivery(task.id)
    f.acknowledge_component(task.id); f.acknowledge_delivery(task.id)
    assert f.jobs['KIT'].components==['SCREW'] and not f.buffers['COMPONENTS'].items

def test_motion_fault_keeps_custody_and_destination_slot_reserved():
    r=Resource('AMR',frozenset(['tote_transport']),max_payload_kg=.1)
    f,b=factory([r]); f.add_job(Job('P','B',('CNC',),'RECEIVING'))
    f.tick(0); task=next(iter(f.tasks.values()))
    b.signals=[(task.id,'PICKED'),(task.id,'FAULT')]; f.tick(1)
    assert task.status=='FAULT' and r.state=='FAULT_HOLD'
    assert f.jobs['P'].location=='ON:AMR'
    assert f.buffers['CNC_INPUT'].inbound[task.id]=='P'
    with pytest.raises(ValueError): f.acknowledge_delivery(task.id)

def test_insufficient_team_never_partially_reserves_resources():
    resources=[Resource(n,frozenset(['team_transport']),max_payload_kg=1,profiles=frozenset(['three_frame'])) for n in ('A','B')]
    f,b=factory(resources)
    f.add_job(Job('MODULE','D',(),'ASSEMBLY_LARGE_OUTPUT',mass_kg=3,required_grasp_points=3))
    f.tick(0)
    assert next(iter(f.tasks.values())).status=='QUEUED'
    assert all(r.state=='AVAILABLE' for r in resources) and not b.started

def test_three_product_types_can_be_in_different_stations_concurrently():
    resources=[Resource('CNC',frozenset(['machine_tending']),max_payload_kg=.1),
        Resource('ASSEMBLER',frozenset(['assembly']),max_payload_kg=.1),
        Resource('QC',frozenset(['inspection_support']),max_payload_kg=.1)]
    f,b=factory(resources)
    f.add_job(Job('HOUSING','A',('CNC','DEBURR','ASSEMBLY','QC','PACKAGING'),'CNC_INPUT'))
    f.add_job(Job('BRACKET','B',('QC','PACKAGING'),'QC_INPUT'))
    f.add_job(Job('KIT','C',('ASSEMBLY','QC','PACKAGING'),'ASSEMBLY_INPUT'))
    f.add_job(Job('INSERT','COMPONENT_B',(),'COMPONENTS'))
    f.tick(0)
    running=[t for t in f.tasks.values() if t.status=='ASSIGNED']
    assert {t.metadata['station'] for t in running}=={'CNC','ASSEMBLY','QC'}
    assert len({name for t in running for name in t.resources})==3

def test_custody_events_are_signed_by_each_robot_and_tampering_fails():
    protocol,_=load_trust()
    keys={n:Ed25519PrivateKey.generate() for n in ('A','B')}
    registry={n:{'public_key':base64.b64encode(k.public_key().public_bytes(Encoding.Raw,PublicFormat.Raw)).decode(),'status':'active'} for n,k in keys.items()}
    signer=CustodySigner(keys); verifier=protocol.Verifier(signer.receiver,registry)
    resources=[Resource(n,frozenset(['team_transport']),max_payload_kg=1,profiles=frozenset(['three_frame'])) for n in keys]
    f,b=factory(resources); f.bus.subscribers.append(signer)
    f.add_job(Job('MODULE','D',(),'ASSEMBLY_LARGE_OUTPUT',mass_kg=1.5,required_grasp_points=2))
    f.tick(0); task=next(iter(f.tasks.values()))
    b.signals=[(task.id,'PICKED')]; f.tick(1)
    b.signals=[(task.id,'DELIVERED')]; f.tick(2)
    events=('TEAM_FORMED','MEASURED_PICKUP','MEASURED_DELIVERY','TEAM_DISSOLVED')
    assert sorted((r['request'],r['from']) for r in signer.records)==sorted((e,n) for e in events for n in keys)
    assert all(verifier.check(r)==(True,'ok') for r in signer.records)
    delivery=next(r for r in signer.records if r['request']=='MEASURED_DELIVERY')
    assert delivery['job']=='MODULE' and delivery['destination']=='FINAL_STAGING'
    assert verifier.check({**delivery,'destination':'SHIPPING'})==(False,'bad signature')

def ship_assigned(f,b):
    b.signals=[(t.id,s) for t in f.tasks.values() if t.status=='ASSIGNED' for s in ('PICKED','DELIVERED')]
    f.tick(f.time+1)

def shipping_line():
    return factory([Resource('FG',frozenset(['finished_goods']),max_payload_kg=.1)])

def test_shipping_drains_so_packaging_does_not_stall_after_two_pallets():
    f,b=shipping_line(); n=f.config['pallet_capacity']
    for p in range(3):
        for i in range(n): f.add_job(Job(f'P{p}-{i:02d}','B',('SHIPPING',),'FINISHED'))
        for _ in range(n+1): ship_assigned(f,b)
    assert len(f.shipped)==3*n and not f.buffers['SHIPPING'].items
    assert f.stations['PACKAGING'].state!='BLOCKED'

def test_a_pallet_ships_whole_or_waits():
    f,b=shipping_line(); n=f.config['pallet_capacity']
    for i in range(f.buffers['SHIPPING'].capacity-n+1): f.add_job(Job(f'OLD{i}','B',(),'SHIPPING'))
    for i in range(n): f.add_job(Job(f'P{i:02d}','B',('SHIPPING',),'FINISHED'))
    f.tick(0)
    assert not any(t.kind=='FINISHED_GOODS' for t in f.tasks.values())
    # A part packed while a pallet is leaving waits for its own pallet.
    f,b=shipping_line()
    for i in range(n): f.add_job(Job(f'P{i:02d}','B',('SHIPPING',),'FINISHED'))
    f.tick(0); ship_assigned(f,b)
    f.add_job(Job('LATE','B',('SHIPPING',),'FINISHED')); f.tick(f.time+1)
    assert not any(t.job_id=='LATE' for t in f.tasks.values())
