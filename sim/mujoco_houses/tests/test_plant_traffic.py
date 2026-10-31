import mujoco

from unitree_demo.flexible_factory.plant_traffic import YIELD_TIMEOUT_S, Delivery

# One TurtleBot-style courier with a person parked just ahead of it.
SCENE = """<mujoco><worldbody>
  <body name="tb0_base_link"><freejoint/><geom size=".1"/>
    <body name="tb0_left_wheel"><joint name="left" type="hinge"/><geom size=".03"/></body>
    <body name="tb0_right_wheel"><joint name="right" type="hinge"/><geom size=".03"/></body>
  </body>
  <body name="person" pos=".4 0 0"><geom size=".1"/></body>
</worldbody><actuator>
  <motor name="tb0_left_wheel_motor" joint="left" ctrlrange="-5 5"/>
  <motor name="tb0_right_wheel_motor" joint="right" ctrlrange="-5 5"/>
</actuator></mujoco>"""


def test_a_blocked_courier_reports_one_yield_timeout_and_keeps_holding():
    model = mujoco.MjModel.from_xml_string(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    courier, person, events = Delivery(model, 0, "turtlebot3_waffle"), model.body("person").id, []
    for step in range(int(2 * YIELD_TIMEOUT_S) + 1):
        data.time = float(step)
        courier.update(model, data, [courier], [person], events)
        assert courier.yielding and abs(data.ctrl[0] + data.ctrl[1]) < 1e-9  # may turn, never drives on
        timeouts = [e for e in events if e["event"] == "YIELD_TIMEOUT"]
        assert len(timeouts) == (step >= YIELD_TIMEOUT_S)
    assert timeouts[0]["t"] == YIELD_TIMEOUT_S and timeouts[0]["robot"] == "tb0"

    # Once the path clears, the next block starts a fresh wait.
    def step_at(time, position):
        model.body_pos[person] = position
        mujoco.mj_forward(model, data)
        data.time = time
        courier.update(model, data, [courier], [person], events)

    now = data.time
    step_at(now + 1, (5, 5, 0))
    assert not courier.yielding
    step_at(now + 2, (.4, 0, 0))
    step_at(now + 2 + YIELD_TIMEOUT_S - 1, (.4, 0, 0))
    assert len([e for e in events if e["event"] == "YIELD_TIMEOUT"]) == 1
    step_at(now + 2 + YIELD_TIMEOUT_S, (.4, 0, 0))
    assert len([e for e in events if e["event"] == "YIELD_TIMEOUT"]) == 2
