import mujoco

model = mujoco.MjModel.from_xml_path("hello.xml")
data = mujoco.MjData(model)

print("nq =", model.nq)
print("nv =", model.nv)

while data.time < 2.0:

    mujoco.mj_step(model, data)

    print(
        "time =", round(data.time, 3),
        "qpos =", data.qpos
    )