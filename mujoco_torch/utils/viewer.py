import mujoco_py
import click

@click.command()
@click.argument('filename')
def main(filename):
    print("Viewing Mujoco XML", filename)
    model = mujoco_py.load_model_from_path(filename)
    sim = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)

    for i in range(100):
        sim.reset()
        for i in range(100):
            sim.step()
            viewer.render()

if __name__ == "__main__":
    main()
