"""
Test X11 GUI - keeps window open longer
"""
import time
import numpy as np

def main():
    print("=" * 50)
    print("X11 GUI Test")
    print("=" * 50)

    import pybullet as p

    print("\nCreating PyBullet GUI directly...")

    # Connect with GUI
    physics_client = p.connect(p.GUI)

    if physics_client < 0:
        print("ERROR: Could not connect to PyBullet GUI")
        return

    print(f"Connected! Client ID: {physics_client}")

    # Add a simple object to see
    p.setGravity(0, 0, -10)

    # Create a sphere
    sphere = p.createCollisionShape(p.GEOM_SPHERE, radius=0.5)
    sphere_body = p.createMultiBody(
        baseMass=1,
        baseCollisionShapeIndex=sphere,
        basePosition=[0, 0, 2]
    )

    # Create ground
    ground = p.createCollisionShape(p.GEOM_PLANE)
    ground_body = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=ground,
        basePosition=[0, 0, 0]
    )

    print("\n" + "=" * 50)
    print("Window should now be visible!")
    print("Look for a PyBullet window with a sphere")
    print("=" * 50)
    print("\nRunning simulation for 30 seconds...")
    print("Press Ctrl+C to stop early\n")

    try:
        for i in range(900):  # 30 seconds at 30 FPS
            p.stepSimulation()
            time.sleep(1/30)

            if i % 90 == 0:
                pos, _ = p.getBasePositionAndOrientation(sphere_body)
                print(f"Time: {i//30}s, Sphere position: z={pos[2]:.2f}")

    except KeyboardInterrupt:
        print("\nStopped by user")

    print("\nClosing...")
    p.disconnect()
    print("Done!")


if __name__ == "__main__":
    main()
