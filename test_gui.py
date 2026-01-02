"""
Test X11 GUI visualization with PyBullet drone
"""
import time
import numpy as np

def main():
    print("Testing X11 connection...")

    # Test if X11 works
    try:
        import pybullet as p
        print("PyBullet imported successfully")
    except ImportError as e:
        print(f"PyBullet import error: {e}")
        return

    # Try to create GUI environment
    print("Creating drone environment with GUI...")
    try:
        from gym_pybullet_drones.envs.HoverAviary import HoverAviary
        from gym_pybullet_drones.utils.enums import ActionType, ObservationType

        env = HoverAviary(
            gui=True,  # GUI有効！
            obs=ObservationType.KIN,
            act=ActionType.RPM,
            pyb_freq=240,
            ctrl_freq=30,
        )
        print("GUI environment created successfully!")

    except Exception as e:
        print(f"Error creating GUI environment: {e}")
        print("\nTroubleshooting:")
        print("1. VcXsrvが起動していますか？")
        print("2. 'Disable access control'にチェックしましたか？")
        print("3. Windowsファイアウォールでブロックされていませんか？")
        return

    # Run random actions to see the drone
    print("\nRunning random actions for 10 seconds...")
    print("You should see a PyBullet window with a drone!")

    obs, info = env.reset()

    start_time = time.time()
    step = 0

    while time.time() - start_time < 10:
        # Random action (RPM values)
        action = np.array([[14000, 14000, 14000, 14000]])  # Hover RPM

        # Add some variation
        noise = np.random.uniform(-1000, 1000, (1, 4))
        action = action + noise

        obs, reward, terminated, truncated, info = env.step(action)
        step += 1

        time.sleep(0.033)  # ~30 FPS

        if step % 30 == 0:
            print(f"Step {step}, Reward: {reward:.2f}")

        if terminated or truncated:
            obs, info = env.reset()

    print("\nTest completed! Closing environment...")
    env.close()
    print("Done!")


if __name__ == "__main__":
    main()
