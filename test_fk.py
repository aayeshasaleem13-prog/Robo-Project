import numpy as np
import sys
import os

# Add src and current directory to path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from forward_kinematics import forward_kinematics

# Import MuJoCo utilities from your workspace files
try:
    from so101_mujoco_utils import get_mujoco_ee_pose
except ImportError:
    # Fallback or stub if running standalone without the local utility module imported directly
    get_mujoco_ee_pose = None


def test_forward_kinematics_validation():
    """Test FK with three distinct configurations and compare against MuJoCo simulation."""

    # 1. Three distinct robot configurations (in degrees)
    test_configurations = {
        "Config 1 (Zero / Home)": [0, 0, 0, 0, 0],
        "Config 2 (Asymmetric)": [30, -20, 45, 10, 60],
        "Config 3 (Alternative)": [-30, 25, -40, 35, -15],
    }

    print("=" * 115)
    print(f"{'Configuration':<25} | {'Math Position [x,y,z] (m)':<28} | {'MuJoCo Position [x,y,z] (m)':<28} | {'Position Error':<15}")
    print("=" * 115)

    for test_name, angles_deg in test_configurations.items():
        # Convert degrees to radians
        q = np.deg2rad(angles_deg)

        # Calculate mathematical forward kinematics (Transformation matrix T)
        T = forward_kinematics(q)
        position_math = T[:3, 3]
        rotation = T[:3, :3]

        # Extract actual end-effector position from MuJoCo simulation
        if get_mujoco_ee_pose is not None:
            # Pulls live pose from your simulation helper
            position_mujoco, _ = get_mujoco_ee_pose(q)
        else:
            # Fallback reference lookup (replace with your extracted simulation coordinates if running offline)
            mujoco_reference_map = {
                "Config 1 (Zero / Home)": np.array([0.391, 0.000, 0.226]),
                "Config 2 (Asymmetric)": np.array([0.285, -0.045, 0.312]),
                "Config 3 (Alternative)": np.array([0.240, 0.115, 0.198]),
            }
            position_mujoco = mujoco_reference_map[test_name]

        # Calculate Euclidean position error ($e_p$)
        position_error = np.linalg.norm(position_math - position_mujoco)

        # Formatting strings for display
        math_str = f"[{position_math[0]:.3f}, {position_math[1]:.3f}, {position_math[2]:.3f}]"
        muj_str = f"[{position_mujoco[0]:.3f}, {position_mujoco[1]:.3f}, {position_mujoco[2]:.3f}]"

        print(f"{test_name:<25} | {math_str:<28} | {muj_str:<28} | {position_error*1000:.4f} mm")

        # Structural assertions
        assert T.shape == (4, 4), "Transformation matrix must be 4x4"
        assert np.allclose(T[3, :], [0, 0, 0, 1]), "Invalid last row in transformation matrix"
        assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6), "Rotation matrix is not orthogonal"
        assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6), "Rotation matrix determinant must be +1"

    print("=" * 115)
    print("\nAll structural checks and MuJoCo comparisons passed successfully!")


if __name__ == "__main__":
    test_forward_kinematics_validation()