# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Plot Data Timestamps
#
# Plots sensor timestamps (cameras and IMU) against each other to look
# for delays and other variability in timing.


# %%
import os
import pandas as pd
import matplotlib.pyplot as plt

# # %matplotlib inline
# %matplotlib notebook


# %%
# Define directory to VIO output csv files as well as ground truth absolute poses.
vio_output_dir = ""
log_kimera_endpoint = False


# %%
# Extract end point
if log_kimera_endpoint:
    output_tv = os.path.join(os.path.expandvars(vio_output_dir), "traj_vio.csv")
    traj_vio = pd.read_csv(output_tv, sep=",", index_col=0)
    end_point = traj_vio.index[-1]

output_imu_file = os.path.join(os.path.expandvars(vio_output_dir), "imu_data.csv")
imu_data = pd.read_csv(output_imu_file, sep=",", index_col=0)


# %% [markdown]
# # Gyro Measurements


# %%
fig = plt.figure()
plt.plot(imu_data.index, imu_data.ang_vel_x)
plt.plot(imu_data.index, imu_data.ang_vel_y)
plt.plot(imu_data.index, imu_data.ang_vel_z)

if log_kimera_endpoint:
    plt.axvline(x=end_point)

plt.ylabel("Gyro Measurements")
plt.xlabel("Timestamps")
plt.legend()
plt.show()


# %% [markdown]
# # Accelerometer Measurements


# %%
fig = plt.figure()
plt.plot(imu_data.index, imu_data.lin_acc_x)
plt.plot(imu_data.index, imu_data.lin_acc_y)
plt.plot(imu_data.index, imu_data.lin_acc_z)

if log_kimera_endpoint:
    plt.axvline(x=end_point)

plt.ylabel("Accelerometer Measurements")
plt.xlabel("Timestamps")
plt.legend()
plt.show()


# %% [markdown]
# # Sensor vs. ROS Timestamps


# %%
fig = plt.figure()
plt.plot(imu_data.index, imu_data.clock)

if log_kimera_endpoint:
    plt.axvline(x=end_point)

plt.xlabel("Ros time")
plt.ylabel("IMU Timestamps")
plt.legend()
plt.show()


# %% [markdown]
# # IMU Timestamp Deltas


# %%
fig = plt.figure()

deltas = []
for i in range(1, len(imu_data.index)):
    delta = imu_data.index[i] - imu_data.index[i - 1]
    delta *= 1e-9
    if delta < 0:
        delta *= 100  # So we can see out-of-order timestamps
    deltas.append(delta)

plt.plot(imu_data.index[1:] * 1e-9, deltas, label="IMU deltas")

plt.xlabel("Timestamps in seconds")
plt.ylabel("Delta in seconds")
plt.legend()
plt.show()


# %% [markdown]
# # Left Camera Timestamp Deltas


# %%
output_infra1_deltas_file = os.path.join(
    os.path.expandvars(vio_output_dir), "timestamps_infra1.csv"
)
infra1_deltas_data = pd.read_csv(output_infra1_deltas_file, sep=",", index_col=0)
fig = plt.figure()

deltas = []
for i in range(1, len(infra1_deltas_data.index)):
    delta = infra1_deltas_data.index[i] - infra1_deltas_data.index[i - 1]
    delta *= 1e-9
    if delta < 0:
        delta *= 100  # So we can see out-of-order timestamps
    deltas.append(delta)

plt.plot(infra1_deltas_data.index[1:] * 1e-9, deltas, label="Infra1 deltas")

plt.xlabel("Timestamps in seconds")
plt.ylabel("Delta in seconds")
plt.legend()
plt.show()


# %% [markdown]
# # Right Camera Timestamp Deltas


# %%
output_infra2_deltas_file = os.path.join(
    os.path.expandvars(vio_output_dir), "timestamps_infra2.csv"
)
infra2_deltas_data = pd.read_csv(output_infra2_deltas_file, sep=",", index_col=0)
fig = plt.figure()

deltas = []
for i in range(1, len(infra2_deltas_data.index)):
    delta = infra2_deltas_data.index[i] - infra2_deltas_data.index[i - 1]
    delta *= 1e-9
    if delta < 0:
        delta *= 100  # So we can see out-of-order timestamps
    deltas.append(delta)

plt.plot(infra2_deltas_data.index[1:] * 1e-9, deltas, label="Infra2 deltas")

plt.xlabel("Timestamps in seconds")
plt.ylabel("Delta in seconds")
plt.legend()
plt.show()


# %% [markdown]
# # All Timestamp Deltas


# %%
output_infra1_deltas_file = os.path.join(
    os.path.expandvars(vio_output_dir), "timestamps_infra1.csv"
)
infra1_deltas_data = pd.read_csv(output_infra1_deltas_file, sep=",", index_col=0)
fig = plt.figure()

deltas = []
for i in range(1, len(infra1_deltas_data.index)):
    delta = infra1_deltas_data.index[i] - infra1_deltas_data.index[i - 1]
    delta *= 1e-9
    if delta < 0:
        delta *= 100  # So we can see out-of-order timestamps
    deltas.append(delta)

plt.plot(infra1_deltas_data.index[1:] * 1e-9, deltas, label="Infra1 deltas")

output_infra2_deltas_file = os.path.join(
    os.path.expandvars(vio_output_dir), "timestamps_infra2.csv"
)
infra2_deltas_data = pd.read_csv(output_infra2_deltas_file, sep=",", index_col=0)

deltas = []
for i in range(1, len(infra2_deltas_data.index)):
    delta = infra2_deltas_data.index[i] - infra2_deltas_data.index[i - 1]
    delta *= 1e-9
    if delta < 0:
        delta *= 100  # So we can see out-of-order timestamps
    deltas.append(delta)

plt.plot(infra2_deltas_data.index[1:] * 1e-9, deltas, label="Infra2 deltas")

plt.xlabel("Timestamps in seconds")
plt.ylabel("Delta in seconds")
plt.legend()
plt.show()
