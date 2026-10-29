from setuptools import find_packages, setup

package_name = 'franka_rl'

setup(
    name = package_name,
    version = '0.0.1',
    packages = find_packages(exclude=['test']),
    data_files = [
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires = ['setuptools'],
    zip_safe = True,
    maintainer = 'Wiktor',
    maintainer_email = "wiktorwszedybyl1@gmail.com",
    description = 'SAC i PPO (stable-baselines3) na zadaniu z franka_task',
    license = 'MIT',
    entry_points = {
        'console_scripts' : [
            'train_sac = franka_rl.train_sac:train_sac',
            'train_ppo = franka_rl.train_ppo:train_ppo',
            'random_baseline = franka_rl.random_baseline:random_baseline',
            'evaluate_rl = franka_rl.evaluate_rl:evaluate_rl'
        ],
    },
)