from setuptools import find_packages, setup

package_name = 'franka_task'

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
    description = 'Zadanie Pick and Place dla FR3: środowisko Gymnasium, DLS-IK, ekspert, ewaluacja',
    license = 'MIT',
    entry_points = {
        'console_scripts' : [
            'collect_demos = franka_task.collect_demos:collect_demos',
        ],
    },
)
