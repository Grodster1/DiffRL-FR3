from setuptools import find_packages, setup

package_name = 'franka_diffusion'

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
    description = 'Diffusion Policy (U-Net 1D + FiLM, DDPM/DDIM)',
    license = 'MIT',
)