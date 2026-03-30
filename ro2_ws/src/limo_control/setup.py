from setuptools import find_packages, setup

package_name = 'limo_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/param', ['param/limo_params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='marco',
    maintainer_email='marco.misseroni@studenti.unitn.it',
    description='Package for controlling the LIMO robot',
    license='Apache License 2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'EKF_node = limo_control.EKF_node:main',
            'MPC_node = limo_control.MPC_node:main',
        ],
    },
)
