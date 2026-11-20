from setuptools import setup

setup(
    name="worldsim_core",
    version="0.2.0",
    packages=["worldsim_core"],
    data_files=[("share/ament_index/resource_index/packages", ["resource/worldsim_core"]),
                ("share/worldsim_core", ["package.xml"])],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Local workspace maintainer",
    maintainer_email="maintainer@localhost.localdomain",
    description="Core of the local world simulator: engine, robot model, action mapping, teleoperation.",
    license="See upstream terms for bundled model code and weights",
    tests_require=["pytest"],
)
