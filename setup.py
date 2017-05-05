from setuptools import setup, find_packages

setup(name="teafacto",
      description="teafacto",
      author="ldgn",
      author_email="ldgn@ldgn.io",
      install_requires=["Theano"],
      packages=find_packages(),
      entry_points={
            'nose.plugins.0.10': [
                  'with-theanoinit = teafacto.theanotestplugin:TheanoConfigNosePlugin',
            ]
        },
      )
