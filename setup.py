from setuptools import setup, find_packages

setup(
    name='taylordiff',  # Replace with your package name
    version='0.1.0',  # Replace with your version
    description='A brief description of your project',
    author='Abhinav Muraleedharan',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/taylordiff',  # Replace with your repository URL
    packages=find_packages(include=['src', 'src.*', 'models', 'models.*']),
    install_requires=[
        # List your project dependencies here
        # e.g., 'numpy', 'pandas', 'scikit-learn', etc.
    ],
    extras_require={
        'dev': [
            'pytest',  # Add any development dependencies here
            'sphinx',  # For documentation
        ],
    },
    python_requires='>=3.6',  # Specify the Python version required
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
