from setuptools import setup, find_packages

setup(
    name='VAE_project',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # List your dependencies here, e.g.,
        'torch',
        'torchvision',
        'numpy',
        'matplotlib',
    ],
    # Add additional package metadata here, e.g., author, description
    author="Felix Watine",
    description="A VAE-based project with multiple models and training scripts.",
    # long_description=open('README.md').read(),
    # long_description_content_type='text/markdown',
    # url="https://github.com/yourusername/vae_project",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)