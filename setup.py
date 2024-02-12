from setuptools import setup, find_packages

setup(
       # the name must match the folder name 'verysimplemodule'
        name="ctfishpy", 
        version='0.1.6',
        author="Abdelwahab Kawafi",
        author_email="<akawafi3@gmail.com>",
        description='Zebrafish bone segmentation using deep learning.',
        # long_description=LONG_DESCRIPTION,
        package_dir={'ctfishpy': 'ctfishpy'},
        package_data={'ctfishpy': ['otolith_unet_221019.pt', 'jaw_unet_230124.pt']},
        packages=find_packages(),
        install_requires=[
                'numpy>=1.19',
                'matplotlib',
                'opencv-python-headless',
                'pandas',
                'matplotlib',
                'seaborn',
                'pathlib2',
                'h5py',
                'napari[pyside2]',
                'torch',
                'monai-weekly',
                'torchio',
                'albumentations',
                'tqdm',
                'scipy',
                'scikit-image',
                'scikit-learn',
                'tifffile',
                'python-dotenv',
                'pytest',
                'pydicom',
                ],
        extras_require={
                "dev": [
                        'neptune-client',
                        'ray[tune]',
                ],
        },
        keywords=['python', 'ctfishpy'],
)