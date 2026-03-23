import os
import subprocess

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_git_commit_number():
    if not os.path.exists('.git'):
        return '0000000'

    cmd_out = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    git_commit_number = cmd_out.stdout.decode('utf-8')[:7]
    return git_commit_number


def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources]
    )
    return cuda_ext


def write_version_to_file(version, target_file):
    with open(target_file, 'w') as f:
        print('__version__ = "%s"' % version, file=f)


if __name__ == '__main__':
    version = '0.2.0+%s' % get_git_commit_number()
    write_version_to_file(version, 'pcseg/version.py')

    setup(
        name='pcseg',
        version=version,
        description='PCSeg',
        install_requires=[
            'numpy',
            'tensorboardX',
            'easydict',
            'pyyaml',
            'tqdm',
            'SharedArray',
            # 'spconv',  # spconv has different names depending on the cuda version
        ],

        author='Jihan Yang',
        author_email='jihanyang13@gmail.com',
        license='Apache License 2.0',
        packages=find_packages(exclude=['tools', 'data', 'output']),
        include_package_data=True,
        package_data={
            "pcseg": [
                # softgroup
                "external_libs/softgroup_ops/ops/*.py",
                "external_libs/softgroup_ops/ops/*.so",

                # pool_by_idx
                "ops/pool_by_idx/*.py",
                "ops/pool_by_idx/*.so",
            ],
        },
        cmdclass={
            'build_ext': BuildExtension,
        },
        ext_modules=[
            # --------------------------------
            # Pool_by_idx (already present)
            # --------------------------------
            make_cuda_ext(
                name='pool_by_idx',
                module='pcseg.ops.pool_by_idx',
                sources=[
                    'src/avg_pool_by_idx.cpp',
                    'src/avg_pool_by_idx_kernel.cu',
                ]
            ),
            # --------------------------------
            # SoftGroup (ADD THIS)
            # --------------------------------
            CUDAExtension(
                name='pcseg.external_libs.softgroup_ops.ops.softgroup_ops',
                sources=[
                    'pcseg/external_libs/softgroup_ops/ops/src/softgroup_api.cpp',
                    'pcseg/external_libs/softgroup_ops/ops/src/softgroup_ops.cpp',
                    'pcseg/external_libs/softgroup_ops/ops/src/cuda.cu',
                ],
                extra_compile_args={
                    'cxx': ['-g'],
                    'nvcc': ['-O2']
                },
            ),
        ],
    )
