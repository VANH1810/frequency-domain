import os
import subprocess
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_call(['cmake', '--version'])
        except FileNotFoundError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}', 
            f'-DPYTHON_EXECUTABLE={sys.executable}'
        ]

        
        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if sys.platform == "win32":
            cmake_args += ['-DCMAKE_GENERATOR_PLATFORM=$ENV:PLATFORM']
            if self.compiler.compiler_type == "msvc":
                build_args += ['--', '/m'] 
        
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j', str(os.cpu_count())] 

        
        env = os.environ.copy()

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        print(f"Configuring CMake in {self.build_temp} with args: {cmake_args}")
        
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)

        print(f"Building CMake project in {self.build_temp} with args: {build_args}")
        
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

        expected_abi_lib_path = self.get_ext_fullpath(ext.name) 
        
        if sys.platform == "win32":
            generic_lib_name = f"{ext.name}.pyd"
        else: 
            generic_lib_name = f"{ext.name}.so"
        
        
        generic_lib_path_in_extdir = os.path.join(extdir, generic_lib_name)

        if os.path.exists(expected_abi_lib_path):
            print(f"Found ABI-tagged library at {expected_abi_lib_path}")
            return 

        elif os.path.exists(generic_lib_path_in_extdir):
            print(f"Found generic library at {generic_lib_path_in_extdir}. Copying to {expected_abi_lib_path}")
            
            os.makedirs(os.path.dirname(expected_abi_lib_path), exist_ok=True)
            
            self.copy_file(generic_lib_path_in_extdir, expected_abi_lib_path)
            return
        
        else:
            print(f"Warning: Expected library not found at {expected_abi_lib_path} or {generic_lib_path_in_extdir}. Searching in build directory...")
            found_lib_in_build_temp = None
            for root, _, files in os.walk(self.build_temp):
                for f in files:                    
                    if f == generic_lib_name or (f.startswith(ext.name) and (f.endswith('.so') or f.endswith('.pyd') or f.endswith('.dylib'))):
                        found_lib_in_build_temp = os.path.join(root, f)
                        break
                if found_lib_in_build_temp:
                    break
            
            if found_lib_in_build_temp:
                print(f"Found library at {found_lib_in_build_temp}. Copying to {expected_abi_lib_path}")
                os.makedirs(os.path.dirname(expected_abi_lib_path), exist_ok=True)
                self.copy_file(found_lib_in_build_temp, expected_abi_lib_path)
                return
            else:
                
                raise RuntimeError(f"Could not find the built library for {ext.name} after CMake build in expected locations.")


setup(
    name='asilla_reader',
    version='0.1.0',
    author='Luu Ngoc Thanh', 
    author_email='thanhln@asilla.net', 
    description='A reader module for Asilla, built with C++ and CMake.',
    long_description='This package provides a C++ extension module for reading data, integrated via CMake.',
    ext_modules=[CMakeExtension('asilla_reader')],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
    install_requires=[
    ],
    include_package_data=True,
    py_modules=['asilla_reader'], 
)
