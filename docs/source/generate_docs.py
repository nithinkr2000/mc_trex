import os

def generate_rst_files(package_name):
    """Generate .rst files for all modules in the package."""
    # Create modules.rst
    with open('modules.rst', 'w') as f:
        f.write(f'{package_name} Package\n')
        f.write('=' * (len(package_name) + 8) + '\n\n')
        f.write('.. toctree::\n')
        f.write('   :maxdepth: 4\n\n')
        f.write(f'   {package_name}\n')
    
    # Create package.rst
    with open(f'{package_name}.rst', 'w') as f:
        f.write(f'{package_name} Package\n')
        f.write('=' * (len(package_name) + 8) + '\n\n')
        f.write('.. automodule:: ' + package_name + '\n')
        f.write('   :members:\n')
        f.write('   :undoc-members:\n')
        f.write('   :show-inheritance:\n\n')
        f.write('Submodules\n')
        f.write('---------\n\n')
        f.write('.. toctree::\n\n')
        
        # Find all Python modules in the package
        package_dir = os.path.join('..', '..', package_name)
        for root, dirs, files in os.walk(package_dir):
            for file in files:
                if file.endswith('.py') and file != '__init__.py':
                    module_name = file[:-3]
                    rel_path = os.path.relpath(root, os.path.join('..', '..'))
                    full_module_name = f"{rel_path.replace(os.sep, '.')}.{module_name}"
                    
                    # Create module RST file
                    module_rst = f'{full_module_name}.rst'
                    with open(module_rst, 'w') as mf:
                        mf.write(f'{full_module_name} module\n')
                        mf.write('=' * (len(full_module_name) + 7) + '\n\n')
                        mf.write(f'.. automodule:: {full_module_name}\n')
                        mf.write('   :members:\n')
                        mf.write('   :undoc-members:\n')
                        mf.write('   :show-inheritance:\n')
                    
                    f.write(f'   {full_module_name}\n')

if __name__ == '__main__':
    # Replace with your package name
    generate_rst_files('mc_trex')
