import subprocess

if __name__ == '__main__':
    for test_arg1 in [3, 2, 1]:
        print(f"running exp for test_arg1 = {test_arg1}")
        subprocess.run(["python", "./tools/main.py", "-c", str(test_arg1)])