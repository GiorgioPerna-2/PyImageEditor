from glob import glob
project_lines = 0
for filename in glob("./**/*.py", recursive=True):
    if filename != "./line-counter.py":
        with open(filename, "r") as f:
            lines = len(f.readlines())
            print(f"{filename = }: {lines = }")
            project_lines += lines
print(f"The number of lines in this project are: {project_lines}")
        