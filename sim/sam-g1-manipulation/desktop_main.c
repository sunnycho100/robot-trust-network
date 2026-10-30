// A bundle executable keeps Qt in one native Mac app, with the project's venv.
#include <dlfcn.h>
#include <stdio.h>

int main(int argc, char **argv) {
  if (argc < 4) {
    fputs("Launch Robot Lab with: python view.py agent\n", stderr);
    return 2;
  }
  void *python = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
  if (!python) {
    fprintf(stderr, "Cannot load project Python: %s\n", dlerror());
    return 1;
  }
  int (*python_main)(int, char **) = dlsym(python, "Py_BytesMain");
  if (!python_main) {
    fputs("Project Python does not export Py_BytesMain\n", stderr);
    return 1;
  }
  return python_main(argc - 2, argv + 2);
}
