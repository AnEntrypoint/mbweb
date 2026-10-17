/* Dynamic loading is unavailable on WASI. GGML's backend registry references dlopen and
 * dlclose unconditionally, but this build links GGML statically with GGML_BACKEND_DL=OFF, so
 * no backend library is ever opened. These stubs report failure if that invariant breaks. */

#include <stddef.h>

void * dlopen(const char * file, int mode) {
    (void)file;
    (void)mode;
    return NULL;
}

void * dlsym(void * handle, const char * symbol) {
    (void)handle;
    (void)symbol;
    return NULL;
}

int dlclose(void * handle) {
    (void)handle;
    return 0;
}

char * dlerror(void) {
    return "dynamic loading is unavailable in the WASI component build";
}
