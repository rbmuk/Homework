#include <stdio.h>

int main() {
    int p = 5;
    for (int a = 0; a < p; ++a) {
        for (int b = 0; b < p; ++b) {
            for (int i = 0; i < p; ++i) {
                printf("ai+b: %d ", (a*i+b)%p);
            }
            printf("\n");
        }
    }

    return 0;
}
