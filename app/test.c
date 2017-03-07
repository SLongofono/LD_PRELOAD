#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE	256

int main (void)
{
	int *my_int_array, i;

	/* Allocate an array of integers */
	my_int_array = (int *) malloc (ARRAY_SIZE * sizeof (int));

	/* Check if the returned pointer is valid */
	if (my_int_array) {
		/* Populate the array */
		for (i = 0; i < ARRAY_SIZE; ++i) {
			my_int_array[i] = i;
		}

		/* Print some info about the array */
		printf ("START (my_int_array) : %p\n", my_int_array);
		printf ("END   (my_int_array) : %p\n", &my_int_array[ARRAY_SIZE]);
	} else {
		/* This is an error */
		printf ("Failed to allocate!\n");
	}

	/* Free the array */
	free (my_int_array);

	printf ("A random number : %d\n", (int) rand ());

	/* Sayonara! */
	return 0;
}
