#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Create a function pointer to libc function types */
typedef void* (*libc_malloc_t)(size_t memory);
typedef void  (*libc_free_t)(void *mem_ptr);
typedef int   (*libc_rand_t)(void);

/*
 * malloc
 *
 * @memory	: Size in bytes of memory to allocate
 * $void*	: Pointer to the allocated memory
 */
void* malloc (size_t memory)
{
	static libc_malloc_t orig_malloc = NULL;

	/* Get the pointer to libc-defined free function */
	orig_malloc = (libc_malloc_t) dlsym (RTLD_NEXT, "malloc");

	/* Print some wrapper messages */
	// printf ("Malloced %d bytes!\n", (int) memory);

	/* Call the libc malloc */
	return orig_malloc (memory);
}

/*
 * free
 *
 * @mem_ptr	: Pointer to the memory area which has to be freed
 */
void free (void *mem_ptr)
{
	static libc_free_t orig_free = NULL;

	/* Get the pointer to libc-defined free function */
	orig_free = (libc_free_t) dlsym (RTLD_NEXT, "free");

	/* Print some wrapper messages before freeing memory */
	// printf ("Freeing Memory : %p\n", mem_ptr);

	/* Free the memory */
	orig_free (mem_ptr);

	/* Nothing more to do */
	return;
}

/*
 * rand
 *
 * $int		: Random number to be returned
 */
int rand ()
{
	static libc_rand_t orig_rand = NULL;
	int scaled_rand;

	/* Get the pointer to libc-defined rand function */
	orig_rand = (libc_rand_t) dlsym (RTLD_NEXT, "rand");

	/* Seed the random number generator */
	srand (time (NULL));

	/* Get a random number from libc */
	scaled_rand = orig_rand () % 100;

	/* Override random number generator */
	return scaled_rand;
}
