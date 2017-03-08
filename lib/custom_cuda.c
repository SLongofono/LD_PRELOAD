#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Include CUDA header */
#include <cuda.h>
#include <vector_types.h>

/* Create typedefs to the target CUDA APIs */
typedef cudaError_t	(*cudaLaunch_t)(const char*);
typedef cudaError_t	(*cudaConfigureCall_t)(dim3, dim3, size_t, cudaStream_t);
typedef cudaError_t	(*cudaSetupArgument_t)(const void*, size_t, size_t);
typedef void		(*cudaRegisterFunction_t)(void**, const char*, char*,
						  const char*, int, uint3*,
						  uint3*, dim3*, dim3*,
						  int*);

/* Statically declare function pointers for real CUDA APIs */
static cudaLaunch_t 		orig_cudaLaunch = NULL;
static cudaConfigureCall_t 	orig_cudaConfigureCall = NULL;
static cudaSetupArgument_t	orig_cudaSetupArgument = NULL;
static cudaRegisterFunction_t 	orig_cudaRegisterFunction = NULL;

/*
 * cudaLaunch
 *
 * @func		: Name of the CUDA kernel to launch
 * $cudaError_t		: Enumerated type specifying the CUDA error
 */
extern "C"
cudaError_t cudaLaunch (const char *func)
{
	if (!orig_cudaLaunch) {
		/* Get the pointer to CUDA-defined launch function */
		orig_cudaLaunch = (cudaLaunch_t) dlsym (RTLD_NEXT, "cudaLaunch");
	}

	/* Print the function name to stdout */
	printf ("Launching kernel!\n");

	/* Call the cuda laucnh function */
	return orig_cudaLaunch (func);
}

/*
 * cudaConfigureCall
 *
 * @gridDim		: Dimensions of the thread grid
 * @blockDim		: Dimensions of the thread block
 * @sharedMem		: !Amount of shared memory
 * @stream		: ID of the CUDA stream
 * $cudaError_t		: Enumerated type specifying the CUDA error
 */
extern "C"
cudaError_t cudaConfigureCall (dim3		gridDim,
			       dim3		blockDim,
			       size_t		sharedMem = 0,
			       cudaStream_t	stream = 0)
{
	if (!orig_cudaConfigureCall) {
		/* Get the pointer to the CUDA-defined configure function */
		orig_cudaConfigureCall = (cudaConfigureCall_t) dlsym (RTLD_NEXT, "cudaConfigureCall");
	}

	/* Print the invocation message */
	printf ("Configuring CUDA...\n");

	/* Call the cuda configure function */
	return orig_cudaConfigureCall (gridDim, blockDim, sharedMem, stream);
}

/*
 * cudaSetupArgument
 *
 * @arg			: Argument to pass to CUDA kernel
 * @size		: Size of the argument
 * @offset		: Offset of the argument in the kernel stack
 * $cudaError_t		: Enumerated type specifying the CUDA error
 */
extern "C"
cudaError_t cudaSetupArgument (const void	*arg,
			       size_t		size,
			       size_t		offset)
{
	if (!orig_cudaSetupArgument) {
		/* Get the pointer to the CUDA-defined setup function */
		orig_cudaSetupArgument = (cudaSetupArgument_t) dlsym (RTLD_NEXT, "cudaSetupArgument");
	}

	/* Print the invocation message */
	printf ("Setting up CUDA...\n");

	/* Call the cuda setup function */
	return orig_cudaSetupArgument(arg, size, offset);
}

/*
 * cudaRegisterFunction
 *
 * @fatCubinHandle	: !PARAM UNKNOWN
 * @hostFun		: !Function to be invoked on host
 * @deviceFun		: !Function to be invoked on device
 *
 * @deviceName		: !Name of the device
 * @threadLimit		: !Maximum number of concurrent threads on the device
 * @tid			: !Identifier for thread
 *
 * @bid			: !Identifier for thread block
 * @bDim		: !Block dimensions
 * @gDim		: !Grid dimensions
 *
 * @wSize		: !Window size
 */
extern "C"
void __cudaRegisterFunction (void 	**fatCubinHandle,
			     const char	*hostFun,
			     char	*deviceFun,
			     const char	*deviceName,
			     int	thread_limit,
			     uint3	*tid,
			     uint3	*bid,
			     dim3	*bDim,
			     dim3	*gDim,
			     int	*wSize)
{
	if (!orig_cudaRegisterFunction) {
		/* Get the pointer to the CUDA-defined register function */
		orig_cudaRegisterFunction = (cudaRegisterFunction_t) dlsym (RTLD_NEXT, "__cudaRegisterFunction");
	}

	/* Print the invocation message */
	printf ("Registering Device Function : %s\n", deviceFun);

	/* Call the cuda register function */
	orig_cudaRegisterFunction (fatCubinHandle, hostFun, deviceFun,
				   deviceName, thread_limit, tid,
				   bid, bDim, gDim,
				   wSize);

	/* Return to caller */
	return;
}
